"""Manage completed checkpoint directories without changing the trainer."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


STEP_DIRECTORY = re.compile(r"global_step_(0|[1-9][0-9]*)")
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def latest_checkpoint(run_dir: Path) -> Path | None:
    try:
        step = int((run_dir / TRACKER_FILENAME).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    checkpoint = run_dir / f"global_step_{step}"
    if (
        step < 0
        or checkpoint.is_symlink()
        or not (checkpoint / "actor").is_dir()
        or not (checkpoint / "data.pt").is_file()
    ):
        return None
    return checkpoint


def resume_paths(value: Path) -> tuple[Path, Path]:
    if not value.is_absolute():
        raise ValueError("--resume requires an absolute run or global_step directory")
    path = value.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {path}")
    if STEP_DIRECTORY.fullmatch(path.name):
        run_dir, checkpoint = path.parent, path
    else:
        run_dir, checkpoint = path, latest_checkpoint(path)
    if checkpoint is None or not (checkpoint / "data.pt").is_file():
        raise ValueError(f"No completed training checkpoint found: {path}")
    actor = checkpoint / "actor"
    for component in ("model", "optim", "extra_state"):
        if not any(actor.glob(f"{component}_world_size_*_rank_*.pt")):
            raise ValueError(
                f"Checkpoint lacks {component} shards: {checkpoint}; "
                "resuming requires model, optimizer, extra state, and data.pt"
            )
    return run_dir, checkpoint


def prune_checkpoints(run_dir: Path, keep: int) -> list[Path]:
    """Prune older steps only after verl publishes its completion tracker.

    Leave newer directories alone: they may contain an in-progress save.
    The tracker is written after all actor files and data.pt have been saved.
    """
    if keep < 1:
        raise ValueError("Checkpoint retention must be at least one")
    latest = latest_checkpoint(run_dir)
    if latest is None:
        return []
    latest_step = int(STEP_DIRECTORY.fullmatch(latest.name).group(1))
    checkpoints = []
    for path in run_dir.iterdir():
        match = STEP_DIRECTORY.fullmatch(path.name)
        if match and not path.is_symlink() and path.is_dir():
            step = int(match.group(1))
            if step <= latest_step:
                checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    removed = []
    for _, path in checkpoints[keep:]:
        shutil.rmtree(path)
        removed.append(path)
    return removed
