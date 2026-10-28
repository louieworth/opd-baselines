"""Load the repository's private .env without overriding exported variables."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_ENTITY = "li-jiang-research"
WANDB_ROOT = ROOT / "wandb"


def load_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in dotenv_values(ROOT / ".env", interpolate=False).items():
        if value is not None:
            env.setdefault(key, value)
    return env


def configure_wandb_paths(env: dict[str, str], run_dir: Path, use_wandb: bool) -> None:
    # The SDK appends wandb/ to WANDB_DIR; use the repository as its root.
    env["WANDB_DIR"] = str(WANDB_ROOT.parent)
    env["WANDB_USE_DOT_WANDB"] = "false"
    env["WANDB_DATA_DIR"] = str(WANDB_ROOT / "staging")
    env["WANDB_CACHE_DIR"] = str(WANDB_ROOT / "cache")
    env["WANDB_ARTIFACT_DIR"] = str(WANDB_ROOT / "artifacts")
    for name in ("WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_ARTIFACT_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    if use_wandb:
        run_id = env.get("WANDB_RUN_ID") or uuid.uuid4().hex
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("WANDB_RUN_ID must be a single filename component")
        env["WANDB_RUN_ID"] = run_id
        # Keep the completion marker with model outputs and map this log ID to it.
        index = WANDB_ROOT / ".run_roots"
        index.mkdir(parents=True, exist_ok=True)
        (index / f"{run_id}.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
