#!/usr/bin/env python3
"""Manually upload completed offline W&B runs after returning from SLURM."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

if __package__:
    from .runtime_env import ROOT, WANDB_ROOT, load_runtime_env
else:
    from runtime_env import ROOT, WANDB_ROOT, load_runtime_env


def training_run_root(directory: Path, records: list[Path]) -> Path | None:
    for record in records:
        run_id = record.stem.removeprefix("run-")
        reference = WANDB_ROOT / ".run_roots" / f"{run_id}.txt"
        if reference.is_file():
            run_root = Path(reference.read_text(encoding="utf-8").strip())
            if run_root.is_absolute() and (run_root / "launch_command.txt").is_file():
                return run_root.resolve()
    # Older logs remain supported in their original experiment directories.
    return next(
        (parent for parent in directory.parents if (parent / "launch_command.txt").is_file()),
        None,
    )


def completed_offline_runs(path: Path) -> list[Path]:
    path = path.resolve()
    candidates = (
        [path] if path.name.startswith("offline-run-")
        else sorted(set(path.rglob("offline-run-*")) | set(WANDB_ROOT.glob("offline-run-*")))
    )
    ready = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        records = list(directory.glob("run-*.wandb"))
        if not records or all(Path(str(record) + ".synced").exists() for record in records):
            continue
        run_root = training_run_root(directory, records)
        if not directory.is_relative_to(path) and (run_root is None or not run_root.is_relative_to(path)):
            continue
        if run_root is None or not (run_root / "training_finished.json").is_file():
            print(f"Skipping unfinished run: {directory}")
            continue
        ready.append(directory)
    return ready


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=WANDB_ROOT,
                        help="training run, W&B offline-run directory, or output tree")
    parser.add_argument("--entity", help="override WANDB_ENTITY from the repository .env")
    parser.add_argument("--project", help="override WANDB_PROJECT from the repository .env")
    parser.add_argument("--dry-run", action="store_true", help="print uploads without connecting to W&B")
    args = parser.parse_args(argv)
    env_path = ROOT / ".env"
    if not env_path.is_file():
        parser.error(f"Required configuration file does not exist: {env_path}")
    sync_config = dotenv_values(env_path, interpolate=False)
    for key in ("WANDB_ENTITY", "WANDB_PROJECT"):
        if not (sync_config.get(key) or "").strip():
            parser.error(f"{key} must be set to a non-empty value in {env_path}")
    entity = args.entity or sync_config["WANDB_ENTITY"].strip()
    project = args.project or sync_config["WANDB_PROJECT"].strip()
    env = load_runtime_env()
    if env.get("SLURM_JOB_ID") and not args.dry_run:
        parser.error("Sync after the SLURM job finishes, from the login node outside the job")
    path = args.path.resolve() if args.path.is_absolute() else (ROOT / args.path).resolve()
    if not path.is_dir():
        parser.error(f"Output directory does not exist: {path}")
    runs = completed_offline_runs(path)
    if not runs:
        print("No completed, unsynced offline W&B runs found.")
        return 0
    # Only this explicit upload command uses online mode. Training stays offline.
    env["WANDB_MODE"] = "online"
    env["WANDB_ENTITY"] = entity
    env["WANDB_PROJECT"] = project
    exit_code = 0
    for directory in runs:
        command = [sys.executable, "-m", "wandb", "sync", "--entity", entity, "--project", project,
                   "--include-offline", "--no-include-online", "--no-include-synced", "--mark-synced"]
        command.append(str(directory))
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            exit_code = result.returncode or exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
