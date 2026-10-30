#!/usr/bin/env python3
"""Mirror local directories to the same absolute path on every alive Ray node.

A long-lived Ray cluster without a shared POSIX filesystem needs the repository
and model weights at identical paths on each node. Files whose size and mtime
already match are skipped, so rerunning after an edit copies only the changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
# Run artifacts directly under the mirrored root, not packages that share these names.
SKIP_TOP_DIRS = {"outputs", "wandb", "tmp"}


def local_files(root: Path) -> dict[str, tuple[int, int]]:
    files = {}
    for directory, dirs, names in os.walk(root):
        top = Path(directory) == root
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not (top and name in SKIP_TOP_DIRS)]
        for name in names:
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                stat = path.stat()
                files[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return files


@ray.remote(num_cpus=0)
def remote_files(root: str) -> dict[str, tuple[int, int]]:
    return local_files(Path(root)) if Path(root).is_dir() else {}


@ray.remote(num_cpus=0)
def write_file(root: str, relative: str, data: bytes, mode: int, mtime_ns: int) -> None:
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".sync-tmp")
    temporary.write_bytes(data)
    os.chmod(temporary, mode)
    os.utime(temporary, ns=(mtime_ns, mtime_ns))
    os.replace(temporary, path)


def sync(root: Path, nodes: list[dict]) -> None:
    source = local_files(root)
    for node in nodes:
        pin = NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
        existing = ray.get(remote_files.options(scheduling_strategy=pin).remote(str(root)))
        stale = [name for name, meta in source.items() if existing.get(name) != meta]
        size = sum(source[name][0] for name in stale)
        print(f"{node['NodeManagerAddress']}: {root} -> {len(stale)} files, {size / 2**30:.2f} GiB", flush=True)
        pending = []
        for name in stale:
            path = root / name
            ref = ray.put(path.read_bytes())
            pending.append(write_file.options(scheduling_strategy=pin).remote(
                str(root), name, ref, path.stat().st_mode & 0o7777, source[name][1]))
            if len(pending) >= 8:
                done, pending = ray.wait(pending, num_returns=len(pending) - 4)
                ray.get(done)
        ray.get(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="absolute directories to mirror")
    args = parser.parse_args()
    roots = [path.resolve() for path in args.paths]
    if any(not path.is_dir() for path in roots):
        parser.error("every path must be an existing directory")
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), logging_level="ERROR")
    here = ray.get_runtime_context().get_node_id()
    nodes = [node for node in ray.nodes() if node["Alive"] and node["NodeID"] != here]
    for root in roots:
        sync(root, nodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
