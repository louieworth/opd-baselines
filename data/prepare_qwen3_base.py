#!/usr/bin/env python3
"""Prepare Qwen3-4B-Base for ChatML rollout stop/pad semantics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


SOURCE = "Qwen/Qwen3-4B-Base"
REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
MANIFEST = "PREPARED.json"
REWRITTEN = {"tokenizer_config.json", "generation_config.json", "config.json"}
EOS_TOKEN = "<|im_end|>"
NATIVE_EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|fim_pad|>"
ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def weight_files(snapshot: Path) -> list[str]:
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set((read_json(index).get("weight_map") or {}).values()))
    elif (snapshot / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"{snapshot}: no safetensors weights found")
    missing = [name for name in names if not (snapshot / name).is_file()]
    if not names or missing:
        raise FileNotFoundError(f"{snapshot}: incomplete weight set; missing={missing}")
    return names


def token_ids(snapshot: Path) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if not tokenizer.chat_template or EOS_TOKEN not in tokenizer.chat_template:
        raise ValueError(f"{snapshot}: expected a ChatML template containing {EOS_TOKEN}")
    unknown = tokenizer.unk_token_id
    ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in (EOS_TOKEN, NATIVE_EOS_TOKEN, PAD_TOKEN)
    }
    missing = [token for token, token_id in ids.items() if token_id is None or token_id == unknown]
    if missing or len(set(ids.values())) != len(ids):
        raise ValueError(f"required stop/pad tokens are missing or aliased: ids={ids}, missing={missing}")
    return {token: int(token_id) for token, token_id in ids.items()}


def validate_target(target: Path, ids: dict[str, int]) -> None:
    weight_files(target)
    tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
    if tokenizer.eos_token_id != ids[EOS_TOKEN] or tokenizer.pad_token_id != ids[PAD_TOKEN]:
        raise RuntimeError(
            f"prepared tokenizer resolves eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}; "
            f"expected eos={ids[EOS_TOKEN]}, pad={ids[PAD_TOKEN]}"
        )


def prepare(snapshot: Path, target: Path, source: str, revision: str, force: bool) -> None:
    snapshot = snapshot.resolve()
    target = target.resolve()
    if target == snapshot or snapshot in target.parents or target in snapshot.parents:
        raise ValueError("target must not overlap the Hugging Face cache snapshot")
    if target.is_symlink():
        raise FileExistsError(f"{target}: refusing to replace a symlink")
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} exists; pass --force to replace a prior prepared output")
        marker = target / MANIFEST
        if not marker.is_file() or read_json(marker).get("prepared_by") != Path(__file__).name:
            raise FileExistsError(f"{target}: not recognized as an output of this script; refusing to replace it")

    for name in REWRITTEN:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"{snapshot}: missing {name}")
    weights = weight_files(snapshot)
    ids = token_ids(snapshot)

    tokenizer_cfg = read_json(snapshot / "tokenizer_config.json")
    generation_cfg = read_json(snapshot / "generation_config.json")
    model_cfg = read_json(snapshot / "config.json")
    tokenizer_cfg.update({"eos_token": EOS_TOKEN, "pad_token": PAD_TOKEN})
    generation_cfg.update(
        {
            "eos_token_id": [ids[EOS_TOKEN], ids[NATIVE_EOS_TOKEN]],
            "pad_token_id": ids[PAD_TOKEN],
        }
    )
    model_cfg["eos_token_id"] = ids[EOS_TOKEN]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    backup: Path | None = None
    try:
        for entry in snapshot.iterdir():
            if entry.name in REWRITTEN or entry.name == MANIFEST or entry.name.startswith("."):
                continue
            os.symlink(entry, staging / entry.name)
        write_json(staging / "tokenizer_config.json", tokenizer_cfg)
        write_json(staging / "generation_config.json", generation_cfg)
        write_json(staging / "config.json", model_cfg)
        write_json(
            staging / MANIFEST,
            {
                "prepared_by": Path(__file__).name,
                "source": source,
                "revision": revision,
                "snapshot": str(snapshot),
                "weight_files": weights,
                "eos_token": EOS_TOKEN,
                "eos_token_id": ids[EOS_TOKEN],
                "generation_eos_token_ids": generation_cfg["eos_token_id"],
                "pad_token": PAD_TOKEN,
                "pad_token_id": ids[PAD_TOKEN],
            },
        )
        validate_target(staging, ids)
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
            os.replace(target, backup)
        os.replace(staging, target)
        validate_target(target, ids)
    except BaseException:
        if target.exists() and backup is not None and backup.exists():
            shutil.rmtree(target)
            os.replace(backup, target)
        elif target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--target", type=Path, default=Path("models/Qwen3-4B-Base-chatml"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    snapshot = Path(
        snapshot_download(
            repo_id=args.source,
            revision=args.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
    )
    prepare(snapshot, args.target, args.source, args.revision, args.force)
    print(f"prepared {args.target}")
    return 0


def ensure_prepared(target: Path) -> None:
    """Download once on the training platform, then reuse validated local weights."""
    marker = target / MANIFEST
    if marker.is_file():
        manifest = read_json(marker)
        if manifest.get("source") != SOURCE or manifest.get("revision") != REVISION:
            raise ValueError(f"{target}: prepared model does not match the pinned source")
        validate_target(target, token_ids(target))
        print(f"Reusing prepared student: {target.name}")
        return
    if target.exists():
        raise FileExistsError(f"{target}: unrecognized model directory; set MODEL_PATH to use it explicitly")
    snapshot = Path(snapshot_download(repo_id=SOURCE, revision=REVISION))
    prepare(snapshot, target, SOURCE, REVISION, False)


if __name__ == "__main__":
    raise SystemExit(main())
