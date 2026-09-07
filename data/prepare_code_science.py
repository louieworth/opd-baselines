#!/usr/bin/env python3
"""Prepare pinned MBPP+ and GPQA-Diamond data, including offline grading inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import urllib.request
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

if __package__:
    from .prompt_modes import CODE_ANSWER_INSTRUCTION, CHOICE_ANSWER_INSTRUCTION
else:
    from prompt_modes import CODE_ANSWER_INSTRUCTION, CHOICE_ANSWER_INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]
GPQA_REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
SOURCES = {
    "mbppplus": {
        "filename": "MbppPlus-v0.2.0.jsonl.gz",
        "url": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
        "sha256": "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63",
        "count": 378,
    },
    "gpqa_diamond": {
        "filename": "gpqa-dataset.zip",
        "url": f"https://raw.githubusercontent.com/idavidrein/gpqa/{GPQA_REVISION}/dataset.zip",
        "sha256": "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc",
        "count": 198,
    },
}


def fetch_source(name: str, raw_dir: Path) -> Path:
    source = SOURCES[name]
    path = raw_dir / source["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {name} from its pinned official source")
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"{name}: download SHA256 mismatch")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError(f"{path}: source SHA256 mismatch")
    return path


def eval_row(source, ability, qid, prompt, truth):
    return {
        "data_source": source,
        "ability": ability,
        "prompt": [{"role": "user", "content": prompt}],
        "reward_model": {"ground_truth": truth, "style": "rule"},
        "extra_info": {"index": str(qid), "problem": prompt, "solution": ""},
    }


def convert_mbpp(rows: list[dict]) -> list[dict]:
    result = []
    for task in rows:
        for key in ("task_id", "prompt", "entry_point", "canonical_solution", "base_input"):
            if not task.get(key):
                raise ValueError(f"MBPP+: missing {key}")
        # The pinned release encodes Mbpp/793's empty extra suite as {}.
        if not isinstance(task.get("plus_input"), list) and task.get("plus_input") != {}:
            raise ValueError("MBPP+: missing plus_input list")
        prompt = (
            "Write Python code to solve the following task. Include the required function "
            f"`{task['entry_point']}` and any imports or helper functions.\n\n"
            f"{task['prompt'].strip()}\n\n{CODE_ANSWER_INSTRUCTION}"
        )
        # Reference code and hidden tests are only supplied to the CPU grader.
        result.append(eval_row("mbppplus", "code", task["task_id"], prompt,
                               json.dumps(task, ensure_ascii=False, sort_keys=True)))
    return result


def convert_gpqa(rows: list[dict], seed: int = 0) -> list[dict]:
    result = []
    for i, row in enumerate(rows):
        qid = str(row.get("Record ID") or i)
        question = row["Question"].strip()
        choices = [row[key].strip() for key in (
            "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3",
        )]
        # Preserve the official distractors, including two rows with repeated
        # incorrect options. Shuffle indices so the answer mapping stays exact.
        if not question or not all(choices) or choices[0] in choices[1:]:
            raise ValueError(f"GPQA-Diamond/{qid}: empty question or invalid choices")
        order = list(range(4))
        random.Random(f"gpqa-diamond:{seed}:{qid}").shuffle(order)
        options = "\n".join(f"{letter}. {choices[j]}" for letter, j in zip("ABCD", order, strict=True))
        prompt = f"{question}\n\n{options}\n\n{CHOICE_ANSWER_INSTRUCTION}"
        result.append(eval_row("gpqa_diamond", "science", qid, prompt, "ABCD"[order.index(0)]))
    return result


def prepare(names=tuple(SOURCES), *, rebuild=False, seed=0, raw_dir=None, output_dir=None):
    raw_dir = Path(raw_dir) if raw_dir is not None else ROOT / "data/raw/code_science"
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "data/eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = output_dir / f"{name}.parquet"
        if out.exists() and not rebuild:
            print(f"Using {out}")
            continue
        raw = fetch_source(name, raw_dir)
        if name == "mbppplus":
            with gzip.open(raw, "rt", encoding="utf-8") as handle:
                rows = convert_mbpp([json.loads(line) for line in handle if line.strip()])
        else:
            # The authors publish this password in their README for dataset access.
            with zipfile.ZipFile(raw) as archive:
                content = archive.read("dataset/gpqa_diamond.csv", pwd=b"deserted-untie-orchid")
            rows = convert_gpqa(list(csv.DictReader(io.StringIO(content.decode("utf-8-sig")))), seed)
        if len(rows) != SOURCES[name]["count"] or len({r["extra_info"]["index"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: incorrect row count or duplicate IDs")
        table = pa.Table.from_pylist(rows)
        table = table.replace_schema_metadata({
            b"source_url": SOURCES[name]["url"].encode(),
            b"source_sha256": SOURCES[name]["sha256"].encode(),
            b"gpqa_shuffle_seed": str(seed).encode(),
        })
        temporary = out.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, out)
        print(f"Wrote {out} ({len(rows)} questions)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="GPQA option shuffle seed; use --rebuild to change")
    args = parser.parse_args()
    prepare(args.benchmarks, rebuild=args.rebuild, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
