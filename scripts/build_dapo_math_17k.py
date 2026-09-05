#!/usr/bin/env python3
"""Build the pinned DAPO-Math-17k parquet used by the baseline configs."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_NAME = "dapo_math_17k"
SOURCE_REPO = "thunlp/OPD"
SOURCE_COMMIT = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"
SOURCE_PATH = "datasets/dapo-math-17k-processed.parquet"
SOURCE_URL = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/{SOURCE_PATH}"
SOURCE_DATA_SOURCE = "math_dapo"
SOURCE_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{{}}."
NORMALIZED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("data/train/dapo_math_17k.parquet")


def convert_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for index, row in enumerate(rows):
        if row.get("data_source") != SOURCE_DATA_SOURCE:
            raise ValueError(f"row {index}: unexpected data_source {row.get('data_source')!r}")
        prompt = row.get("prompt") or []
        if len(prompt) != 1 or prompt[0].get("role") != "user":
            raise ValueError(f"row {index}: expected one user message")
        content = str(prompt[0].get("content", ""))
        if not content.endswith(SOURCE_SUFFIX):
            raise ValueError(f"row {index}: prompt does not match the pinned release format")
        question = content[: -len(SOURCE_SUFFIX)].strip()
        ground_truth = str((row.get("reward_model") or {}).get("ground_truth", "")).strip()
        if not question or not ground_truth:
            raise ValueError(f"row {index}: empty question or ground truth")
        source_index = str((row.get("extra_info") or {}).get("index", index))
        converted.append(
            {
                "prompt": [{"role": "user", "content": question + NORMALIZED_SUFFIX}],
                "reward_model": {"ground_truth": ground_truth, "style": "rule-lighteval/MATH_v2"},
                "data_source": DATASET_NAME,
                "ability": "math",
                "extra_info": {"index": source_index, "problem": question, "solution": ""},
            }
        )
    return converted


def fetch_source(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"dapo-math-17k-processed.{SOURCE_COMMIT[:12]}.parquet"
    if target.exists():
        return target
    temporary = target.with_suffix(".tmp")
    print(f"downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, temporary)
    os.replace(temporary, target)
    return target


def build(source: Path, output: Path) -> Path:
    rows = pq.read_table(source).to_pylist()
    converted = convert_rows(rows)
    normalize = lambda text: "".join(text.casefold().split())
    heldout = set()
    eval_files = [ROOT / f"data/eval/{name}.parquet" for name in ("aime25", "aime26", "hmmt26", "amobench")]
    for path in eval_files:
        if path.is_file():
            heldout.update(normalize(row["extra_info"]["problem"]) for row in pq.read_table(path).to_pylist())
    converted = [row for row in converted if normalize(row["extra_info"]["problem"]) not in heldout]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(converted), temporary)
    os.replace(temporary, output)
    print(f"wrote {output} ({len(converted):,} rows)")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local source parquet; skips download")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/thunlp_opd"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    os.chdir(ROOT)
    source = args.source if args.source else fetch_source(args.cache_dir)
    build(source, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
