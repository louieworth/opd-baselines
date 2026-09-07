#!/usr/bin/env python3
"""Prepare portable train/evaluation parquet files from pinned sources."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__:
    from .build_dapo_math_17k import ROOT, NORMALIZED_SUFFIX, build, fetch_source
else:
    from build_dapo_math_17k import ROOT, NORMALIZED_SUFFIX, build, fetch_source

SOURCES = {
    "aime25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl"),
    "aime26": ("math-ai/aime26", "79037aebdb6580008fb960d17cb21fd3099083e3", "aime2026.jsonl"),
    "hmmt26": ("MathArena/hmmt_feb_2026", "02fba4f74d8e68e73e66a02d540fd979c05c274c", "data/train-00000-of-00001.parquet"),
    "amobench": ("meituan-longcat/AMO-Bench", "2f422616c25d862984408fbbfaed63a961e8e025", "data/test-00000-of-00001.parquet"),
}


def convert_eval(name: str, rows: list[dict]) -> list[dict]:
    result = []
    for i, row in enumerate(rows):
        if name == "amobench":
            if row["answer_type"] == "description":
                continue
            if row["answer_type"] not in {"number", "set", "variable"}:
                raise ValueError(f"Unknown AMO answer type: {row['answer_type']}")
            question = prompt = row["prompt"].strip()
            qid = row["question_id"]
            truth = {"answer": row["answer"], "answer_type": row["answer_type"], "question_id": qid}
            if qid == 5:
                truth["try_list"] = [f"n={n}" for n in range(1, 21)]
            elif qid == 37:
                truth["try_list"] = [f"a={n},b={n+1},c={n+2}" for n in range(2, 19)]
            if row["answer_type"] == "variable" and not truth.get("try_list"):
                raise ValueError(f"Missing variable probes for AMO question {qid}")
            answer = json.dumps(truth, ensure_ascii=False, sort_keys=True)
        else:
            question = row["problem"].strip()
            prompt = question + NORMALIZED_SUFFIX
            answer = str(row.get("answer", row.get("solution", ""))).strip()
            if answer.startswith("\\boxed{") and answer.endswith("}"):
                answer = answer[7:-1]
            qid = row.get("id", row.get("problem_idx", i))
        if not question or not answer:
            raise ValueError(f"{name}/{i}: empty question or answer")
        result.append({
            "data_source": name, "ability": "math",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"index": str(qid), "problem": question, "solution": ""},
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild from pinned public sources")
    args = parser.parse_args()
    os.chdir(ROOT)
    eval_rebuilt = False
    for name, (repo, revision, filename) in SOURCES.items():
        out = Path(f"data/eval/{name}.parquet")
        if out.is_file() and not args.rebuild:
            print(f"Using {out}")
            continue
        raw = Path("data/raw") / name / Path(filename).name
        raw.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        if not raw.exists():
            print(f"Downloading {repo}@{revision}")
            tmp = raw.with_suffix(raw.suffix + ".tmp")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, raw)
        rows = (pq.read_table(raw).to_pylist() if raw.suffix == ".parquet"
                else [json.loads(line) for line in raw.read_text().splitlines() if line.strip()])
        converted = convert_eval(name, rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(converted), tmp, compression="zstd")
        os.replace(tmp, out)
        print(f"Wrote {out} ({len(converted):,} rows)")
        eval_rebuilt = True
    train = Path("data/train/dapo_math_17k.parquet")
    if args.rebuild or eval_rebuilt or not train.is_file():
        build(fetch_source(Path("data/raw/thunlp_opd")), train)
    else:
        print(f"Using {train}")
    if __package__:
        from .prepare_code_science import prepare
    else:
        from prepare_code_science import prepare
    prepare(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
