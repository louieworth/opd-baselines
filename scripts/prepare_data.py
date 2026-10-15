#!/usr/bin/env python3
"""Prepare portable train/evaluation parquet files from pinned sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import urllib.request
import zipfile
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
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet"),
    # The authors also publish this archive publicly, without the HF access gate.
    "gpqa_diamond": ("idavidrein/gpqa", "d46dc8d5e01b40bcde0bed6bee68a5de953a58f8", "dataset.zip"),
    "amc23": ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057", "test-00000-of-00001.parquet"),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
}
MMLU_SUBSETS = {
    "mmlu_college_chemistry": "college_chemistry",
    "mmlu_college_medicine": "college_medicine",
}
EXPECTED_EVAL_ROWS = {
    "mmlu": 14042, "mmlu_college_chemistry": 100, "mmlu_college_medicine": 173,
    "gpqa_diamond": 198, "amc23": 40, "math500": 500, "gsm8k": 1319,
}
GPQA_SHUFFLE_SEED = 42
CHOICE_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer "
    "(only the choice letter A, B, C, or D) within \\boxed{}."
)


def choice_prompt(question: str, choices: list[str]) -> str:
    if len(choices) != 4 or any(not str(choice).strip() for choice in choices):
        raise ValueError("Expected four nonempty answer choices")
    return question + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    ) + CHOICE_SUFFIX


def convert_eval(name: str, rows: list[dict]) -> list[dict]:
    result = []
    rng = random.Random(GPQA_SHUFFLE_SEED)
    for i, row in enumerate(rows):
        if name in MMLU_SUBSETS and row["subject"] != MMLU_SUBSETS[name]:
            continue
        ability = "math"
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
        elif name == "mmlu" or name in MMLU_SUBSETS:
            question = row["question"].strip()
            prompt = choice_prompt(question, row["choices"])
            gold_index = int(row["answer"])
            if not 0 <= gold_index < 4:
                raise ValueError(f"mmlu/{i}: invalid answer index {gold_index}")
            answer = "ABCD"[gold_index]
            qid = f"{row['subject']}/{i}"
            ability = "general"
        elif name == "gpqa_diamond":
            question = row["Question"].strip()
            choices = [row["Correct Answer"].strip()] + [
                row[f"Incorrect Answer {j}"].strip() for j in range(1, 4)
            ]
            order = list(range(4))
            rng.shuffle(order)
            prompt = choice_prompt(question, [choices[j] for j in order])
            answer = "ABCD"[order.index(0)]
            qid = row.get("Record ID", i)
            ability = "science"
        else:
            question_key = "question" if name in {"amc23", "gsm8k"} else "problem"
            question = row[question_key].strip()
            prompt = question + NORMALIZED_SUFFIX
            answer = str(row.get("answer", row.get("solution", ""))).strip()
            if name == "gsm8k":
                if "####" not in answer:
                    raise ValueError(f"gsm8k/{i}: missing final-answer separator")
                answer = answer.rsplit("####", 1)[1].strip()
            if answer.startswith("\\boxed{") and answer.endswith("}"):
                answer = answer[7:-1]
            qid = row.get("id", row.get("unique_id", row.get("problem_idx", i)))
        if not question or not answer:
            raise ValueError(f"{name}/{i}: empty question or answer")
        result.append({
            "data_source": name, "ability": ability,
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": answer, "style": "rule"},
            "extra_info": {"index": str(qid), "problem": question, "solution": ""},
        })
    return result


def read_eval_rows(name: str, raw: Path) -> list[dict]:
    if name == "gpqa_diamond":
        # Read only the Diamond member; do not extract other benchmark splits.
        with zipfile.ZipFile(raw) as archive:
            members = [p for p in archive.namelist() if Path(p).name == "gpqa_diamond.csv"]
            if len(members) != 1:
                raise ValueError("Expected exactly one gpqa_diamond.csv in the authors' archive")
            text = archive.read(members[0], pwd=b"deserted-untie-orchid").decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
    if raw.suffix == ".parquet":
        return pq.read_table(raw).to_pylist()
    return [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild from pinned public sources")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="absolute dataset output root")
    parser.add_argument("--eval-only", action="store_true", help="prepare evaluation files without rebuilding training data")
    parser.add_argument("--eval-datasets", nargs="+", choices=list(SOURCES) + list(MMLU_SUBSETS), default=list(SOURCES))
    args = parser.parse_args(argv)
    if not args.data_dir.is_absolute():
        parser.error("--data-dir must be an absolute path")
    eval_rebuilt = False
    for name in args.eval_datasets:
        source_name = "mmlu" if name in MMLU_SUBSETS else name
        repo, revision, filename = SOURCES[source_name]
        out = args.data_dir / "eval" / f"{name}.parquet"
        if out.is_file() and not args.rebuild:
            print(f"Using {out}")
            continue
        raw = args.data_dir / "raw" / source_name / revision[:12] / Path(filename).name
        raw.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}"
            if name == "gpqa_diamond"
            else f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
        )
        if not raw.exists():
            print(f"Downloading {repo}@{revision}")
            tmp = raw.with_suffix(raw.suffix + ".tmp")
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as target:
                shutil.copyfileobj(response, target)
            os.replace(tmp, raw)
        rows = read_eval_rows(name, raw)
        converted = convert_eval(name, rows)
        expected = EXPECTED_EVAL_ROWS.get(name)
        if expected is not None and len(converted) != expected:
            raise ValueError(f"{name}: expected {expected} questions, got {len(converted)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(converted), tmp, compression="zstd")
        os.replace(tmp, out)
        source = {"repo": repo, "revision": revision, "url": url, "num_questions": len(converted),
                  "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
        if name in MMLU_SUBSETS:
            source["subset"] = MMLU_SUBSETS[name]
        if name == "gpqa_diamond":
            source.update({"subset": "gpqa_diamond", "shuffle_seed": GPQA_SHUFFLE_SEED})
        out.with_suffix(".source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({len(converted):,} rows)")
        eval_rebuilt = True
    if args.eval_only:
        return 0
    train = args.data_dir / "train/dapo_math_17k.parquet"
    if args.rebuild or eval_rebuilt or not train.is_file():
        build(fetch_source(args.data_dir / "raw/thunlp_opd"), train)
    else:
        print(f"Using {train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
