"""GPQA letter grading and EvalPlus MBPP+ execution with the official oracles."""

from __future__ import annotations

import ast
from functools import lru_cache
import json
import multiprocessing
import os
import re
import sys

from eval.baseline_reward import extract_last_boxed


def final_response(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""  # Truncated thinking is not a final answer.
    return text.strip()


def extract_choice(text: str) -> str | None:
    text = final_response(text)
    if r"\boxed{" in text:
        boxed = extract_last_boxed(text)
        if boxed is None:
            return None
        boxed = re.sub(r"\\(?:text|mathrm)\{([A-D])\}", r"\1", boxed)
        return boxed if boxed in "ABCD" and len(boxed) == 1 else None
    # Only an explicit final answer or a standalone letter on the final line.
    last_line = text.splitlines()[-1].strip() if text else ""
    match = re.fullmatch(
        r"(?:(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is\s*:?|:)\s*)?"
        r"\(?([A-D])\)?[.!]?", last_line, flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def score_gpqa(solution: str, truth: str) -> dict[str, float]:
    if truth not in ("A", "B", "C", "D"):
        raise ValueError(f"Invalid GPQA-Diamond answer label: {truth!r}")
    predicted = extract_choice(solution)
    score = float(predicted == truth)
    return {"score": score, "acc": score, "formatted": float(predicted is not None)}


@lru_cache(maxsize=384)
def mbpp_oracle(ground_truth: str):
    from evalplus.data.mbpp import mbpp_deserialize_inputs
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    task = json.loads(ground_truth)
    expected = {}
    for suite in ("base", "plus"):
        key = f"{suite}_input"
        task[key] = mbpp_deserialize_inputs(task["task_id"], task[key])
        expected[suite] = trusted_exec(
            task["prompt"] + task["canonical_solution"], task[key], task["entry_point"],
            record_time=True, output_not_none=task["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS,
        )
    return task, expected


def extract_code(text: str, entrypoint: str) -> str:
    from evalplus.sanitize import sanitize

    text = final_response(text)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    # Never choose a working snippet from earlier reasoning after a final failure.
    candidate = blocks[-1] if blocks else text
    if not candidate.strip():
        return ""
    return sanitize(candidate, entrypoint=entrypoint)


@lru_cache(maxsize=1)
def check_execution_runtime():
    """A broken execution environment must fail the run, not score every answer zero."""
    from evalplus.eval import PASS, untrusted_check

    status, details = untrusted_check(
        "mbpp", "def evalplus_runtime_probe(x):\n    return x", [[1]], "evalplus_runtime_probe",
        expected=[1], atol=0, ref_time=[0.001], fast_check=True,
    )
    if status != PASS:
        import psutil
        from evalplus.eval import query_maximum_memory_bytes

        raise RuntimeError(
            "EvalPlus failed its known-correct execution probe "
            f"(status={status!r}, details={list(details)!r}, platform={sys.platform}, "
            f"start_method={multiprocessing.get_start_method()}, pid={os.getpid()}, "
            f"worker_vms_bytes={psutil.Process().memory_info().vms}, "
            f"max_memory_bytes={query_maximum_memory_bytes()}). "
            "Check child-process stderr, startup time, subprocess permissions, and resource limits. "
            "Run python -m eval.reward_async --check in the Linux training environment."
        )


def score_mbpp(solution: str, ground_truth: str) -> dict[str, float]:
    from evalplus.eval import PASS, untrusted_check
    from evalplus.eval.utils import TimeoutException, time_limit

    check_execution_runtime()
    entrypoint = json.loads(ground_truth)["entry_point"]
    try:
        with time_limit(10):
            code = extract_code(solution, entrypoint)
    except (TimeoutException, RecursionError):
        return {"score": 0.0, "acc": 0.0, "formatted": 0.0}
    try:
        tree = ast.parse(code)
        formatted = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint
                        for node in tree.body)
    except (SyntaxError, ValueError, RecursionError):
        formatted = False
    if not formatted:
        return {"score": 0.0, "acc": 0.0, "formatted": 0.0}

    task, expected = mbpp_oracle(ground_truth)
    correct = True
    for suite in ("base", "plus"):
        outputs, timings = expected[suite]
        status, _ = untrusted_check(
            "mbpp", code, task[f"{suite}_input"], task["entry_point"],
            expected=outputs, atol=task["atol"], ref_time=timings, fast_check=True,
        )
        if status != PASS:
            correct = False
            break
    score = float(correct)
    return {"score": score, "acc": score, "formatted": 1.0}
