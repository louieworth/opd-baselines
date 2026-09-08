"""Run math, science, and code grading in separate process main threads."""

import asyncio
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from eval.baseline_reward import compute_score as score_sync
from eval import mbpp_worker

_pool = None
_code_pool = None


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    global _code_pool
    if data_source == "mbppplus":
        # This thread only transports requests. EvalPlus and SIGALRM execute in
        # a clean process main thread, without nested spawn reimporting Ray.
        if _code_pool is None:
            _code_pool = ThreadPoolExecutor(max_workers=1)
            atexit.register(_code_pool.shutdown)
        return await asyncio.get_running_loop().run_in_executor(
            _code_pool, partial(mbpp_worker.compute_score, solution_str, ground_truth)
        )
    # verl's synchronous reward functions run in threads, where SIGALRM-based
    # Math-Verify parsing fails. A spawned process has an independent main thread.
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        atexit.register(_pool.shutdown)
    return await asyncio.get_running_loop().run_in_executor(
        _pool, partial(score_sync, data_source, solution_str, ground_truth, extra_info)
    )


async def check_mbpp_runtime():
    """Exercise the production reward path before allocating training workers."""
    import json

    truth = json.dumps({
        "task_id": "Mbpp/9999", "prompt": "", "entry_point": "identity",
        "canonical_solution": "def identity(x):\n    return x",
        "base_input": [[1]], "plus_input": [[-1]], "atol": 0.0,
    })
    for code, expected in (("def identity(x):\n    return x", 1.0),
                           ("def identity(x):\n    return abs(x)", 0.0)):
        result = await compute_score("mbppplus", code, truth)
        if result["acc"] != expected:
            raise RuntimeError(f"EvalPlus reward self-check expected {expected}, got {result!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check EvalPlus through the production reward worker.")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    asyncio.run(check_mbpp_runtime())
    print("EvalPlus reward self-check passed (correct solution accepted, plus-suite failure rejected).")
