"""Run math, science, and code grading in separate process main threads."""

import asyncio
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from eval.baseline_reward import compute_score as score_sync

_pool = None


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # verl's synchronous reward functions run in threads, where SIGALRM-based
    # Math-Verify parsing fails. A spawned process has an independent main thread.
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        atexit.register(_pool.shutdown)
    return await asyncio.get_running_loop().run_in_executor(
        _pool, partial(score_sync, data_source, solution_str, ground_truth, extra_info)
    )
