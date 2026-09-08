"""Persistent, clean Python process for EvalPlus, independent of Ray's entrypoint."""

import atexit
from contextlib import redirect_stdout
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback


_process = None


def shutdown():
    global _process
    if _process is None:
        return
    process, _process = _process, None
    try:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass  # A failed request may leave buffered input after worker exit.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        process.stdout.close()


atexit.register(shutdown)


def compute_score(solution: str, ground_truth: str):
    """Serialized by reward_async's single I/O thread; grading runs in a main thread."""
    global _process
    if _process is None:
        env = os.environ.copy()
        # Set these before NumPy is imported. Large BLAS thread pools can exhaust
        # EvalPlus's address-space limit even for a trivial correct program.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        _process = subprocess.Popen(
            [sys.executable, "-u", "-m", "eval.mbpp_worker"],
            cwd=Path(__file__).resolve().parents[1], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
    try:
        _process.stdin.write(json.dumps([solution, ground_truth]) + "\n")
        _process.stdin.flush()
        line = _process.stdout.readline()
    except (BrokenPipeError, OSError) as exc:
        raise RuntimeError("EvalPlus worker communication failed; inspect worker stderr.") from exc
    if not line:
        raise RuntimeError(
            f"EvalPlus worker exited without a score (exitcode={_process.poll()}); inspect worker stderr."
        )
    response = json.loads(line)
    if "error" in response:
        raise RuntimeError(f"EvalPlus worker failed:\n{response['error']}")
    return response["result"]


def _serve():
    # Only this standalone CPU worker uses fork, never the Ray/CUDA process.
    # No Ray imports, model state, or math-grader dependencies enter this worker.
    if sys.platform == "linux":
        multiprocessing.set_start_method("fork", force=True)
    from eval.code_science_reward import score_mbpp

    for line in sys.stdin:
        try:
            solution, ground_truth = json.loads(line)
            # Keep library diagnostics out of the JSON response channel.
            with redirect_stdout(sys.stderr):
                result = score_mbpp(solution, ground_truth)
            response = {"result": result}
        except Exception:
            response = {"error": traceback.format_exc()}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    _serve()
