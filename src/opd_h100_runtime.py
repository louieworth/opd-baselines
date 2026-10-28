"""Isolated H100 experiment hooks; existing recipes never import this module."""

from functools import wraps
import json
import os
from pathlib import Path
import time


def detach_report(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_report(item) for item in value)
    return value


def record(kind, **values):
    import torch
    root = os.environ.get("OPD_H100_RECORD_DIR")
    if not root or not torch.cuda.is_available():
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    row = dict(kind=kind, rank=rank, time=time.time(),
               allocated_gib=torch.cuda.memory_allocated() / 2**30,
               reserved_gib=torch.cuda.memory_reserved() / 2**30, **values)
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"rank-{rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def install():
    import torch
    from src import opd_runtime
    opd_runtime.install()
    from verl.workers import engine_workers
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    if getattr(FSDPEngineWithLMHead, "_opd_h100_installed", False):
        return
    original_forward = FSDPEngineWithLMHead.forward_step

    @wraps(original_forward)
    def forward_step(self, *args, **kwargs):
        self._h100_micro = getattr(self, "_h100_micro", 0) + 1
        if torch.cuda.is_available() and (self._h100_micro - 1) % 8 == 0:
            record("micro_start", micro=self._h100_micro)
        loss, output = original_forward(self, *args, **kwargs)
        # Keep the loss graph intact. Only reporting copies go into output_lst.
        output = dict(output)
        output["model_output"] = detach_report(output.get("model_output", {}))
        output["metrics"] = detach_report(output.get("metrics", {}))
        return loss, output

    FSDPEngineWithLMHead.forward_step = forward_step
    original_update = engine_workers.ActorRolloutRefWorker.update_actor

    @wraps(original_update)
    def update_actor(self, data, *args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._h100_update = getattr(self, "_h100_update", 0) + 1
        tokens = int(data["input_ids"].offsets().diff().sum().item())
        started = time.monotonic()
        try:
            output = original_update(self, data, *args, **kwargs)
        except BaseException as exc:
            record("actor_error", update=self._h100_update, error=type(exc).__name__,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            raise
        torch.cuda.synchronize()
        record("actor_update", update=self._h100_update, tokens=tokens,
               seconds=time.monotonic() - started,
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        return output

    engine_workers.ActorRolloutRefWorker.update_actor = update_actor
    original_weights = engine_workers.ActorRolloutRefWorker.update_weights

    @wraps(original_weights)
    async def update_weights(self, *args, **kwargs):
        opd_runtime._release_teacher_heads(opd_runtime.runtime_options())
        if torch.distributed.is_initialized():
            # TP peers must finish releasing their actor caches before any
            # replica restores weights on the peers' devices.
            torch.distributed.barrier()
        record("before_vllm_weights")
        output = await original_weights(self, *args, **kwargs)
        record("after_vllm_weights")
        return output

    engine_workers.ActorRolloutRefWorker.update_weights = update_weights
    FSDPEngineWithLMHead._opd_h100_installed = True
    print("H100_HOOK_INSTALLED: detached reporting outputs; loss graph unchanged", flush=True)
