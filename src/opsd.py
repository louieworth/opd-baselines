"""Reverse KL with pointwise upper clipping on student-generated trajectories.

This external-teacher recipe approximates a full-vocabulary reverse-KL objective
using teacher top-k probabilities, without renormalizing the retained support.
The local loss plugs into verl's existing FSDP worker and optimizer interface.
"""

from __future__ import annotations

from functools import partial
import math
from typing import Any

from src import opd
from src.config import override, positive_int


POINTWISE_KEYS = {"distillation_pointwise_clip"}
CONFIG_KEYS = opd.CONFIG_KEYS | POINTWISE_KEYS
ENVIRONMENT_OVERRIDES = opd.ENVIRONMENT_OVERRIDES
LOSS_DEFAULTS = {
    "distillation_loss_mode": "reverse_kl_topk",
    "distillation_use_policy_gradient": False,
    "distillation_use_task_rewards": False,
    "distillation_loss_max_clamp": None,
    "distillation_log_prob_min_clamp": None,
}


def pointwise_clip(cfg: dict[str, Any]) -> float | None:
    value = cfg.get("distillation_pointwise_clip", 0.05)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("distillation_pointwise_clip must be a positive finite number or null")
    return float(value)


def validate_config(cfg: dict[str, Any]) -> None:
    for key, expected in LOSS_DEFAULTS.items():
        value = cfg.get(key, expected)
        if value != expected or type(value) is not type(expected):
            raise ValueError(f"OPSD requires {key}: {expected!r}")
    opd.validate_config({**LOSS_DEFAULTS, "distillation_topk": 64, **cfg})
    pointwise_clip(cfg)
    positive_int(cfg, "distillation_topk", 64)
    if float(cfg.get("temperature", 1.0)) != 1.0:
        raise ValueError("OPSD requires temperature: 1.0 to match vLLM teacher prompt log-probabilities")
    if float(cfg.get("kl_loss_coef", 0.0)) != 0.0:
        raise ValueError("OPSD requires kl_loss_coef: 0.0 (direct distillation only)")


def build_overrides(cfg: dict[str, Any]) -> list[str]:
    validate_config(cfg)
    return opd.build_overrides({**LOSS_DEFAULTS, "distillation_topk": 64, **cfg}) + [
        override("+opsd.enabled", True),
        override("+opsd.pointwise_clip", pointwise_clip(cfg)),
    ]


def reverse_kl_terms(student_logits, teacher_log_probs, teacher_ids, clip):
    """Return per-position loss/diagnostics; the teacher is always detached.

    p_S * (log p_S - log p_T) is upper-clipped BEFORE summing vocabulary
    entries. Negative entries and negative sums are preserved. Top-k logps
    retain their full-vocabulary normalization, so omitted tail terms are zero.
    """
    import torch

    logits = student_logits.float()
    teacher_log_probs = teacher_log_probs.detach().float()
    selected = logits.gather(-1, teacher_ids.long()) - torch.logsumexp(logits, dim=-1, keepdim=True)
    teacher_probs = teacher_log_probs.exp()
    student_probs = selected.exp()
    # A zero teacher probability has infinite reverse-KL contribution. Handle
    # its clipped constant separately to avoid 0 * inf in clamp's backward.
    zero_teacher = torch.isneginf(teacher_log_probs)
    finite_terms = student_probs * (selected - teacher_log_probs.masked_fill(zero_teacher, 0.0))
    terms = finite_terms.masked_fill(zero_teacher, float("inf"))
    clipped = terms if clip is None else finite_terms.clamp(max=clip).masked_fill(zero_teacher, clip)
    clip_fraction = torch.zeros_like(terms[..., 0]) if clip is None else (terms > clip).float().mean(-1)
    return {
        "distillation_losses": clipped.sum(-1),
        "opsd_unclipped_losses": terms.detach().sum(-1),
        "opsd_clip_fraction": clip_fraction,
        "student_mass": student_probs.detach().sum(-1),
        "teacher_mass": teacher_probs.sum(-1),
    }


def opsd_loss(*, pointwise_clip, loss_coef=1.0, loss_agg_mode="token-mean", loss_scale_factor=None,
              student_logits=None, model_output=None, data=None, dp_group=None):
    """Dual callback: process FSDP logits, then aggregate response loss for backward."""
    if student_logits is not None:
        import torch
        from torch.utils.checkpoint import checkpoint
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

        logps = data["teacher_logprobs"].values().unsqueeze(0)
        ids = data["teacher_ids"].values().unsqueeze(0)
        if get_ulysses_sequence_parallel_world_size() > 1:
            logps = slice_input_tensor(logps, dim=1)
            ids = slice_input_tensor(ids, dim=1)
        if logps.shape != ids.shape or logps.shape[:2] != student_logits.shape[:2]:
            raise ValueError("OPSD teacher top-k tensors must align with packed student logits")

        # Bound the extra FP32 vocabulary workspace and recompute it in backward.
        # No additional model forward is introduced by this loss checkpoint.
        kernel = partial(reverse_kl_terms, clip=pointwise_clip)
        chunks = []
        for args in zip(student_logits.split(256, dim=1), logps.split(256, dim=1), ids.split(256, dim=1), strict=True):
            chunks.append(checkpoint(kernel, *args, use_reentrant=False) if student_logits.requires_grad else kernel(*args))
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=1) for key in chunks[0]}

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    mask = data["response_mask"]
    if mask.is_nested:
        mask = data.select("response_mask").to_padded_tensor()["response_mask"]
    mask = mask.bool()
    batch_info = {key: data[key] for key in ("dp_size", "batch_num_tokens", "global_batch_size")}
    batch_info["loss_scale_factor"] = loss_scale_factor
    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    loss = agg_loss(losses, mask, loss_agg_mode, **batch_info)
    normalized = batch_info["dp_size"] > 1 or any(
        batch_info[key] is not None for key in ("batch_num_tokens", "global_batch_size", "loss_scale_factor")
    )
    aggregation = AggregationType.SUM if normalized else AggregationType.MEAN
    metrics = {"distillation/loss": Metric(aggregation, loss)}
    for field, name in (
        ("opsd_unclipped_losses", "opsd/unclipped_loss"),
        ("opsd_clip_fraction", "opsd/clip_fraction"),
        ("teacher_mass", "opsd/teacher_topk_mass"),
        ("student_mass", "opsd/student_topk_mass"),
    ):
        values = no_padding_2_padding(model_output[field], data)
        metrics[name] = Metric(aggregation, agg_loss(values, mask, loss_agg_mode, **batch_info))
    # No PPO, detached-advantage surrogate, post-sum clamp, or nonnegative floor.
    return loss * loss_coef, metrics


class OPSDLossMixin:
    """Install the loss on existing actors after native worker initialization."""

    def init_workers(self):
        config = self.config
        actor = config.actor_rollout_ref.actor
        model = config.actor_rollout_ref.model
        distill = config.distillation.distillation_loss
        if config.trainer.use_legacy_worker_impl != "disable" or actor.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError("OPSD requires native FSDP/FSDP2 engine workers")
        if not model.use_remove_padding or model.get("use_fused_kernels", False):
            raise ValueError("OPSD requires use_remove_padding=true and use_fused_kernels=false for student logits")
        if not config.distillation.enabled or config.get("trd", {}).get("enabled", False):
            raise ValueError("OPSD requires teacher distillation without TRD refinement")
        validate_config({
            "teacher_model_path": config.distillation.teacher_model.model_path,
            **{key: distill[key.removeprefix("distillation_")] for key in LOSS_DEFAULTS},
            "distillation_pointwise_clip": config.opsd.pointwise_clip,
            "distillation_topk": distill.topk,
            "temperature": config.actor_rollout_ref.rollout.temperature,
            "kl_loss_coef": actor.kl_loss_coef if actor.use_kl_loss else 0.0,
        })
        if actor.entropy_coeff != 0 or config.distillation.teacher_model.inference.temperature != 1.0:
            raise ValueError("OPSD requires entropy_coeff=0 and teacher temperature=1.0")
        super().init_workers()
        if self.tokenizer.get_vocab() != self.teacher_model_manager.tokenizer.get_vocab():
            raise ValueError("OPSD requires matching teacher/student token-to-ID vocabularies")
        self.actor_rollout_wg.set_loss_fn(partial(
            opsd_loss,
            pointwise_clip=config.opsd.pointwise_clip,
            loss_coef=distill.distillation_loss_coef,
            loss_agg_mode=actor.loss_agg_mode,
            loss_scale_factor=actor.loss_scale_factor,
        ))
