"""Entropy-aware reverse-KL variant: top-k k1/PPO plus a gated reverse KL.

This recipe intentionally replaces the paper's forward-KL auxiliary term with
reverse KL. Both distributions and teacher entropy use the same normalized
teacher top-k support. It is not a reproduction of the paper's EOPD objective.
"""

from __future__ import annotations

from functools import partial
import math
from typing import Any

from src import opd
from src.config import override, strict_bool


ENTROPY_KEYS = {"eopd_entropy_threshold", "eopd_aux_loss_coef"}
CONFIG_KEYS = opd.CONFIG_KEYS | ENTROPY_KEYS
ENVIRONMENT_OVERRIDES = opd.ENVIRONMENT_OVERRIDES
DEFAULTS = {"distillation_loss_mode": "k1", "distillation_topk": 16, "distillation_use_policy_gradient": True}


def entropy_parameters(cfg: dict[str, Any]) -> tuple[float, float]:
    values = []
    for key, default in (("eopd_entropy_threshold", 0.8), ("eopd_aux_loss_coef", 1.0)):
        value = cfg.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a nonnegative finite number")
        values.append(float(value))
    return tuple(values)


def validate_config(cfg: dict[str, Any]) -> None:
    normalized = {**DEFAULTS, **cfg}
    if normalized["distillation_loss_mode"] != "k1":
        raise ValueError("EOPD requires distillation_loss_mode: k1")
    if not strict_bool(normalized, "distillation_use_policy_gradient", True):
        raise ValueError("EOPD requires distillation_use_policy_gradient: true")
    opd.validate_config(normalized)
    if opd.topk_size(normalized) is None:
        raise ValueError("EOPD requires a positive distillation_topk")
    entropy_parameters(cfg)


def build_overrides(cfg: dict[str, Any]) -> list[str]:
    validate_config(cfg)
    threshold, coefficient = entropy_parameters(cfg)
    return opd.build_overrides({**DEFAULTS, **cfg}) + [
        override("+eopd.enabled", True),
        override("+eopd.entropy_threshold", threshold),
        override("+eopd.aux_loss_coef", coefficient),
    ]


def entropy_reverse_kl_terms(student_logps, teacher_logps, threshold):
    """A teacher-only gate selects positions for the direct conditional RKL."""
    teacher_logps = teacher_logps.detach().float().log_softmax(-1)
    teacher_probs = teacher_logps.exp()
    entropy = -(teacher_probs * teacher_logps.masked_fill(teacher_probs == 0, 0.0)).sum(-1)
    high_entropy = (entropy > threshold).float()
    reverse_kl = (student_logps.exp() * (student_logps - teacher_logps)).sum(-1)
    return {
        "eopd_aux_losses": high_entropy * reverse_kl,
        "eopd_teacher_entropy": entropy,
        "eopd_high_entropy": high_entropy,
    }


def eopd_loss(*, config, distillation_config, entropy_threshold, aux_loss_coef,
              student_logits=None, model_output=None, data=None, dp_group=None):
    from verl.utils import tensordict_utils as tu

    # Both losses share one top-k selection and its dense vocabulary gradient.
    # Scoring returns before this auxiliary callback is evaluated.
    base = opd.topk_k1_loss(config=config, distillation_config=distillation_config,
                            student_logits=student_logits, model_output=model_output, data=data, dp_group=dp_group,
                            auxiliary_terms=partial(entropy_reverse_kl_terms, threshold=entropy_threshold))
    if tu.get_non_tensor_data(data, "opd_topk_scoring", False) or student_logits is not None:
        return base

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    loss, metrics = base
    batch_info = {key: data[key] for key in ("dp_size", "batch_num_tokens", "global_batch_size")}
    batch_info["loss_scale_factor"] = config.loss_scale_factor
    mask = data["response_mask"].bool()
    auxiliary = None
    for field, metric_name in (
        ("eopd_aux_losses", "aux_reverse_kl"),
        ("eopd_teacher_entropy", "teacher_topk_entropy"),
        ("eopd_high_entropy", "high_entropy_fraction"),
    ):
        values = no_padding_2_padding(model_output[field], data)
        value = agg_loss(values, mask, config.loss_agg_mode, **batch_info)
        metrics[f"eopd/{metric_name}"] = Metric(AggregationType.SUM, value)
        if field == "eopd_aux_losses":
            auxiliary = value
    # Normalize over all valid response positions, not just the selected ones.
    return loss + aux_loss_coef * auxiliary, metrics


class EOPDMixin(opd.OPDTopKMixin):
    """Share OPD rollout/old-logprob caching and install the entropy-aware loss."""

    def init_workers(self):
        if not self._topk_enabled() or not self.config.get("eopd", {}).get("enabled", False):
            raise ValueError("EOPD requires enabled top-k reverse-KL distillation")
        entropy_parameters({
            "eopd_entropy_threshold": self.config.eopd.entropy_threshold,
            "eopd_aux_loss_coef": self.config.eopd.aux_loss_coef,
        })
        super().init_workers()

    def _build_loss_fn(self):
        base = super()._build_loss_fn()
        return partial(eopd_loss, **base.keywords,
                       entropy_threshold=self.config.eopd.entropy_threshold,
                       aux_loss_coef=self.config.eopd.aux_loss_coef)
