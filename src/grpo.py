"""Verifier-reward GRPO configuration for verl's native training loop."""

from __future__ import annotations

from typing import Any

from src.config import override
from src.opd import CONFIG_KEYS as OPD_ONLY_KEYS


CONFIG_KEYS: set[str] = set()
ENVIRONMENT_OVERRIDES: dict[str, str] = {}


def validate_config(cfg: dict[str, Any]) -> None:
    stray = sorted(OPD_ONLY_KEYS.intersection(cfg))
    if stray:
        raise ValueError(f"GRPO config contains OPD-only keys: {stray}")


def build_overrides(cfg: dict[str, Any]) -> list[str]:
    return [
        override("algorithm.adv_estimator", "grpo"),
        override("algorithm.use_kl_in_reward", False),
        override("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"),
        override("distillation.enabled", False),
    ]
