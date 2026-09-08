"""Reverse-KL OPD: sampled k1, or a top-k conditional k1/PPO expectation."""

from __future__ import annotations

from functools import partial
from typing import Any

from src.config import model_reference, override, positive_int, strict_bool


CONFIG_KEYS = {
    "teacher_model_path",
    "teacher_enable_resource_pool",
    "teacher_n_gpus_per_node",
    "teacher_tp_size",
    "teacher_gpu_memory_utilization",
    "distillation_num_workers",
    "distillation_loss_mode",
    "distillation_topk",
    "distillation_use_policy_gradient",
    "distillation_use_task_rewards",
    "distillation_loss_coef",
    "distillation_loss_max_clamp",
    "distillation_log_prob_min_clamp",
}
ENVIRONMENT_OVERRIDES = {"TEACHER_MODEL_PATH": "teacher_model_path"}


def topk_size(cfg: dict[str, Any]) -> int | None:
    value = cfg.get("distillation_topk")
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("distillation_topk must be a positive integer or null")
    return value


def uses_topk_k1(cfg: dict[str, Any]) -> bool:
    return cfg.get("distillation_loss_mode", "k1") == "k1" and topk_size(cfg) is not None


def validate_config(cfg: dict[str, Any]) -> None:
    if not cfg.get("teacher_model_path"):
        raise ValueError("OPD requires teacher_model_path")
    topk = topk_size(cfg)
    mode = cfg.get("distillation_loss_mode", "k1")
    if mode == "forward_kl_topk":
        raise ValueError("Only reverse-KL distillation is supported; use k1 or reverse_kl_topk")
    if mode == "reverse_kl_topk" and topk is None:
        raise ValueError("reverse_kl_topk requires a positive distillation_topk")
    if topk is not None and mode not in {"k1", "reverse_kl_topk"}:
        raise ValueError("distillation_topk is supported for k1 and reverse_kl_topk; use null for other estimators")
    if uses_topk_k1(cfg) and not strict_bool(cfg, "distillation_use_policy_gradient", True):
        raise ValueError("top-k k1 requires distillation_use_policy_gradient: true")


def build_overrides(cfg: dict[str, Any]) -> list[str]:
    validate_config(cfg)
    prompt_length = positive_int(cfg, "max_prompt_length", 2048)
    response_length = positive_int(cfg, "max_completion_length", 8192)
    train_max_seq = prompt_length + response_length
    loss_clamp = cfg.get("distillation_loss_max_clamp", 10.0)
    log_prob_clamp = cfg.get("distillation_log_prob_min_clamp", -10.0)
    overrides = [
        override("algorithm.adv_estimator", "grpo"),
        override("algorithm.use_kl_in_reward", False),
        override("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"),
        override("distillation.enabled", True),
        override("distillation.num_workers", positive_int(cfg, "distillation_num_workers", 8)),
        override("distillation.teacher_model.enable_resource_pool", strict_bool(cfg, "teacher_enable_resource_pool", False)),
        override("distillation.teacher_model.n_gpus_per_node", int(cfg.get("teacher_n_gpus_per_node", 0))),
        override("distillation.teacher_model.nnodes", 1),
        override("distillation.teacher_model.model_path", model_reference(str(cfg["teacher_model_path"]))),
        override("distillation.teacher_model.inference.tensor_model_parallel_size", positive_int(cfg, "teacher_tp_size", 1)),
        override("distillation.teacher_model.inference.name", "vllm"),
        override("distillation.teacher_model.inference.gpu_memory_utilization", float(cfg.get("teacher_gpu_memory_utilization", 0.3))),
        override("distillation.teacher_model.inference.prompt_length", prompt_length),
        override("distillation.teacher_model.inference.response_length", response_length),
        override("distillation.teacher_model.inference.max_model_len", train_max_seq + 1),
        override("distillation.teacher_model.inference.max_num_batched_tokens", train_max_seq + 1),
        override("distillation.distillation_loss.loss_mode", "reverse_kl_topk" if uses_topk_k1(cfg) else cfg.get("distillation_loss_mode", "k1")),
        override("distillation.distillation_loss.topk", topk_size(cfg)),
        override("distillation.distillation_loss.use_task_rewards", strict_bool(cfg, "distillation_use_task_rewards", False)),
        override("distillation.distillation_loss.use_policy_gradient", strict_bool(cfg, "distillation_use_policy_gradient", True)),
        override("distillation.distillation_loss.policy_loss_mode", "vanilla"),
        override("distillation.distillation_loss.distillation_loss_coef", float(cfg.get("distillation_loss_coef", 1.0))),
        override("distillation.distillation_loss.loss_max_clamp", None if loss_clamp is None else float(loss_clamp)),
        override("distillation.distillation_loss.log_prob_min_clamp", None if log_prob_clamp is None else float(log_prob_clamp)),
    ]
    if uses_topk_k1(cfg) or cfg.get("distillation_loss_mode") == "reverse_kl_topk":
        overrides.append(override("distillation.distillation_loss._target_", "src.opd.make_topk_loss_config"))
    if uses_topk_k1(cfg):
        overrides.append(override("+opd.topk_enabled", True))
    return overrides


def _requires_topk_actor_loss(*args, **kwargs):
    raise RuntimeError("reverse_kl_topk requires a local reverse-KL trainer from src.runtime")


def make_topk_loss_config(**kwargs):
    """Hydra factory registers the transport settings in each Ray process."""
    from verl.trainer.distillation.losses import (
        DISTILLATION_LOSS_REGISTRY, DistillationLossSettings, register_distillation_loss,
    )
    from verl.workers.config import DistillationLossConfig

    if "reverse_kl_topk" not in DISTILLATION_LOSS_REGISTRY:
        register_distillation_loss(DistillationLossSettings(names=["reverse_kl_topk"], use_topk=True))(
            _requires_topk_actor_loss
        )
    if topk_size({"distillation_topk": kwargs.get("topk")}) is None:
        raise ValueError("reverse_kl_topk requires a positive topk; set distillation_topk: null in the recipe for vanilla k1")
    return DistillationLossConfig(**kwargs)


def topk_k1_terms(student_logps, teacher_logps, old_logps, *, signal_clip=10.0,
                  clip_low=0.2, clip_high=0.2, dual_clip=3.0):
    """PPO expectation over teacher top-k, with both distributions normalized.

    Before clipping, its gradient equals that of KL(student_topk || teacher_topk).
    The fixed old distribution supplies both the expectation weights and PPO
    denominator. Each candidate gets its own detached k1 advantage and ratio.
    """
    import torch

    teacher_logps = teacher_logps.detach().float().log_softmax(-1)
    old_logps = old_logps.detach().float()
    k1 = student_logps - teacher_logps
    advantage = -k1.detach()
    if signal_clip is not None:
        advantage = advantage.clamp(min=-signal_clip, max=signal_clip)
    log_ratio = (student_logps - old_logps).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    raw = -advantage * ratio
    clipped = -advantage * ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    upper = torch.maximum(raw, clipped)
    lower = torch.minimum(-advantage * dual_clip, upper)
    weights = old_logps.exp()
    return {
        "distillation_losses": (weights * torch.where(advantage < 0, lower, upper)).sum(-1),
        "topk_reverse_kl": (student_logps.detach().exp() * k1.detach()).sum(-1),
        "topk_pg_clipfrac": (weights * (clipped > raw).float()).sum(-1),
        "topk_pg_clipfrac_lower": (weights * ((upper > -advantage * dual_clip) & (advantage < 0)).float()).sum(-1),
        "topk_ppo_kl": (weights * -log_ratio.detach()).sum(-1),
    }


def topk_k1_loss(*, config, distillation_config, student_logits=None, model_output=None, data=None, dp_group=None,
                 auxiliary_terms=None):
    """FSDP logits callback and final loss callback, using native actor updates."""
    import torch
    from verl.utils import tensordict_utils as tu

    scoring = tu.get_non_tensor_data(data, "opd_topk_scoring", False)
    loss_config = distillation_config.distillation_loss
    if student_logits is not None:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
        from src.topk import selected_log_probs

        ids = data["teacher_ids"].values().unsqueeze(0)
        teacher_logps = data["teacher_logprobs"].values().unsqueeze(0)
        old_logps = None if scoring else data["opd_old_topk_log_probs"].values().unsqueeze(0)
        if get_ulysses_sequence_parallel_world_size() > 1:
            ids = slice_input_tensor(ids, dim=1)
            teacher_logps = slice_input_tensor(teacher_logps, dim=1)
            if old_logps is not None:
                old_logps = slice_input_tensor(old_logps, dim=1)
        if ids.shape != teacher_logps.shape or ids.shape[:2] != student_logits.shape[:2]:
            raise ValueError("OPD top-k teacher probabilities must align with packed student logits")
        # Normalize student logits over exactly the same teacher-selected support.
        student_logps = selected_log_probs(student_logits, ids)
        if scoring:
            # verl's FSDP output interface expects one scalar per sequence position.
            return {f"opd_old_topk_{i}": student_logps[..., i].contiguous() for i in range(ids.shape[-1])}
        if old_logps.shape != student_logps.shape:
            raise ValueError("OPD top-k old probabilities must match the teacher-selected support")
        output = topk_k1_terms(
            student_logps, teacher_logps, old_logps,
            signal_clip=loss_config.loss_max_clamp,
            clip_low=loss_config.clip_ratio_low if loss_config.clip_ratio_low is not None else loss_config.clip_ratio,
            clip_high=loss_config.clip_ratio_high if loss_config.clip_ratio_high is not None else loss_config.clip_ratio,
            dual_clip=loss_config.get("clip_ratio_c", 3.0),
        )
        output["topk_teacher_mass"] = teacher_logps.detach().exp().sum(-1)
        if auxiliary_terms is not None:
            output.update(auxiliary_terms(student_logps, teacher_logps))
        return output

    if scoring:
        return model_output["log_probs"].values().new_zeros(()), {}

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    batch_info = {key: data[key] for key in ("dp_size", "batch_num_tokens", "global_batch_size")}
    batch_info["loss_scale_factor"] = config.loss_scale_factor
    mask = data["response_mask"].bool()
    losses = no_padding_2_padding(model_output["distillation_losses"], data)
    if "rollout_is_weights" in data:
        losses = losses * data["rollout_is_weights"]
    distill_loss = agg_loss(losses, mask, config.loss_agg_mode, **batch_info)
    metrics = {"distillation/loss": Metric(AggregationType.SUM, distill_loss)}
    for field in ("topk_reverse_kl", "topk_pg_clipfrac", "topk_pg_clipfrac_lower", "topk_ppo_kl", "topk_teacher_mass"):
        values = no_padding_2_padding(model_output[field], data)
        metrics[f"distillation/{field}"] = Metric(
            AggregationType.SUM, agg_loss(values, mask, config.loss_agg_mode, **batch_info)
        )
    # Match native OPD's reward combination and coefficient semantics.
    if loss_config.use_task_rewards:
        task_loss, task_metrics = ppo_loss(config, model_output, data, dp_group)
        metrics.update(task_metrics)
        return task_loss + loss_config.distillation_loss_coef * distill_loss, metrics
    return distill_loss, metrics


class OPDTopKMixin:
    """Extend the existing old-logprob pass to cache the top-k PPO reference."""

    def _topk_enabled(self):
        return self.config.get("opd", {}).get("topk_enabled", False)

    def _build_loss_fn(self):
        from verl.utils.config import omega_conf_to_dataclass

        return partial(
            topk_k1_loss,
            config=omega_conf_to_dataclass(self.config.actor_rollout_ref.actor),
            distillation_config=omega_conf_to_dataclass(self.config.distillation),
        )

    def init_workers(self):
        if self._topk_enabled():
            config = self.config
            actor = config.actor_rollout_ref.actor
            model = config.actor_rollout_ref.model
            if config.trainer.use_legacy_worker_impl != "disable" or actor.strategy not in {"fsdp", "fsdp2"}:
                raise ValueError("top-k k1 requires native FSDP/FSDP2 engine workers")
            if not model.use_remove_padding or model.get("use_fused_kernels", False):
                raise ValueError("top-k k1 requires use_remove_padding=true and use_fused_kernels=false")
            loss = config.distillation.distillation_loss
            if not config.distillation.enabled or loss.loss_mode != "reverse_kl_topk" or not loss.use_policy_gradient:
                raise ValueError("top-k k1 requires enabled reverse_kl_topk policy-gradient distillation")
            if config.algorithm.get("rollout_correction", {}).get("bypass_mode", False):
                raise ValueError("top-k k1 requires the FSDP old-logprob pass; rollout bypass is unsupported")
        super().init_workers()
        if self._topk_enabled():
            if self.tokenizer.get_vocab() != self.teacher_model_manager.tokenizer.get_vocab():
                raise ValueError("top-k k1 requires matching teacher/student token-to-ID vocabularies")
            self.actor_rollout_wg.set_loss_fn(self._build_loss_fn())

    def _compute_old_log_prob(self, batch):
        if not self._topk_enabled():
            return super()._compute_old_log_prob(batch)
        import torch
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        packed = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(packed, calculate_entropy=True, compute_loss=True,
                             distillation_use_topk=True, opd_topk_scoring=True)
        output = self.actor_rollout_wg.compute_log_prob(packed)
        topk = self.config.distillation.distillation_loss.topk
        fields = [tu.get(output, f"opd_old_topk_{i}") for i in range(topk)]
        reference = torch.nested.nested_tensor_from_jagged(
            torch.stack([field.values() for field in fields], dim=-1), fields[0].offsets(),
        )
        tensors = {
            "old_log_probs": no_padding_2_padding(tu.get(output, "log_probs"), packed).float(),
            "entropys": no_padding_2_padding(tu.get(output, "entropy"), packed).float(),
            "opd_old_topk_log_probs": reference.detach(),
        }
        routed_experts = tu.get(output, "routed_experts")
        if routed_experts is not None:
            tensors["routed_experts"] = routed_experts
        return DataProto.from_tensordict(tu.get_tensordict(tensors)), tu.get(output, "metrics")["mfu"]
