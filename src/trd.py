"""Teacher-refined trajectories with the unchanged native OPD loss."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from data.refine import refine_token_ids
from src import opd
from src.config import override, positive_int, strict_bool


REFINE_KEYS = {"refine_max_prompt_length", "refine_max_new_tokens"}
CONFIG_KEYS = opd.CONFIG_KEYS | REFINE_KEYS
ENVIRONMENT_OVERRIDES = opd.ENVIRONMENT_OVERRIDES


def validate_config(cfg: dict[str, Any]) -> None:
    opd.validate_config(cfg)
    if strict_bool(cfg, "teacher_enable_resource_pool", False):
        raise ValueError("TRD requires the teacher to share the student GPU resource pool")
    if "refine_max_prompt_length" in cfg:
        positive_int(cfg, "refine_max_prompt_length")
    if "refine_max_new_tokens" in cfg:
        budget = positive_int(cfg, "refine_max_new_tokens")
        if budget > positive_int(cfg, "max_completion_length", 8192):
            raise ValueError("refine_max_new_tokens cannot exceed max_completion_length")


def build_overrides(cfg: dict[str, Any]) -> list[str]:
    validate_config(cfg)
    prompt_length = positive_int(cfg, "max_prompt_length", 2048)
    response_length = positive_int(cfg, "max_completion_length", 8192)
    refine_length = positive_int(cfg, "refine_max_prompt_length", prompt_length + response_length + 1024)
    refine_budget = positive_int(cfg, "refine_max_new_tokens", response_length)
    # Refinement consumes x + y_o + instructions, followed by y_r (refine_budget,
    # NOT the full response_length). Scoring still consumes the original student
    # prompt + y_r + one token, so both bounds are kept.
    context_length = max(refine_length + refine_budget, prompt_length + response_length + 1)
    overrides = dict(item.split("=", 1) for item in opd.build_overrides(cfg))
    overrides["distillation.teacher_model.inference.max_model_len"] = str(context_length)
    overrides["distillation.teacher_model.inference.max_num_batched_tokens"] = str(context_length)
    return [f"{key}={value}" for key, value in overrides.items()] + [
        override("+trd.enabled", True),
        override("+trd.max_prompt_length", refine_length),
        # Cap on y_r. Defaults to the full response budget (previous behaviour);
        # a smaller value stops vLLM from reserving KV space for the worst case,
        # which is what limits teacher decode concurrency.
        override("+trd.max_new_tokens", positive_int(cfg, "refine_max_new_tokens", response_length)),
    ]


def configure_teacher_context(config) -> int | None:
    """Reconcile the final Hydra overrides before constructing any teacher engine."""
    if not config.get("trd", {}).get("enabled", False):
        return None
    prompt_length = positive_int(config.data, "max_prompt_length")
    response_length = positive_int(config.actor_rollout_ref.rollout, "response_length")
    refine_length = positive_int(config.trd, "max_prompt_length")
    refine_budget = positive_int(config.trd, "max_new_tokens", response_length)
    required = max(refine_length + refine_budget, prompt_length + response_length + 1)
    inference = config.distillation.teacher_model.inference
    requested = inference.get("max_model_len")
    capacity = max(required, positive_int(inference, "max_model_len") if requested is not None else 0)
    # Native distillation subsequently converts x + y_r into a scoring prompt
    # with one output token. Keep that separate from the rewrite's y_r budget.
    inference.prompt_length = prompt_length
    inference.response_length = response_length
    inference.max_model_len = capacity
    inference.max_num_batched_tokens = max(
        capacity, positive_int(inference, "max_num_batched_tokens", capacity),
    )
    if requested != capacity:
        print(f"TRD teacher max_model_len: {requested} -> {capacity} "
              f"(rewrite prompt limit={refine_length}, output budget={refine_budget})")
    return required


def check_teacher_context(prompt_length: int, response_length: int, capacity: int, *, stage: str) -> None:
    required = prompt_length + response_length
    if capacity < required:
        raise ValueError(
            "TRD teacher context must fit the complete rewrite prompt and y_r output budget: "
            f"{stage}, prompt_tokens={prompt_length}, output_budget={response_length}, "
            f"required_context={required}, teacher_max_model_len={capacity}. "
            "The teacher engine must be initialized with at least required_context tokens; "
            "check the final runtime configuration and restart with the updated src.runtime entrypoint."
        )


def replace_responses(batch, responses: list[list[int]], pad_token_id: int) -> None:
    """Replace y_o in-place before native teacher scoring and old-logprob scoring."""
    import torch

    prompts = batch.batch["prompts"]
    old_responses = batch.batch["responses"]
    if len(responses) != len(prompts):
        raise ValueError("TRD requires exactly one rewrite per student trajectory")
    width = old_responses.shape[-1]
    if any(not ids or len(ids) > width for ids in responses):
        raise ValueError(f"TRD rewrites must contain between 1 and {width} student tokens")
    if batch.batch["position_ids"].ndim != 2:
        raise ValueError("TRD currently supports text-only trajectories")
    rewritten = torch.full_like(old_responses, pad_token_id)
    response_mask = torch.zeros_like(batch.batch["response_mask"])
    for i, ids in enumerate(responses):
        rewritten[i, :len(ids)] = torch.as_tensor(ids, dtype=rewritten.dtype, device=rewritten.device)
        response_mask[i, :len(ids)] = 1
    prompt_mask = batch.batch["attention_mask"][:, :prompts.shape[-1]]
    attention_mask = torch.cat([prompt_mask, response_mask.to(prompt_mask.dtype)], dim=-1)

    # Nothing derived from the old answer may survive into the new loss/rewards.
    stale_tensors = {
        "rm_scores", "rollout_log_probs", "routed_experts", "teacher_ids", "teacher_logprobs",
        "old_log_probs", "opd_old_topk_log_probs", "rollout_is_weights", "advantages", "returns", "ref_log_prob",
        "token_level_scores", "token_level_rewards", "entropys",
    }
    for key in stale_tensors.intersection(batch.batch.keys()):
        del batch.batch[key]
    for key in batch.meta_info.pop("reward_extra_keys", []):
        batch.non_tensor_batch.pop(key, None)
    for key in ("turn_scores", "tool_rewards", "extras"):
        batch.non_tensor_batch.pop(key, None)
    batch.batch["responses"] = rewritten
    batch.batch["response_mask"] = response_mask
    batch.batch["input_ids"] = torch.cat([prompts, rewritten], dim=-1)
    batch.batch["attention_mask"] = attention_mask
    batch.batch["position_ids"] = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


class TRDTrajectoryMixin:
    """Insert one teacher generation stage into the native colocated OPD hook."""

    def init_workers(self):
        if not self.config.distillation.enabled or self.config.distillation.teacher_model.enable_resource_pool:
            raise ValueError("TRD requires enabled, colocated teacher distillation")
        correction = self.config.algorithm.get("rollout_correction")
        if correction and correction.get("bypass_mode", False):
            raise ValueError("TRD requires FSDP old_log_probs recomputation; rollout bypass is unsupported")
        configure_teacher_context(self.config)
        super().init_workers()
        capacity = self.teacher_model_manager.config.teacher_model.inference.max_model_len
        prompt_limit = self.config.trd.max_prompt_length
        output_budget = positive_int(self.config.trd, "max_new_tokens",
                                    self.config.actor_rollout_ref.rollout.response_length)
        check_teacher_context(prompt_limit, output_budget, capacity, stage="teacher initialization")
        print(f"TRD teacher context verified: rewrite_prompt_limit={prompt_limit}, "
              f"output_budget={output_budget}, teacher_max_model_len={capacity}")

    def _generate_refinements(self, batch) -> list[list[int]]:
        teacher = self.teacher_model_manager
        teacher_tokenizer = teacher.tokenizer
        if not hasattr(self, "_trd_shared_vocabulary"):
            self._trd_shared_vocabulary = self.tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
        mode = self.config.data.prompt_mode
        limit = self.config.trd.max_prompt_length
        # The tensor's padded storage width is not the configured generation budget.
        response_length = positive_int(self.config.actor_rollout_ref.rollout, "response_length")
        response_width = batch.batch["responses"].shape[-1]
        if response_width < response_length:
            raise ValueError(f"TRD response storage width={response_width} is smaller than "
                             f"the configured rewrite output budget={response_length}")
        prompt_width = batch.batch["prompts"].shape[-1]
        refine_budget = positive_int(self.config.trd, "max_new_tokens", response_length)
        prompts = []
        for i, messages in enumerate(batch.non_tensor_batch["raw_prompt"]):
            valid = batch.batch["attention_mask"][i, prompt_width:].bool()
            initial_ids = batch.batch["responses"][i][valid].tolist()
            initial_response = self.tokenizer.decode(initial_ids, skip_special_tokens=True)
            ids = refine_token_ids(teacher_tokenizer, messages, initial_response, mode, limit)
            check_teacher_context(
                len(ids), refine_budget, teacher.config.teacher_model.inference.max_model_len,
                stage=f"rewrite sample {i}",
            )
            prompts.append(ids)

        rollout = self.config.actor_rollout_ref.rollout
        # Requesting the full response_length makes vLLM reserve KV cache for the
        # worst case, so teacher decode concurrency collapses (~24 with a 7.2 GB
        # cache) even though rewrites average ~3.9k tokens. trd.max_new_tokens
        # defaults to response_length, preserving the original behaviour.
        sampling_params = {
            # Explicit max_tokens is essential: native distillation defaults to
            # generating only ONE token when requesting prompt log-probabilities.
            "max_tokens": refine_budget,
            "temperature": rollout.temperature,
            "top_p": rollout.top_p,
            "top_k": rollout.top_k,
            "repetition_penalty": 1.0,
        }

        async def generate_all():
            return await asyncio.gather(*[
                teacher.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=ids, sampling_params=dict(sampling_params),
                )
                for ids in prompts
            ])

        teacher.wake_up()
        try:
            outputs = teacher._run_single(generate_all())
        finally:
            teacher.sleep()
        responses = []
        for output in outputs:
            if output.stop_reason == "aborted" or not output.token_ids:
                raise RuntimeError("TRD teacher returned an aborted or empty rewrite")
            if self._trd_shared_vocabulary:
                # Keep the exact sampled token sequence, including EOS, when
                # student and teacher share the Qwen vocabulary.
                responses.append(list(output.token_ids))
            else:
                text = teacher_tokenizer.decode(output.token_ids, skip_special_tokens=False)
                responses.append(self.tokenizer.encode(text, add_special_tokens=False))
        return responses

    def _compute_teacher_colocate(self, batch):
        responses = self._generate_refinements(batch)
        replace_responses(batch, responses, self.tokenizer.pad_token_id)
        # Deliberately score x + y_r, exactly as OPD scores x + y_o. The refine
        # prompt is used only to GENERATE y_r, never to condition the loss.
        teacher_batch = super()._compute_teacher_colocate(batch)
        # Native rollout already graded y_o. Regrade y_r so training metrics
        # and the native advantage plumbing refer to the replacement answer.
        batch.union(self.reward_loop_manager.compute_rm_score(batch))
        return teacher_batch
