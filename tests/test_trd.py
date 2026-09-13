"""Exercise the trajectory handoff without requiring Ray, vLLM, or CUDA."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from data.prompt_modes import ANSWER_INSTRUCTION
from data.refine import refine_prompt, refine_token_ids, refine_token_ids_batch
from src.trd import REFINE_KEYS, TRDTrajectoryMixin, check_teacher_context, configure_teacher_context, replace_responses
from train import build_command, load_config, main


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("topk", [None, 64])
def test_trd_changes_only_trajectory_stage_and_teacher_context(topk):
    recipe = load_config(ROOT / "configs/qwen3_4b_trd.yaml")
    recipe.pop("wandb_api_key", None)
    recipe["distillation_topk"] = topk
    # Hold sampling and resource settings fixed when comparing the algorithms;
    # packaged recipes can tune those independently.
    original_recipe = {key: value for key, value in recipe.items() if key not in REFINE_KEYS}
    original_recipe.update(method="opd", experiment_name="qwen3_4b_opd")
    effective = []
    for cfg in (original_recipe, recipe):
        command = build_command(cfg, Path("outputs/TEST"), [])
        with initialize_config_dir(config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None):
            effective.append(OmegaConf.to_container(compose(config_name="ppo_trainer", overrides=command[3:]), resolve=True))
    original, refined = effective
    assert refined.pop("trd") == {"enabled": True, "max_prompt_length": 12288, "max_new_tokens": 8192}
    inference = refined["distillation"]["teacher_model"]["inference"]
    assert inference["max_model_len"] == inference["max_num_batched_tokens"] == 20480
    for key in ("max_model_len", "max_num_batched_tokens"):
        inference[key] = original["distillation"]["teacher_model"]["inference"][key]
    assert refined["trainer"]["experiment_name"] == "qwen3_4b_trd"
    refined["trainer"]["experiment_name"] = original["trainer"]["experiment_name"]
    refined["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"] = original["actor_rollout_ref"]["rollout"]["trace"]["experiment_name"]
    assert refined == original
    loss = refined["distillation"]["distillation_loss"]
    assert loss["loss_mode"] == ("k1" if topk is None else "reverse_kl_topk")
    assert loss["use_policy_gradient"] is True
    assert loss["loss_max_clamp"] == 10
    assert loss["clip_ratio_low"] == loss["clip_ratio_high"] == 0.2


def test_trd_eval_only_removes_refinement(capsys):
    assert main([str(ROOT / "configs/qwen3_4b_trd.yaml"), "--eval-only", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "distillation.enabled=false" in out
    assert "+trd." not in out
    assert "distillation.teacher_model.model_path=" not in out


class Tokenizer:
    pad_token_id = 0
    padding_side = "left"
    truncation_side = "left"

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))

    def decode(self, ids, *, skip_special_tokens):
        text = "".join(map(chr, ids))
        return text.replace("\x02", "").replace("\x00", "") if skip_special_tokens else text

    def batch_decode(self, sequences, *, skip_special_tokens):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]

    def __call__(self, texts, *, add_special_tokens, padding, truncation, max_length, return_tensors):
        assert not add_special_tokens and padding and truncation and return_tensors == "pt"
        assert self.padding_side == self.truncation_side == "right"
        rows = [list(map(ord, text))[:max_length] for text in texts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt
        prefix = "thinking:" if enable_thinking else "non-thinking:"
        rendered = [prefix + conversation[0]["content"] for conversation in messages]
        return [self.encode(text, add_special_tokens=False) for text in rendered] if tokenize else rendered


def test_refine_prompt_preserves_question_and_initial_answer():
    question = "Solve {x + 1 = 3}.\n" + ANSWER_INSTRUCTION
    prompt = refine_prompt([{"role": "user", "content": question}], "x = {5}")
    assert "**Problem:**\nSolve {x + 1 = 3}." in prompt
    assert "**Your Initial Solution:**\nx = {5}" in prompt
    assert "Preserve the overall structure and reasoning path" in prompt
    assert prompt.count(ANSWER_INSTRUCTION) == 1


@pytest.mark.parametrize("mode", ["plaint", "thinking", "non-thinking"])
def test_refine_budget_clips_answer_after_rendering_and_keeps_instructions(mode):
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "Question"}]
    ids = refine_token_ids(tokenizer, messages, "whole initial answer", mode, 1000)
    assert refine_token_ids(tokenizer, messages, "whole initial answer", mode, len(ids)) == ids
    clipped = refine_token_ids(tokenizer, messages, "whole initial answer", mode, len(ids) - 1)
    assert len(clipped) == len(ids) - 1
    assert tokenizer.decode(clipped, skip_special_tokens=False) == tokenizer.decode(
        ids, skip_special_tokens=False,
    ).replace("whole initial answer", "whole initial answe")
    assert tokenizer.padding_side == tokenizer.truncation_side == "left"


@pytest.mark.parametrize("original_length", [20745, 23793])
def test_packaged_budget_clips_reported_overflows(original_length):
    recipe = load_config(ROOT / "configs/qwen3_4b_trd.yaml")
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "Question"}]
    mode = recipe["mode"]
    command = build_command(recipe, Path("outputs/TEST"), [])
    with initialize_config_dir(config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    assert configure_teacher_context(config) == 20480
    limit = config.trd.max_prompt_length
    overhead = len(refine_token_ids(tokenizer, messages, "", mode, limit))
    initial_response = "x" * (original_length - overhead)
    assert len(refine_token_ids(tokenizer, messages, initial_response, mode, original_length)) == original_length
    ids = refine_token_ids(tokenizer, messages, initial_response, mode, limit)
    assert len(ids) == limit == 12288
    assert tokenizer.decode(ids, skip_special_tokens=False) == "thinking:" + refine_prompt(
        messages, initial_response[:limit - overhead],
    )
    check_teacher_context(
        len(ids), config.trd.max_new_tokens, config.distillation.teacher_model.inference.max_model_len,
        stage="reported overflow",
    )


@pytest.mark.parametrize("mode", ["plaint", "thinking", "non-thinking"])
def test_batch_clipping_handles_short_exact_and_overlong_prompts(mode):
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "question"}]
    limit = 600
    overhead = len(refine_token_ids(tokenizer, messages, "", mode, limit))
    answers = ["", "short", "x" * (limit - overhead), "x" * 23793]
    result = refine_token_ids_batch(tokenizer, [messages] * len(answers), answers, mode, limit)
    assert list(map(len, result)) == [overhead, overhead + 5, limit, limit]
    for ids, answer in zip(result, answers):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        assert text.endswith(refine_prompt(messages, answer[:limit - overhead]))


def test_batch_clipping_reserves_instructions_when_question_exceeds_budget():
    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "question" + "x" * 23793}]
    ids = refine_token_ids_batch(tokenizer, [messages], ["answer"], "thinking", 600)[0]
    assert len(ids) == 600
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert text.startswith("thinking:Your task is to rewrite")
    assert "**Instructions:**" in text
    assert text.endswith(ANSWER_INSTRUCTION)


@pytest.mark.parametrize("response_length,student_context,refine_input,teacher_context", [
    (4096, 8192, 8192, 12288),
    (8192, 12288, 12288, 20480),
    (16384, 20480, 20480, 36864),
])
def test_recipe_completion_budget_updates_validation_refinement_and_contexts(
    response_length, student_context, refine_input, teacher_context,
):
    recipe = load_config(ROOT / "configs/qwen3_4b_trd.yaml")
    assert recipe["max_completion_length"] == 8192
    assert recipe["refine_prompt_token_reserve"] == 2048
    assert not {"val_max_completion_length", "refine_max_new_tokens", "refine_max_prompt_length"} & recipe.keys()
    recipe["max_completion_length"] = response_length
    command = build_command(recipe, Path("outputs/TEST"), [])
    with initialize_config_dir(config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    rollout = config.actor_rollout_ref.rollout
    assert config.data.max_response_length == rollout.response_length == response_length
    assert rollout.prompt_length == 4096
    assert rollout.max_model_len == rollout.max_num_batched_tokens == student_context
    assert rollout.log_prob_max_token_len_per_gpu == student_context
    assert config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == student_context
    assert config.trd.max_prompt_length == refine_input
    assert config.trd.max_new_tokens == response_length
    assert configure_teacher_context(config) == teacher_context
    inference = config.distillation.teacher_model.inference
    assert inference.max_model_len == inference.max_num_batched_tokens == teacher_context
    assert inference.prompt_length == 2048
    assert inference.response_length == response_length


@pytest.mark.parametrize("reserve", [0, -1, True, "invalid"])
def test_invalid_refinement_reserve_is_rejected(reserve):
    recipe = load_config(ROOT / "configs/qwen3_4b_trd.yaml")
    recipe["refine_prompt_token_reserve"] = reserve
    with pytest.raises(ValueError, match="refine_prompt_token_reserve"):
        build_command(recipe, Path("outputs/TEST"), [])


class Batch:
    """Minimal in-place DataProto interface, backed by real torch tensors."""

    def __init__(self):
        prompts = torch.tensor([[0, 65, 66], [67, 68, 69]])
        responses = torch.tensor([[111, 108, 100, 2, 0, 0], [98, 97, 100, 2, 0, 0]])
        self.batch = {
            "prompts": prompts, "responses": responses,
            "input_ids": torch.cat([prompts, responses], dim=-1),
            "response_mask": torch.tensor([[1, 1, 1, 1, 0, 0]] * 2),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]),
            "position_ids": torch.zeros((2, 9), dtype=torch.long),
            "rm_scores": torch.ones((2, 6)),
            "rollout_log_probs": torch.ones((2, 6)),
            "routed_experts": torch.ones((2, 9, 1, 1)),
            "old_log_probs": torch.ones((2, 6)),
        }
        self.non_tensor_batch = {
            "raw_prompt": [[{"role": "user", "content": "question A"}], [{"role": "user", "content": "question B"}]],
            "uid": ["A", "B"], "acc": [1, 1], "extras": ["old", "old"],
        }
        self.meta_info = {"reward_extra_keys": ["acc"], "temperature": 1.0}

    def union(self, other):
        for target, values in ((self.batch, other.batch), (self.non_tensor_batch, other.non_tensor_batch), (self.meta_info, other.meta_info)):
            for key, value in values.items():
                assert key not in target, f"stale data not removed: {key}"
                target[key] = value
        return self


def test_replace_response_rebuilds_masks_positions_and_removes_stale_values():
    batch = Batch()
    prompts = batch.batch["prompts"].clone()
    replace_responses(batch, [[70, 2], [80, 81, 82, 2]], pad_token_id=0)
    assert torch.equal(batch.batch["prompts"], prompts)
    assert batch.batch["responses"].tolist() == [[70, 2, 0, 0, 0, 0], [80, 81, 82, 2, 0, 0]]
    assert batch.batch["response_mask"].sum(-1).tolist() == [2, 4]
    assert batch.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]
    assert batch.batch["position_ids"].tolist() == [[0, 0, 1, 2, 3, 3, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6, 6, 6]]
    assert torch.equal(batch.batch["input_ids"], torch.cat([prompts, batch.batch["responses"]], dim=-1))
    for key in ("rm_scores", "rollout_log_probs", "old_log_probs", "routed_experts"):
        assert key not in batch.batch
    assert "acc" not in batch.non_tensor_batch
    assert "reward_extra_keys" not in batch.meta_info
    assert batch.non_tensor_batch["uid"] == ["A", "B"]


@pytest.mark.parametrize("responses", [[], [[1], []], [[1], [1] * 7]])
def test_bad_rewrite_batch_does_not_mutate_training_data(responses):
    batch = Batch()
    old = batch.batch["responses"].clone()
    with pytest.raises(ValueError):
        replace_responses(batch, responses, 0)
    assert torch.equal(old, batch.batch["responses"])
    assert "rm_scores" in batch.batch


class Teacher:
    tokenizer = Tokenizer()

    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail
        self.requests = []
        self.server_manager = self
        # Mirror native scoring configuration, whose response_length becomes 1.
        self.config = OmegaConf.create({"teacher_model": {"inference": {"max_model_len": 1006, "response_length": 1}}})

    def wake_up(self):
        self.events.append("teacher_wake")

    def sleep(self):
        self.events.append("teacher_sleep")

    def _run_single(self, coroutine):
        return asyncio.run(coroutine)

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        assert request_id
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        self.requests.append((prompt, dict(sampling_params)))
        assert sampling_params.pop("max_tokens") == 6  # Not native teacher's 1-token default.
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("generation failed")
        suffix = "A" if "question A" in prompt else "B"
        self.events.append("rewrite_" + suffix)
        return SimpleNamespace(token_ids=list(map(ord, "new" + suffix + "\x02")), stop_reason="completed")


class NativeTrainer:
    def init_workers(self):
        self.events.append("init_workers")

    def _compute_teacher_colocate(self, batch):
        self.teacher_model_manager.wake_up()
        assert batch.batch["responses"][:, :4].tolist() == [list(map(ord, "newA")), list(map(ord, "newB"))]
        assert batch.batch["prompts"].tolist() == [[0, 65, 66], [67, 68, 69]]
        self.events.append("score_y_r_with_original_prompt")
        self.teacher_model_manager.sleep()
        return "native_teacher_scores"


class Trainer(TRDTrajectoryMixin, NativeTrainer):
    def __init__(self, fail=False):
        self.events = ["student_rollout", "student_sleep"]
        self.teacher_model_manager = Teacher(self.events, fail=fail)
        self.tokenizer = Tokenizer()
        self.config = OmegaConf.create({
            "data": {"prompt_mode": "plaint", "max_prompt_length": 3},
            "trd": {"enabled": True, "max_prompt_length": 1000},
            "distillation": {"enabled": True, "teacher_model": {
                "enable_resource_pool": False, "inference": {
                    "max_model_len": 1006, "prompt_length": 3, "response_length": 6,
                    "max_num_batched_tokens": 1006,
                },
            }},
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "actor_rollout_ref": {"rollout": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                               "response_length": 6}},
        })
        self.reward_loop_manager = SimpleNamespace(compute_rm_score=self.reward)

    def reward(self, batch):
        self.events.append("grade_y_r")
        assert batch.batch["responses"][0, :4].tolist() == list(map(ord, "newA"))
        return SimpleNamespace(
            batch={"rm_scores": torch.zeros((2, 6))}, non_tensor_batch={"acc": [0, 0]},
            meta_info={"reward_extra_keys": ["acc"]},
        )


def test_teacher_generation_sleeps_before_native_scoring_and_regrades_y_r():
    trainer, batch = Trainer(), Batch()
    assert trainer._compute_teacher_colocate(batch) == "native_teacher_scores"
    assert trainer.events == [
        "student_rollout", "student_sleep", "teacher_wake", "rewrite_A", "rewrite_B", "teacher_sleep",
        "teacher_wake", "score_y_r_with_original_prompt", "teacher_sleep", "grade_y_r",
    ]
    assert batch.batch["responses"][:, 4].tolist() == [2, 2]  # Generated EOS is retained.
    assert batch.batch["response_mask"].sum(-1).tolist() == [5, 5]
    assert batch.non_tensor_batch["acc"] == [0, 0]
    for i, (prompt, params) in enumerate(trainer.teacher_model_manager.requests):
        assert "**Your Initial Solution:**\n" + ("old" if i == 0 else "bad") in prompt
        assert params == {"max_tokens": 6, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}


def test_teacher_generation_failure_still_sleeps_and_never_trains_y_o():
    trainer, batch = Trainer(fail=True), Batch()
    with pytest.raises(RuntimeError, match="generation failed"):
        trainer._compute_teacher_colocate(batch)
    assert trainer.events[-1] == "teacher_sleep"
    assert "score_y_r_with_original_prompt" not in trainer.events
    assert "rm_scores" in batch.batch


def test_shared_vocabulary_keeps_exact_sampled_tokens(monkeypatch):
    trainer = Trainer()
    # A decode/encode round trip need not retain the sampled segmentation.
    monkeypatch.setattr(trainer.tokenizer, "encode", lambda *args, **kwargs: [999])
    assert trainer._generate_refinements(Batch()) == [list(map(ord, "newA\x02")), list(map(ord, "newB\x02"))]


def test_refinement_uses_its_own_output_budget(monkeypatch):
    trainer = Trainer()
    trainer.config.trd.max_new_tokens = 3

    async def generate(*, request_id, prompt_ids, sampling_params):
        assert sampling_params["max_tokens"] == 3
        return SimpleNamespace(token_ids=[70, 2], stop_reason="completed")

    monkeypatch.setattr(trainer.teacher_model_manager, "generate", generate)
    assert trainer._generate_refinements(Batch()) == [[70, 2], [70, 2]]
    assert trainer.events[-2:] == ["teacher_wake", "teacher_sleep"]


def test_native_dataproto_handoff():
    protocol = pytest.importorskip("verl.protocol")
    from tensordict import TensorDict
    import numpy as np

    fixture = Batch()
    batch = protocol.DataProto(
        batch=TensorDict(fixture.batch, batch_size=[2]),
        non_tensor_batch={key: np.array(value, dtype=object) for key, value in fixture.non_tensor_batch.items()},
        meta_info=fixture.meta_info,
    )
    trainer = Trainer()
    original_reward = trainer.reward_loop_manager.compute_rm_score

    def reward(data):
        result = original_reward(data)
        return protocol.DataProto(
            batch=TensorDict(result.batch, batch_size=[2]),
            non_tensor_batch={key: np.array(value) for key, value in result.non_tensor_batch.items()},
            meta_info=result.meta_info,
        )

    trainer.reward_loop_manager.compute_rm_score = reward
    assert trainer._compute_teacher_colocate(batch) == "native_teacher_scores"
    assert batch.batch["response_mask"].sum(-1).tolist() == [5, 5]
    assert batch.non_tensor_batch["acc"].tolist() == [0, 0]
    assert batch.batch["input_ids"][0][batch.batch["attention_mask"][0].bool()].tolist() == [65, 66, *map(ord, "newA"), 2]
    assert "rollout_log_probs" not in batch.batch
    assert "old_log_probs" not in batch.batch


@pytest.mark.parametrize("mode", ["plaint", "thinking", "non-thinking"])
def test_teacher_generation_tokenizes_and_clips_the_whole_batch(monkeypatch, mode):
    trainer = Trainer()
    trainer.config.data.prompt_mode = mode
    trainer.config.trd.max_prompt_length = 600
    tokenization_batches = []
    original_tokenize = Tokenizer.__call__

    def tokenize(self, texts, **kwargs):
        assert trainer.events[-1] != "teacher_wake"
        tokenization_batches.append(len(texts))
        return original_tokenize(self, texts, **kwargs)

    monkeypatch.setattr(Tokenizer, "__call__", tokenize)
    monkeypatch.setattr(trainer.tokenizer, "batch_decode", lambda *args, **kwargs: ["x" * 23793, "bad"])
    for _ in range(2):
        assert trainer._compute_teacher_colocate(Batch()) == "native_teacher_scores"
        long_prompt, short_prompt = [prompt for prompt, _ in trainer.teacher_model_manager.requests[-2:]]
        assert len(long_prompt) == 600
        assert len(short_prompt) < 600
        assert "question A" in long_prompt and "**Instructions:**" in long_prompt
        assert "**Your Initial Solution:**\nbad" in short_prompt
    assert tokenization_batches == [6, 6]  # One batch call for all three segments of both samples.


def test_trd_rejects_rollout_bypass_before_initializing_gpus():
    trainer = Trainer()
    trainer.config.algorithm.rollout_correction.bypass_mode = True
    with pytest.raises(ValueError, match="old_log_probs recomputation"):
        trainer.init_workers()
    assert "init_workers" not in trainer.events


@pytest.mark.parametrize("requested,output_budget,refine_budget,required,expected", [
    (18433, 16384, 8192, 32768, 32768),
    (32768, 16384, 8192, 32768, 32768),
    (65536, 16384, 8192, 32768, 65536),
    (None, 16384, 8192, 32768, 32768),
    (32768, 32768, 8192, 34817, 34817),  # Scoring is the larger context requirement.
    (32768, 16384, 16384, 40960, 40960),
    (None, 16384, None, 40960, 40960),  # Omitted rewrite budget uses the student budget.
])
def test_final_hydra_overrides_reserve_full_rewrite_budget(requested, output_budget, refine_budget, required, expected):
    recipe = load_config(ROOT / "configs/qwen3_4b_trd.yaml")
    recipe["refine_max_prompt_length"] = 24576
    extras = [
        f"distillation.teacher_model.inference.max_model_len={requested if requested is not None else 'null'}",
        f"actor_rollout_ref.rollout.response_length={output_budget}",
    ]
    if refine_budget is None:
        recipe.pop("refine_max_new_tokens", None)
    else:
        extras.append(f"trd.max_new_tokens={refine_budget}")
    command = build_command(recipe, Path("outputs/TEST"), extras)
    with initialize_config_dir(config_dir=str(ROOT / "third_party/verl/verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=command[3:])
    assert configure_teacher_context(config) == required
    inference = config.distillation.teacher_model.inference
    assert inference.max_model_len == expected
    assert inference.max_num_batched_tokens >= expected
    assert inference.prompt_length == 2048  # Independent of 4096-token validation prompts.
    assert inference.response_length == output_budget
    before = OmegaConf.to_container(config, resolve=True)
    configure_teacher_context(config)
    assert OmegaConf.to_container(config, resolve=True) == before


def test_runtime_context_configuration_is_disabled_for_other_recipes():
    config = OmegaConf.create({"trd": {"enabled": False}})
    assert configure_teacher_context(config) is None
    assert OmegaConf.to_container(config) == {"trd": {"enabled": False}}


def test_actual_teacher_capacity_is_checked_before_initial_validation():
    trainer = Trainer()
    trainer.teacher_model_manager.config.teacher_model.inference.max_model_len = 1000
    with pytest.raises(ValueError, match="teacher initialization.*required_context=1006.*teacher_max_model_len=1000"):
        trainer.init_workers()
    assert "teacher_wake" not in trainer.events
    assert not trainer.teacher_model_manager.requests


def test_configured_generation_budget_is_independent_of_padded_response_width():
    trainer, batch = Trainer(), Batch()
    extra_padding = 2048 - batch.batch["responses"].shape[-1]
    for key in ("responses", "response_mask", "input_ids", "attention_mask", "position_ids"):
        batch.batch[key] = torch.nn.functional.pad(batch.batch[key], (0, extra_padding))
    responses = trainer._generate_refinements(batch)
    assert responses == [list(map(ord, "newA\x02")), list(map(ord, "newB\x02"))]
    assert all(params["max_tokens"] == 6 for _, params in trainer.teacher_model_manager.requests)


def test_configured_generation_budget_must_fit_response_storage():
    trainer = Trainer()
    trainer.config.actor_rollout_ref.rollout.response_length = 7
    with pytest.raises(ValueError, match="storage width=6.*budget=7"):
        trainer._generate_refinements(Batch())
    assert "teacher_wake" not in trainer.events
