"""CRISP-style self-teacher OPD: a periodically refreshed copy of the student scores the student's own
rollouts after a concise teacher prompt, with exact full-vocabulary reverse KL."""

import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from src import opd_self_teacher as self_teacher
from src.opd_math import reverse_kl_loss
from src.opd_options import options

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_baseline  # noqa: E402

FULL_VOCAB = dict(opd_loss_token_mode='full_vocab')


def test_self_teacher_options_are_validated():
    assert options({})['opd_teacher_source'] == 'vllm' and options({})['opd_teacher_refresh_steps'] == 0
    chosen = options(dict(FULL_VOCAB, opd_teacher_source='self', opd_teacher_refresh_steps=20))
    assert chosen['opd_teacher_source'] == 'self' and chosen['opd_teacher_refresh_steps'] == 20
    for bad in (dict(FULL_VOCAB, opd_teacher_source='peer'),
                dict(FULL_VOCAB, opd_teacher_source='self', opd_teacher_refresh_steps=-1),
                dict(FULL_VOCAB, opd_teacher_source='self', opd_teacher_refresh_steps=True),
                dict(opd_loss_token_mode='sample', opd_teacher_source='self'),  # needs the whole distribution
                dict(FULL_VOCAB, opd_teacher_refresh_steps=20)):                # only a self-teacher is refreshed
        with pytest.raises(ValueError):
            options(bad)


def test_teacher_becomes_the_student_after_every_refresh_period():
    # Update u is the u-th actor update; the teacher is the student after updates 20 and 40.
    assert [u for u in range(1, 57) if self_teacher.refresh_due(u, 20)] == [21, 41]
    assert not any(self_teacher.refresh_due(u, 0) for u in range(1, 57))


def test_refresh_copies_the_student_into_the_teacher():
    torch.manual_seed(0)
    student, teacher = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3).to(torch.bfloat16)
    self_teacher.copy_parameters(student, teacher)
    for s, t in zip(student.parameters(), teacher.parameters()):
        torch.testing.assert_close(t, s.detach().to(torch.bfloat16))
    with pytest.raises(ValueError):
        self_teacher.copy_parameters(student, torch.nn.Linear(4, 2))


def tiny_qwen3(tied):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
                         tie_word_embeddings=tied)
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize('tied', [True, False])
def test_teacher_hidden_states_and_head_reproduce_its_logits_for_each_response_token(tied):
    model = tiny_qwen3(tied)
    ids, prompt, response = [3, 7, 1, 9, 4, 4, 2, 8], 3, 5
    hidden = self_teacher.response_hidden(model, ids, prompt, response, 'cpu')
    weight, bias = self_teacher.gather_head(model, 'cpu', torch.float32)
    expected = model(input_ids=torch.tensor([ids])).logits[0, prompt - 1:prompt - 1 + response]
    assert hidden.shape == (response, 16) and bias is None
    torch.testing.assert_close(F.linear(hidden, weight), expected, atol=1e-5, rtol=1e-5)


def rollout(prompts, responses):
    """Left-padded prompts and right-padded responses, as the rollout hands them to the teacher manager."""
    from verl.protocol import DataProto
    width_p, width_r = max(map(len, prompts)), max(map(len, responses))
    rows, masks = [], []
    for p, r in zip(prompts, responses):
        rows.append([0] * (width_p - len(p)) + p + r + [0] * (width_r - len(r)))
        masks.append([0] * (width_p - len(p)) + [1] * (len(p) + len(r)) + [0] * (width_r - len(r)))
    ids, mask = torch.tensor(rows), torch.tensor(masks)
    return DataProto(batch=TensorDict(dict(input_ids=ids, attention_mask=mask, prompts=ids[:, :width_p],
                                           responses=ids[:, width_p:]), batch_size=len(rows)),
                     non_tensor_batch={'raw_prompt': np.array([[{'role': 'user', 'content': 'q' * len(p)}]
                                                               for p in prompts] + [None], dtype=object)[:-1]})


def test_self_teacher_scoring_passes_the_teacher_prompt_and_response_without_vllm(monkeypatch, tmp_path):
    from src import opd_runtime
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(dict(
        FULL_VOCAB, opd_teacher_source='self', opd_teacher_refresh_steps=20, teacher_prompt='Be concise.',
        teacher_enable_thinking=True, feature_cache=str(tmp_path))))

    def template(messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        return [90 + len(messages[0]['content'])]  # one token per formatted prompt

    async def generate(**kwargs):
        raise AssertionError('a self-teacher never calls the teacher server')

    manager = SimpleNamespace(_opd_tokenizer=SimpleNamespace(apply_chat_template=template), pad_token_id=0,
                              generate=generate)
    data = rollout([[5, 6, 7], [8, 9]], [[11, 12, 13, 14], [15, 16]])
    out = asyncio.run(opd_runtime.teacher_batch(manager, data))
    first, second = out.non_tensor_batch['opd_teacher_features']
    teacher_prompt = 90 + len('Be concise.\n\n' + 'qqq')
    assert first == dict(prompt_length=3, response_length=4, teacher_prompt_length=1,
                         teacher_input_ids=[teacher_prompt, 11, 12, 13, 14])
    assert second['teacher_input_ids'][1:] == [15, 16] and second['prompt_length'] == 2
    assert out.batch['teacher_logprobs'].abs().sum() == 0  # unused: the loss is the full-vocabulary KL


def test_self_teacher_loss_is_the_full_vocabulary_reverse_kl_on_response_positions(monkeypatch):
    from src import opd_runtime
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(dict(
        FULL_VOCAB, opd_teacher_source='self', opd_teacher_refresh_steps=20, opd_distribution_chunk_size=2,
        opd_token_top_k=4)))  # a toy vocabulary of 7
    torch.manual_seed(0)
    vocab, width = 7, 5
    ids = [1, 2, 3, 4, 5, 6]  # prompt 2 tokens, response 4 tokens
    student = torch.randn(1, len(ids), vocab, requires_grad=True)
    hidden, weight = torch.randn(4, width), torch.randn(vocab, width)
    calls = []

    def response_hidden(module, teacher_ids, prompt_length, response_length, device):
        calls.append((list(teacher_ids), prompt_length, response_length, device))
        return hidden

    monkeypatch.setattr(self_teacher, 'response_hidden', response_hidden)
    monkeypatch.setattr(self_teacher, 'output_head', lambda device: (weight, None))
    monkeypatch.setattr(self_teacher, 'teacher_module', lambda: 'teacher')
    reference = dict(prompt_length=2, response_length=4, teacher_prompt_length=3,
                     teacher_input_ids=[9, 9, 9, 3, 4, 5, 6])
    data = {'input_ids': torch.nested.nested_tensor([torch.tensor(ids)], layout=torch.jagged),
            'opd_teacher_features': [reference]}
    config = SimpleNamespace(distillation_loss=SimpleNamespace(loss_max_clamp=None))
    kl = opd_runtime.packed_outputs(student, data, config)['opd_kl'][0]
    expected = reverse_kl_loss(student[0, 1:5], F.linear(hidden, weight), 'full_vocab', top_k=4, maximum=None)
    assert calls == [([9, 9, 9, 3, 4, 5, 6], 3, 4, student.device)]
    torch.testing.assert_close(kl[1:5], expected)
    assert kl[0] == 0 and kl[5] == 0
    # The response tokens the teacher scores must be the student's own.
    reference['teacher_input_ids'] = [9, 9, 9, 3, 4, 5, 7]
    with pytest.raises(ValueError):
        opd_runtime.packed_outputs(student, data, config)


def crisp_recipe(monkeypatch, **changes):
    monkeypatch.setattr(run_baseline, 'model_reference', lambda model, **kwargs: model)
    monkeypatch.setattr(run_baseline, 'effective_model', lambda cfg: cfg['model_path'])
    cfg = run_baseline.load_config(
        ROOT / 'configs/qwen3_1_7b_thinking_opd_efficient_reasoning_full_vocab_h100_8gpu_train.yaml')
    cfg.update(teacher_model_path=cfg['model_path'], opd_teacher_source='self', opd_teacher_refresh_steps=20,
               teacher_enable_thinking=True, **changes)
    return cfg


def test_self_teacher_recipe_builds_the_reference_model_as_the_teacher(monkeypatch, tmp_path):
    cfg = crisp_recipe(monkeypatch)
    command = run_baseline.build_command(cfg, tmp_path, [])
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'third_party/verl/verl/trainer/config'), version_base=None):
        native = hydra.compose(config_name='ppo_trainer', overrides=command[3:])
    # use_kl_loss only creates the reference model; the KL-to-reference term itself has weight 0.
    assert native.actor_rollout_ref.actor.use_kl_loss is True
    assert native.actor_rollout_ref.actor.kl_loss_coef == 0.0
    opts = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.OPD_EXTENSIONS_CONFIG)
    assert opts['opd_teacher_source'] == 'self' and opts['opd_teacher_refresh_steps'] == 20


def test_self_teacher_must_be_the_student_model(monkeypatch, tmp_path):
    cfg = crisp_recipe(monkeypatch)
    cfg['teacher_model_path'] = 'Qwen/Qwen3-8B'
    with pytest.raises(ValueError, match='teacher_model_path'):
        run_baseline.build_command(cfg, tmp_path, [])


@pytest.mark.parametrize('recipe', ['qwen3_4b_crisp.yaml', 'qwen3_8b_crisp.yaml', 'qwen3_14b_crisp.yaml'])
def test_crisp_recipes_pass_the_trainer_config_validation(monkeypatch, tmp_path, recipe):
    from verl.utils.config import validate_config
    monkeypatch.setattr(run_baseline, 'model_reference', lambda model, **kwargs: model)
    monkeypatch.setattr(run_baseline, 'effective_model', lambda cfg: cfg['model_path'])
    command = run_baseline.build_command(run_baseline.load_config(ROOT / 'configs' / recipe), tmp_path, [])
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'third_party/verl/verl/trainer/config'), version_base=None):
        native = hydra.compose(config_name='ppo_trainer', overrides=command[3:])
    validate_config(native, use_reference_policy=True, use_critic=False)  # what the trainer checks before building
