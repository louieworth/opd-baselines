"""Shared Lightning-OPD numerics: FP32 updates, sequence means, sampled floors."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import hydra
import pytest
import torch
from tensordict import TensorDict
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_baseline

from verl.protocol import DataProto
from verl.trainer.distillation.losses import distillation_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, DistillationLossConfig, FSDPOptimizerConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.utils.padding import left_right_2_no_padding


RECIPES = [path for path in sorted((ROOT / 'configs').rglob('*.yaml'))
           if isinstance((cfg := yaml.safe_load(path.read_text())), dict)
           and cfg.get('method') in {'opd', 'grpo', 'adopd'}]


@pytest.mark.parametrize('path', RECIPES, ids=lambda p: str(p.relative_to(ROOT / 'configs')))
def test_all_recipes_use_shared_training_numerics(monkeypatch, tmp_path, path):
    monkeypatch.setattr(run_baseline, 'model_reference', lambda model, **kw: model)
    monkeypatch.setattr(run_baseline, 'effective_model', lambda cfg: cfg['model_path'])
    cfg = run_baseline.load_config(path)
    # Some older examples still contain relative eval paths. Normalize this
    # fixture without changing their datasets or editing those recipe files.
    for key in ('train_files', 'val_files'):
        values = cfg[key]
        resolve = lambda value: str((ROOT / value).resolve())
        cfg[key] = [resolve(v) for v in values] if isinstance(values, list) else resolve(values)
    before = deepcopy(cfg)
    command = run_baseline.build_command(cfg, tmp_path, [])
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'third_party/verl/verl/trainer/config'),
                                     version_base=None):
        native = hydra.compose(config_name='ppo_trainer', overrides=command[3:])
    actor = native.actor_rollout_ref.actor
    assert actor.fsdp_config.model_dtype == 'fp32'
    assert actor.fsdp_config.mixed_precision.param_dtype == 'bf16'
    assert actor.fsdp_config.mixed_precision.reduce_dtype == 'fp32'
    assert actor.loss_agg_mode == 'seq-mean-token-mean'
    assert actor.optim.optimizer_impl == 'torch.optim' and actor.optim.optimizer == 'AdamW'
    assert native.actor_rollout_ref.rollout.dtype == 'bfloat16'
    assert native.actor_rollout_ref.rollout.name == 'vllm'
    assert native.data.shuffle is False
    assert native.trainer.total_epochs == cfg['num_epochs']
    assert native.trainer.total_training_steps == cfg.get('total_training_steps')
    if cfg['method'] != 'grpo':
        assert native.distillation.distillation_loss.log_prob_min_clamp == -10.
        assert native.distillation.distillation_loss.loss_max_clamp == 10.
    assert cfg == before


def test_fp32_adamw_accumulates_small_updates_while_forward_remains_bf16():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        model.weight.fill_(.02)
    optimizer = build_optimizer(model.parameters(), FSDPOptimizerConfig(lr=1e-6, weight_decay=.01))
    initial = model.weight.detach().clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert model(torch.ones(1, 2)).dtype == torch.bfloat16
    for _ in range(100):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
    assert model.weight.dtype == torch.float32
    assert torch.all(model.weight < initial - 9e-5)
    assert optimizer.state[model.weight]['exp_avg'].dtype == torch.float32
    assert optimizer.state[model.weight]['exp_avg_sq'].dtype == torch.float32


def loss_batch():
    """Unequal response lengths, left/right padding, and deliberately stale stats."""
    prompts = torch.tensor([[1, 2], [0, 3]])
    responses = torch.tensor([[4, 5, 6], [7, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 0, 0]])
    q = torch.tensor([0., -12., -20., -1., 0., -3., 0.], requires_grad=True)
    teacher = torch.zeros(2, 5, 1)
    teacher[0, 1:4, 0] = torch.tensor([-3., -35., -90.])
    teacher[1, 1, 0] = -1.
    teacher.requires_grad_()
    old = torch.tensor([[-12., -20., -1.], [-3., 0., 0.]], requires_grad=True)
    batch = DataProto(batch=TensorDict(dict(
        prompts=prompts, responses=responses, input_ids=torch.cat([prompts, responses], -1),
        attention_mask=attention, position_ids=attention.cumsum(-1) - 1,
        response_mask=attention[:, 2:].clone(), old_log_probs=old,
        teacher_logprobs=teacher, teacher_ids=torch.zeros_like(teacher, dtype=torch.long),
    ), batch_size=2))
    return batch, q, teacher, old


def packed(batch, dp_size=1):
    data = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=4, global_batch_size=2)
    return data


def configs(minimum=-10., maximum=10.):
    actor = ActorConfig(strategy='fsdp', rollout_n=1, use_dynamic_bsz=True)
    actor.global_batch_info.update(dp_size=99, batch_num_tokens=9999, global_batch_size=999)
    distill = SimpleNamespace(distillation_loss=DistillationLossConfig(
        loss_mode='k1', use_task_rewards=False, log_prob_min_clamp=minimum, loss_max_clamp=maximum))
    return actor, distill


def lightning_costs(minimum, maximum):
    # Independent scalar oracle for Slime's -sampled_advantage.
    def cost(q, p):
        if minimum is not None:
            q, p = max(q, minimum), max(p, minimum)
        value = q - p
        return value if maximum is None else min(max(value, -maximum), maximum)
    return torch.tensor([cost(-12., -3.), cost(-20., -35.), cost(-1., -90.), cost(-3., -1.)])


@pytest.mark.parametrize('minimum', [None, -10.])
@pytest.mark.parametrize('maximum', [None, 2., 10.])
def test_native_sample_loss_matches_lightning_values_gradients_and_sequence_weights(minimum, maximum):
    batch, q, teacher, old = loss_batch()
    data = packed(batch)
    actor, distill = configs(minimum, maximum)
    loss, _ = distillation_loss(actor, distill, {'log_probs': q}, data)
    costs = lightning_costs(minimum, maximum)
    torch.testing.assert_close(loss, (costs[:3].mean() + costs[3]) / 2)
    loss.backward()
    expected = torch.zeros_like(q)
    expected[1:4] = costs[:3] / 6
    expected[5] = costs[3] / 2
    torch.testing.assert_close(q.grad, expected)
    # A log q below the floor STILL receives PG through the unfloored ratio.
    assert q.grad[1] != 0
    assert teacher.grad is None and old.grad is None
    assert actor.global_batch_info['dp_size'] == 1
    assert actor.global_batch_info['global_batch_size'] == 2
    assert actor.global_batch_info['batch_num_tokens'] == 4


@pytest.mark.parametrize('dp_size', [1, 2])
def test_native_sequence_mean_preserves_global_gradient_across_microbatches_and_dp(dp_size):
    batch, q, _, _ = loss_batch()
    actor, distill = configs()
    whole, _ = distillation_loss(actor, distill, {'log_probs': q}, packed(batch))
    full_gradient, = torch.autograd.grad(whole, q)
    parts = []
    for index, start, end in [(0, 0, 5), (1, 5, 7)]:
        micro = packed(batch[index:index+1], dp_size)
        part, _ = distillation_loss(actor, distill, {'log_probs': q[start:end]}, micro)
        parts.append(part)
    combined = sum(parts) / dp_size
    torch.testing.assert_close(combined, whole)
    gradient, = torch.autograd.grad(combined, q)
    torch.testing.assert_close(gradient, full_gradient)


@pytest.mark.parametrize('minimum,maximum', [(-10., 10.), (None, None)])
def test_adopd_answer_runtime_uses_the_same_sampled_cost_and_gradient(monkeypatch, minimum, maximum):
    from src.adopd_runtime import sampled_answer_loss
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_use_group_rollouts=False,
        adopd_length_penalty=0., adopd_reasoning_budget=8192)))
    batch, q, _, _ = loss_batch()
    batch.batch['adopd_teacher_log_probs'] = torch.tensor([[-3., -35., -90.], [-1., 0., 0.]])
    batch.batch['adopd_answer_mask'] = batch.batch['response_mask'].bool()
    batch.batch['adopd_trace_mask'] = torch.zeros(2, 3, dtype=torch.bool)
    batch.batch['adopd_trace_length'] = torch.zeros(2)
    batch.batch['adopd_normalizer'] = torch.full((2,), 2.)
    data = packed(batch)
    actor, distill = configs(minimum, maximum)
    ordinary, _ = distillation_loss(actor, distill, {'log_probs': q}, data)
    adopd, _ = sampled_answer_loss(actor, distill, {'log_probs': q}, data)
    torch.testing.assert_close(adopd, ordinary)
    native_gradient, = torch.autograd.grad(ordinary, q, retain_graph=True)
    adopd_gradient, = torch.autograd.grad(adopd, q)
    torch.testing.assert_close(adopd_gradient, native_gradient)


def test_adopd_group_advantage_uses_the_same_floored_answer_cost(monkeypatch):
    import numpy as np
    from src.adopd_trainer import AdopdTrainerMixin
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_use_group_rollouts=True, group_size=2,
        adopd_length_penalty=.1, adopd_reasoning_budget=8192)))
    trainer = object.__new__(AdopdTrainerMixin)
    _, distill = configs()
    trainer.config = SimpleNamespace(distillation=distill)
    batch = DataProto(batch=TensorDict(dict(old_log_probs=torch.tensor([[-12.], [-20.]]),
        adopd_teacher_log_probs=torch.tensor([[-30.], [-40.]]),
        adopd_answer_mask=torch.ones(2, 1, dtype=torch.bool),
        adopd_trace_mask=torch.ones(2, 1, dtype=torch.bool), adopd_trace_length=torch.ones(2)),
        batch_size=2), non_tensor_batch={'uid': np.array(['same', 'same'])})
    result = trainer._compute_advantage(batch)
    # Both floored costs are zero. Their raw gaps (18 and 20) would produce a
    # nonzero group preference, inconsistent with the answer loss being trained.
    torch.testing.assert_close(result.batch['adopd_advantage'], torch.zeros(2))
