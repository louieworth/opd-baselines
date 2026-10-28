"""CPU regression checks for tail-only OPD loss and answer verification."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from src.opd_logging import comparable_metrics
from src.opd_math import METRIC_NAMES
from src.opd_options import options
from src.opd_response_mask import response_loss_mask, select_actor_loss

ROOT = Path(__file__).resolve().parents[1]
BASE = 'qwen3_1_7b_thinking_opd_efficient_reasoning_sample'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_baseline


def test_last_100_counts_valid_tokens_per_response():
    lengths = torch.tensor([105, 100, 99, 1, 0])
    mask = (torch.arange(110).unsqueeze(0) < lengths.unsqueeze(-1)).long()
    original = mask.clone()
    selected = response_loss_mask(mask, 100)
    assert selected.sum(-1).tolist() == [100, 100, 99, 1, 0]
    assert selected[0, :5].count_nonzero() == 0
    assert selected[0, 5:105].all()
    assert selected[0, 105:].count_nonzero() == 0
    torch.testing.assert_close(mask, original)
    # An existing masked position remains excluded and does not consume a token.
    holes = torch.tensor([[1, 0, 1, 1, 0, 1, 0]])
    assert response_loss_mask(holes, 2).tolist() == [[0, 0, 0, 1, 0, 1, 0]]


def test_verified_requires_a_correct_answer_and_keeps_short_responses():
    mask = torch.tensor([[1, 1, 0]] * 5)
    acc = np.array([1.0, 0.0, 0.5, np.nan, 1.0])
    assert response_loss_mask(mask, 100, acc).sum(-1).tolist() == [2, 0, 0, 0, 2]
    torch.testing.assert_close(response_loss_mask(mask, None), mask)
    with pytest.raises(ValueError, match='one acc value'):
        response_loss_mask(mask, 100, [1, 0])


def rollout_batch():
    from verl.protocol import DataProto

    prompts = torch.tensor([[0, 11, 12], [13, 14, 15]])
    responses = torch.arange(100, 205).repeat(2, 1)
    attention = torch.ones(2, 108, dtype=torch.long)
    attention[0, 0] = 0
    attention[1, -3:] = 0
    tensors = dict(prompts=prompts, responses=responses,
                   input_ids=torch.cat((prompts, responses), -1), attention_mask=attention,
                   position_ids=attention.cumsum(-1).clamp_min(1) - 1,
                   response_mask=attention[:, 3:].clone(),
                   teacher_logprobs=torch.full((2, 108, 1), -2.5),
                   teacher_ids=torch.zeros(2, 108, 1, dtype=torch.long),
                   advantages=torch.zeros(2, 105))
    return DataProto(batch=TensorDict(tensors, batch_size=2),
                     non_tensor_batch={'acc': np.array([1., 0.])})


def packed_loss_inputs(verified=True):
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    batch = rollout_batch()
    opts = options(dict(opd_loss_last_n_tokens=100, opd_require_verified=verified))
    actor_batch, _ = select_actor_loss(batch, opts)
    data = left_right_2_no_padding(actor_batch.to_tensordict())
    values = torch.linspace(-2., -1., int(batch.batch['attention_mask'].sum()), requires_grad=True)
    data['old_log_probs'] = no_padding_2_padding(values.detach(), data)
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=int(actor_batch.batch['response_mask'].sum()) * 2,
                         global_batch_size=4)
    output = {'log_probs': values}
    output.update({name: torch.ones_like(values) * (i + 1) for i, name in enumerate(METRIC_NAMES)})
    return batch, actor_batch, data, values, output


def native_configs():
    from verl.workers.config import ActorConfig, DistillationLossConfig

    actor = ActorConfig(strategy='fsdp', rollout_n=1, use_dynamic_bsz=True)
    actor.global_batch_info.update(dp_size=1, batch_num_tokens=9999, global_batch_size=99)
    distill = SimpleNamespace(distillation_loss=DistillationLossConfig(loss_mode='k1', use_task_rewards=False))
    return actor, distill


@pytest.mark.parametrize('verified', [False, True])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_native_k1_loss_only_backpropagates_selected_response_tokens(monkeypatch, verified, diagnostics):
    monkeypatch.delenv('OPD_EXTENSIONS_CONFIG', raising=False)
    from src.opd_runtime import vocabulary_loss
    from verl.workers.utils.padding import no_padding_2_padding

    opts = options(dict(opd_loss_last_n_tokens=100, opd_require_verified=verified,
                        opd_enable_metrics=diagnostics))
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(opts))
    batch, actor_batch, data, values, output = packed_loss_inputs(verified)
    original_mask = batch.batch['response_mask'].clone()
    actor, distill = native_configs()
    loss, metrics = vocabulary_loss(actor, distill, output, data)
    padded = no_padding_2_padding(values, data)
    mask = actor_batch.batch['response_mask']
    # Inherit the sample k1 policy-gradient objective, including its stop-gradient.
    token_loss = (padded - data['old_log_probs']).exp() * (padded.detach() + 2.5)
    expected = ((token_loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).sum()
    expected = expected / data['global_batch_size'] * data['dp_size']
    torch.testing.assert_close(loss, expected)
    actual_gradient, = torch.autograd.grad(loss, values, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient.count_nonzero().item() == (100 if verified else 200)
    assert actor.global_batch_info['batch_num_tokens'] == data['batch_num_tokens']
    assert torch.isfinite(loss)
    torch.testing.assert_close(batch.batch['response_mask'], original_mask)
    assert actor_batch.batch['input_ids'].data_ptr() == batch.batch['input_ids'].data_ptr()
    assert actor_batch.batch['attention_mask'].data_ptr() == batch.batch['attention_mask'].data_ptr()
    if diagnostics:
        assert metrics['opd/valid_tokens'].aggregate() == mask.sum().item()


def test_empty_microbatch_has_zero_gradient_and_compatible_dp_metrics(monkeypatch):
    monkeypatch.delenv('OPD_EXTENSIONS_CONFIG', raising=False)
    from src.opd_runtime import vocabulary_loss
    from verl.utils.metric import Metric

    opts = options(dict(opd_loss_last_n_tokens=100, opd_require_verified=True, opd_enable_metrics=True))
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(opts))
    _, _, data, values, output = packed_loss_inputs()
    actor, distill = native_configs()
    _, active_metrics = vocabulary_loss(actor, distill, output, data)
    empty = data.clone(recurse=False)
    empty['response_mask'] = torch.zeros_like(data['response_mask'])
    loss, empty_metrics = vocabulary_loss(actor, distill, output, empty)
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert values.grad.count_nonzero() == 0
    assert active_metrics.keys() == empty_metrics.keys()
    for name, active in active_metrics.items():
        if isinstance(active, Metric):
            assert np.isfinite(Metric.aggregate_dp([active, empty_metrics[name]]))
        else:
            assert np.isfinite(empty_metrics[name])
    assert empty['response_mask'].count_nonzero() == 0


@pytest.mark.parametrize('all_wrong', [False, True])
def test_trainer_filters_only_actor_batch_and_skips_empty_updates(monkeypatch, all_wrong):
    monkeypatch.delenv('OPD_EXTENSIONS_CONFIG', raising=False)
    from src.train import BaselineTrainer
    from verl.protocol import DataProto

    opts = options(dict(opd_loss_last_n_tokens=100, opd_require_verified=True))
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(opts))
    batch = rollout_batch()
    if all_wrong:
        batch.non_tensor_batch['acc'][:] = 0
    original_mask = batch.batch['response_mask'].clone()
    trainer = object.__new__(BaselineTrainer)
    trainer.config = SimpleNamespace(distillation=SimpleNamespace(enabled=True))
    trainer.use_legacy_worker_impl = 'disable'
    updates, cleanups = [], []

    def update(actor_batch):
        updates.append(actor_batch)
        return DataProto.from_single_dict(data={}, meta_info={'metrics': {'actor/distillation/loss': 1.}})

    monkeypatch.setattr(trainer, '_update_actor_with_features', update)
    monkeypatch.setattr(trainer, '_clear_teacher_features', cleanups.append)
    result = trainer._update_actor(batch)
    metrics = result.meta_info['metrics']
    assert len(updates) == (0 if all_wrong else 1)
    assert cleanups == ([None] if all_wrong else [])
    assert metrics['actor/opd/selected_trajectories'] == (0 if all_wrong else 1)
    assert metrics['actor/opd/update_skipped'] == int(all_wrong)
    assert metrics['actor/opd/loss_tokens'] == (0 if all_wrong else 100)
    logged = comparable_metrics(metrics, 1)
    assert logged['train/opd/filtered_trajectories'] == (2 if all_wrong else 1)
    if updates:
        assert updates[0].batch['response_mask'].sum(-1).tolist() == [100, 0]
    torch.testing.assert_close(batch.batch['response_mask'], original_mask)


def test_missing_verifier_result_is_not_silently_accepted():
    batch = rollout_batch()
    batch.non_tensor_batch.pop('acc')
    with pytest.raises(ValueError, match='verifier acc'):
        select_actor_loss(batch, options(dict(opd_require_verified=True)))


@pytest.mark.parametrize('verified', [False, True])
def test_variants_preserve_sample_recipe_and_propagate_to_ray(monkeypatch, tmp_path, verified):
    monkeypatch.setattr(run_baseline, 'model_reference', lambda model, **kwargs: model)
    base = run_baseline.load_config(ROOT / 'configs' / f'{BASE}.yaml')
    suffix = 'last100_verified' if verified else 'last100'
    cfg = run_baseline.load_config(ROOT / 'configs' / f'{BASE}_{suffix}.yaml')
    assert cfg['opd_loss_last_n_tokens'] == 100
    assert cfg['opd_require_verified'] is verified
    for key, value in base.items():
        if key not in {'experiment_name', 'opd_loss_last_n_tokens', 'opd_require_verified'}:
            assert cfg[key] == value, key
    assert cfg['experiment_name'] != base['experiment_name']
    assert run_baseline.run_directory_name(cfg, 'same-time') != run_baseline.run_directory_name(base, 'same-time')
    other = {**cfg, 'opd_require_verified': not verified}
    assert run_baseline.run_directory_name(cfg, 'same-time') != run_baseline.run_directory_name(other, 'same-time')
    command = run_baseline.build_command(cfg, tmp_path, [])
    directory = ROOT / 'third_party/verl/verl/trainer/config'
    with hydra.initialize_config_dir(config_dir=str(directory), version_base=None):
        native = hydra.compose(config_name='ppo_trainer', overrides=command[3:])
    extension = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.OPD_EXTENSIONS_CONFIG)
    assert extension['opd_loss_last_n_tokens'] == 100
    assert extension['opd_require_verified'] is verified
    assert native.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook == 'src.opd_runtime.install'
    assert native.actor_rollout_ref.actor.loss_agg_mode == 'seq-mean-token-mean'


@pytest.mark.parametrize('key,value', [('opd_loss_last_n_tokens', 0), ('opd_loss_last_n_tokens', -1),
                                     ('opd_loss_last_n_tokens', True), ('opd_loss_last_n_tokens', 1.5),
                                     ('opd_loss_last_n_tokens', '100'), ('opd_require_verified', 'false')])
def test_response_selection_options_are_validated(key, value):
    with pytest.raises(ValueError, match=key):
        options({key: value})
