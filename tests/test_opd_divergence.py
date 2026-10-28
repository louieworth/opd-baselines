"""HDL-inspired OPD response selection, sample-loss gradients, and H100 launch config."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from src.opd_options import options, selects_response_tokens
from src.opd_response_mask import response_loss_mask, sampled_token_divergence, select_actor_loss

ROOT = Path(__file__).resolve().parents[1]
RECIPE = 'qwen3_1_7b_thinking_opd_efficient_reasoning_sample_divergence_top1_h100_8gpu_train'
sys.path.insert(0, str(ROOT / 'scripts'))
import run_baseline


def rollout_batch():
    from verl.protocol import DataProto

    prompts = torch.tensor([[0, 1, 2], [3, 4, 5]])
    responses = torch.tensor([[6, 7, 8, 9, 10], [11, 12, 13, 0, 0]])
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]])
    old = torch.full((2, 5), -2.)
    # Non-response and padded rows must not win selection, even with NaNs.
    teacher = torch.full((2, 8, 1), float('nan'))
    teacher[:, 2:7, 0] = old + torch.tensor([[-.1, -9., -3., -1., -2.],
                                           [-11., -1., -1., float('nan'), float('nan')]])
    return DataProto(batch=TensorDict(dict(
        prompts=prompts, responses=responses, input_ids=torch.cat((prompts, responses), -1),
        attention_mask=attention, response_mask=attention[:, 3:].clone(),
        position_ids=attention.cumsum(-1).clamp_min(1) - 1, old_log_probs=old,
        teacher_logprobs=teacher, teacher_ids=torch.zeros_like(teacher, dtype=torch.long),
        advantages=torch.zeros_like(old),
    ), batch_size=2), non_tensor_batch={'acc': np.array([1., 0.])})


def test_top1_uses_absolute_gap_per_response_and_ignores_padding():
    batch = rollout_batch()
    original = batch.batch['response_mask'].clone()
    opts = options({'opd_loss_top_k_divergence': 1})
    actor_batch, metrics = select_actor_loss(batch, opts)
    assert actor_batch.batch['response_mask'].tolist() == [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
    assert metrics['actor/opd/loss_tokens'] == 2
    assert metrics['actor/opd/selected_divergence_mean'] == 10.
    assert metrics['actor/opd/selected_divergence_max'] == 11.
    torch.testing.assert_close(batch.batch['response_mask'], original)
    assert actor_batch.batch['input_ids'].data_ptr() == batch.batch['input_ids'].data_ptr()
    assert actor_batch.batch['attention_mask'].data_ptr() == batch.batch['attention_mask'].data_ptr()


def test_raw_scores_are_not_floored_or_clipped_and_do_not_require_grad():
    batch = rollout_batch()
    batch.batch['old_log_probs'].requires_grad_(True)
    batch.batch['teacher_logprobs'][0, 2:7, 0] = torch.tensor([-15., -35., -12., -13., -14.])
    scores = sampled_token_divergence(batch)
    assert scores[0].tolist() == [13., 33., 10., 11., 12.]
    assert not scores.requires_grad
    actor_batch, _ = select_actor_loss(batch, options({'opd_loss_top_k_divergence': 1}))
    assert actor_batch.batch['response_mask'][0].tolist() == [0, 1, 0, 0, 0]


def test_top1_handles_positive_and_negative_gaps_holes_ties_and_empty_rows():
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0]])
    gap = torch.tensor([[4., 100., -4., 1.], [float('nan')] * 4, [100., -3., 100., 100.]])
    selected = response_loss_mask(mask, divergence_scores=gap.abs(), top_k_divergence=1)
    assert selected.tolist() == [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]
    assert response_loss_mask(mask[:, :0], divergence_scores=gap[:, :0], top_k_divergence=1).shape == (3, 0)
    # Asking for more positions than exist never introduces padding tokens.
    torch.testing.assert_close(response_loss_mask(mask, divergence_scores=gap.abs(), top_k_divergence=8), mask)


def test_optional_tail_and_verified_filters_limit_divergence_candidates():
    batch = rollout_batch()
    opts = options(dict(opd_loss_top_k_divergence=1, opd_loss_last_n_tokens=2, opd_require_verified=True))
    actor_batch, _ = select_actor_loss(batch, opts)
    assert actor_batch.batch['response_mask'].tolist() == [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]


@pytest.mark.parametrize('field', ['old_log_probs', 'teacher_logprobs'])
def test_missing_scoring_inputs_fail_before_actor_update(field):
    batch = rollout_batch()
    batch.batch.pop(field)
    with pytest.raises(ValueError, match='requires old_log_probs and teacher_logprobs'):
        select_actor_loss(batch, options({'opd_loss_top_k_divergence': 1}))


def test_nonfinite_valid_scores_and_misaligned_teacher_fail():
    batch = rollout_batch()
    batch.batch['teacher_logprobs'][0, 3, 0] = float('nan')
    with pytest.raises(ValueError, match='must be finite'):
        select_actor_loss(batch, options({'opd_loss_top_k_divergence': 1}))
    batch.batch['teacher_logprobs'] = torch.zeros(2, 8, 2)
    with pytest.raises(ValueError, match='one sampled-token'):
        sampled_token_divergence(batch)


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '1'])
def test_invalid_divergence_k_is_rejected(value):
    with pytest.raises(ValueError, match='opd_loss_top_k_divergence'):
        options({'opd_loss_top_k_divergence': value})


@pytest.mark.parametrize('maximum', [None, 10.])
def test_native_sample_loss_gradient_only_reaches_top1_and_uses_selected_token_count(monkeypatch, maximum):
    monkeypatch.delenv('OPD_EXTENSIONS_CONFIG', raising=False)
    from src.opd_runtime import vocabulary_loss
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig, DistillationLossConfig
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    batch = rollout_batch()
    opts = options(dict(opd_loss_token_mode='sample', opd_loss_top_k_divergence=1))
    actor_batch, _ = select_actor_loss(batch, opts)
    data = left_right_2_no_padding(actor_batch.to_tensordict())
    # No valid teacher prediction is NaN; final/padding rows may be NaN.
    values = torch.linspace(-2., -1.5, int(batch.batch['attention_mask'].sum()), requires_grad=True)
    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=4, global_batch_size=4)
    actor = ActorConfig(strategy='fsdp', rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    actor.global_batch_info.update(dp_size=1, batch_num_tokens=9999, global_batch_size=99)
    distill = SimpleNamespace(distillation_loss=DistillationLossConfig(
        loss_mode='k1', use_task_rewards=False, use_policy_gradient=True, loss_max_clamp=maximum))
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(opts))
    loss, _ = vocabulary_loss(actor, distill, {'log_probs': values}, data)
    padded = no_padding_2_padding(values, data)
    teacher = no_padding_2_padding(data['teacher_logprobs'], data).squeeze(-1)
    gap = padded.detach().clamp_min(-10.) - teacher.clamp_min(-10.)
    if maximum is not None:
        gap = gap.clamp(-maximum, maximum)
    eligible = actor_batch.batch['response_mask'].bool()
    expected = ((padded - data['old_log_probs']).exp() * gap)[eligible].sum() / 2
    torch.testing.assert_close(loss, expected)
    actual_gradient, = torch.autograd.grad(loss, values, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, values)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient.count_nonzero() == 2
    assert torch.isfinite(actual_gradient).all()
    assert actor.global_batch_info['batch_num_tokens'] == 4


def test_trainer_selects_top1_before_native_actor_update(monkeypatch):
    monkeypatch.delenv('OPD_EXTENSIONS_CONFIG', raising=False)
    from src.train import BaselineTrainer
    from verl.protocol import DataProto

    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(options({'opd_loss_top_k_divergence': 1})))
    trainer = object.__new__(BaselineTrainer)
    trainer.config = SimpleNamespace(distillation=SimpleNamespace(enabled=True))
    trainer.use_legacy_worker_impl = 'disable'
    seen = []

    def update(batch):
        seen.append(batch)
        return DataProto.from_single_dict(data={}, meta_info={'metrics': {}})

    monkeypatch.setattr(trainer, '_update_actor_with_features', update)
    output = trainer._update_actor(rollout_batch())
    assert seen[0].batch['response_mask'].sum(-1).tolist() == [1, 1]
    assert output.meta_info['metrics']['actor/opd/loss_tokens'] == 2


def test_recipe_preserves_baseline_except_algorithm_and_destinations(monkeypatch, tmp_path):
    monkeypatch.setattr(run_baseline, 'model_reference', lambda model, **kwargs: model)
    monkeypatch.setattr(run_baseline, 'effective_model', lambda cfg: cfg['model_path'])
    cfg = run_baseline.load_config(ROOT / 'configs' / f'{RECIPE}.yaml')
    baseline = run_baseline.load_config(ROOT / 'configs/qwen3_1_7b_thinking_opd_efficient_reasoning_full_vocab_h100_8gpu_train.yaml')
    assert cfg['opd_loss_token_mode'] == 'sample'
    assert cfg['opd_loss_top_k_divergence'] == 1
    assert selects_response_tokens(cfg)
    for key in baseline:
        if key not in {'experiment_name', 'output_dir', 'opd_loss_token_mode', 'opd_loss_top_k_divergence'}:
            assert cfg[key] == baseline[key], key
    assert run_baseline.run_directory_name(cfg, 'same-time') != run_baseline.run_directory_name(baseline, 'same-time')
    command = run_baseline.build_command(cfg, tmp_path, [])
    config_dir = ROOT / 'third_party/verl/verl/trainer/config'
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        native = hydra.compose(config_name='ppo_trainer', overrides=command[3:])
    assert native.trainer.nnodes == 2 and native.trainer.n_gpus_per_node == 4
    assert native.data.train_batch_size == 128
    assert native.actor_rollout_ref.rollout.n == 1
    assert native.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert native.distillation.teacher_model.nnodes == 2
    assert native.distillation.teacher_model.n_gpus_per_node == 4
    assert native.distillation.distillation_loss.loss_mode == 'k1'
    assert native.distillation.distillation_loss.use_policy_gradient is True
    assert native.distillation.distillation_loss.use_task_rewards is False
    assert native.trainer.project_name == 'efficient-reasoning'
    env = native.ray_kwargs.ray_init.runtime_env
    assert json.loads(env.env_vars.OPD_EXTENSIONS_CONFIG)['opd_loss_top_k_divergence'] == 1
    assert env.worker_process_setup_hook == 'src.opd_h100_runtime.install'
    for role in [native.actor_rollout_ref.rollout, native.distillation.teacher_model.inference]:
        assert role.enforce_eager is True and role.enable_chunked_prefill is True
        assert role.max_num_batched_tokens == 4096 and role.max_num_seqs == 128
    script = (ROOT / 'scripts/sbatch' / f'{RECIPE}.sbatch').read_text()
    assert '#SBATCH --time=10:00:00' in script
    assert f'configs/{RECIPE}.yaml' in script
    assert (ROOT / 'scripts/h100_8gpu_env.sh').is_file()
    assert (ROOT / 'scripts/run_h100_8gpu_opd.py').is_file()


@pytest.mark.parametrize('runtime', [None, {'student': {'max_num_seqs': 0}},
                                     {'student': {'enforce_eager': 'true'}},
                                     {'student': {}, 'max_num_seqs': 1}])
def test_invalid_h100_runtime_settings_fail(runtime):
    with pytest.raises(ValueError, match='h100_runtime'):
        run_baseline.h100_inference_settings({'h100_runtime': runtime})
