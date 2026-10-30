"""CPU numerical and integration checks for sampled-token V2 ADOPD."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_baseline
from src.adopd_cache import TraceCache, cache_identity
from src.adopd_math import (group_advantages, masked_sequence_mean, per_sequence_loss,
                            response_masks, sampled_kl, length_policy_gradient)
from src.adopd_options import options
from src.adopd_rollout import budget_rollout
from src.adopd_trainer import AdopdTrainerMixin


def recipe():
    return run_baseline.load_config(ROOT / 'configs/qwen3_4b_adopd_v2.yaml')


@pytest.mark.parametrize('mode', ['offline', 'online'])
def test_recipe_accepts_offline_or_online_wandb(tmp_path, mode):
    import yaml
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump({**recipe(), 'wandb_mode': mode}))
    assert run_baseline.load_config(path)['wandb_mode'] == mode
    path.write_text(yaml.safe_dump({**recipe(), 'wandb_mode': 'sometimes'}))
    with pytest.raises(ValueError, match='wandb_mode'):
        run_baseline.load_config(path)


def test_colocated_vllm_engines_get_disjoint_ports_below_ephemeral_range():
    from src.vllm_ports import PORTS_PER_ENGINE, engine_port
    names = [f'vllm_server_{kind}{replica}_0' for kind in ('', 'teacher_') for replica in range(16)]
    starts = {engine_port(name) for name in names}
    assert len(starts) == 32  # one range per student and per teacher replica
    ranges = sorted((start, start + PORTS_PER_ENGINE) for start in starts)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    assert ranges[0][0] >= 1024 and ranges[-1][1] <= 32768  # Linux ephemeral ports start at 32768
    assert engine_port('not_a_server') is None


def test_vllm_server_launch_pins_its_port_range(monkeypatch):
    import ray
    from src import vllm_ports
    class Server:
        async def launch_server(self, *args):
            return os.environ.get('VLLM_PORT')
    monkeypatch.setitem(sys.modules, 'verl.workers.rollout.vllm_rollout.vllm_async_server',
                        SimpleNamespace(vLLMHttpServer=Server))
    monkeypatch.setattr(ray, 'get_runtime_context',
                        lambda: SimpleNamespace(get_actor_name=lambda: 'vllm_server_teacher_3_0'))
    monkeypatch.delenv('VLLM_PORT', raising=False)
    vllm_ports.install()
    assert asyncio.run(Server().launch_server()) == str(vllm_ports.engine_port('vllm_server_teacher_3_0'))
    monkeypatch.delenv('VLLM_PORT', raising=False)


def test_worker_loaded_files_resolve_outside_the_repository(tmp_path, monkeypatch):
    # Ray workers on a long-lived cluster do not start in the repository.
    monkeypatch.chdir(tmp_path)
    native = compose(recipe(), tmp_path)
    for value in (native.actor_rollout_ref.rollout.agent.agent_loop_config_path,
                  native.reward.custom_reward_function.path):
        assert Path(value).is_absolute() and Path(value).is_file()


def compose(cfg, tmp_path, extra=()):
    command = run_baseline.build_command(cfg, tmp_path, list(extra))
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'third_party/verl/verl/trainer/config'), version_base=None):
        return hydra.compose(config_name='ppo_trainer', overrides=command[3:])


@pytest.mark.parametrize('nnodes', [2, 4])
@pytest.mark.parametrize('group', [False, True])
@pytest.mark.parametrize('mode', ['sample', 'full_vocab'])
def test_recipe_native_config_and_workers(group, nnodes, mode, tmp_path):
    cfg = recipe()
    cfg['adopd_use_group_rollouts'] = group
    cfg['h100_nnodes'] = nnodes
    cfg['opd_loss_token_mode'] = mode
    native = compose(cfg, tmp_path)
    g = 8 if group else 1
    gpus = cfg['n_gpus_per_node']
    trajectories = 16 * gpus * nnodes
    assert native.trainer.nnodes == nnodes and native.trainer.n_gpus_per_node == gpus
    assert native.data.train_batch_size == trajectories // g
    assert native.actor_rollout_ref.actor.ppo_mini_batch_size == trajectories // g
    assert native.actor_rollout_ref.rollout.n == g
    assert native.trainer.total_epochs == 1 and native.trainer.total_training_steps is None
    assert native.algorithm.rollout_correction is None
    assert native.actor_rollout_ref.rollout.response_length == 16384
    assert native.actor_rollout_ref.rollout.max_model_len == 4096 + 16384
    assert native.distillation.teacher_model.inference.max_model_len == 2048 + 16384 + 16384 + 1
    assert native.distillation.distillation_loss.loss_max_clamp == (10. if mode == 'sample' else None)
    assert native.distillation.distillation_loss.log_prob_min_clamp == (-10. if mode == 'sample' else None)
    assert native.actor_rollout_ref.rollout.agent.default_agent_loop == 'adopd_agent'
    assert native.data.custom_cls.path == 'pkg://src.adopd_dataset'
    assert native.data.custom_cls.name == 'AdopdDataset'
    assert native.data.adopd_reasoning_budget == cfg['adopd_reasoning_budget']
    assert native.distillation.teacher_model.model_path == native.actor_rollout_ref.model.path
    assert set(Path(p).stem for p in native.data.val_files) == {'amc23', 'aime25', 'aime26'}
    env = native.ray_kwargs.ray_init.runtime_env
    opts = json.loads(env.env_vars.ADOPD_CONFIG)
    assert opts['adopd_use_group_rollouts'] is group
    assert opts['adopd_reasoning_budget'] == 8192 and opts['adopd_length_penalty'] == .1
    prefix = 'sampled' if mode == 'sample' else 'full_vocab'
    expected_objective = f'{prefix}_answer_kl_plus_' + ('group_reinforce' if group else 'length_pg')
    assert opts['objective'] == expected_objective
    assert env.worker_process_setup_hook == 'src.adopd_runtime.install'
    features = json.loads(env.env_vars.OPD_EXTENSIONS_CONFIG)
    if mode == 'sample':
        assert features == {}
    else:
        assert features['opd_loss_token_mode'] == mode
        assert features['feature_cache'] == str(tmp_path / 'adopd_teacher_features')
    assert mode in native.trainer.experiment_name
    assert mode in run_baseline.run_directory_name(cfg, 'time')
    assert f'_g{g}' in native.trainer.experiment_name
    assert f'g{g}' in run_baseline.run_directory_name(cfg, 'time')
    from verl.utils.config import validate_config, omega_conf_to_dataclass
    validate_config(native, use_reference_policy=False, use_critic=False)
    teacher = omega_conf_to_dataclass(native.distillation).teacher_model.inference
    assert teacher.prompt_length == 2048 + 16384 + 16384 and teacher.response_length == 1


@pytest.mark.parametrize('mode', ['sample', 'full_vocab'])
def test_eval_preserves_budget_without_teacher_or_cache(tmp_path, monkeypatch, mode):
    cfg = recipe()
    cfg.update(_adopd_eval_only=True, adopd_use_group_rollouts=True, opd_loss_token_mode=mode,
               teacher_model_path='/missing/teacher', adopd_teacher_trace_cache='/missing/cache')
    native = compose(cfg, tmp_path, ['trainer.val_only=true', 'trainer.val_before_train=true'])
    assert not native.distillation.enabled
    assert native.trainer.val_only
    assert native.actor_rollout_ref.rollout.n == 1
    assert native.data.custom_cls.name == 'AdopdDataset'
    assert native.data.adopd_reasoning_budget == 8192
    opts = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.ADOPD_CONFIG)
    assert opts['enabled'] is False and opts['adopd_reasoning_budget'] == 8192
    assert 'cache_identity' not in opts
    assert native.ray_kwargs.ray_init.runtime_env.env_vars.OPD_EXTENSIONS_CONFIG == '{}'


@pytest.mark.parametrize('key,value', [('adopd_use_group_rollouts', 'true'),
    ('adopd_reasoning_budget', 16383), ('adopd_length_penalty', -1),
    ('adopd_length_penalty', float('nan')), ('adopd_teacher_trace_cache', 'relative'),
    ('adopd_cache_seed', True), ('adopd_cache_batch_size', 0), ('adopd_eval_plain_prompt', 'false'),
    ('opd_loss_token_mode', 'top_k_teacher'), ('opd_distribution_chunk_size', 0)])
def test_invalid_options(key, value):
    with pytest.raises(ValueError):
        options({key: value})


def test_eval_runs_the_plain_prompt_by_default():
    assert options({})['adopd_eval_plain_prompt'] is True


def test_plain_prompt_eval_can_be_limited_to_the_initial_evaluation():
    assert options({'adopd_eval_plain_prompt': 'initial'})['adopd_eval_plain_prompt'] == 'initial'


def test_sampled_kl_value_and_expected_gradient_match_exact_reverse_kl():
    logits = torch.tensor([.3, -.7, 1.1], requires_grad=True)
    teacher = torch.tensor([-.2, .4, -.5], requires_grad=True).log_softmax(0)
    logq = logits.log_softmax(0)
    old = logq.detach()
    q = old.exp()
    sampled = sampled_kl(logq, teacher, old)
    torch.testing.assert_close(sampled, logq - teacher.detach())
    objective = (q * sampled).sum()
    expected = (logq.exp() * (logq - teacher.detach())).sum()
    actual_gradient, = torch.autograd.grad(objective, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert teacher.grad_fn is not None  # detached inside the objective, not at caller


def test_loss_uses_only_sampled_answer_tokens_and_optional_trace_term():
    q = torch.tensor([[-2., -1., -3., -4., -5.]], requires_grad=True)
    p = torch.tensor([[-9., -9., -2., -4.5, -99.]], requires_grad=True)
    old = q.detach().clone()
    answer = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    trace = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool) # forced token at position 1
    losses, distortion, reinforce = per_sequence_loss(q, p, old, answer, trace)
    torch.testing.assert_close(losses, torch.tensor([-.25]))
    assert not reinforce.any()
    losses.sum().backward()
    torch.testing.assert_close(q.grad, torch.tensor([[0., 0., -.5, .25, 0.]]))
    assert p.grad is None
    q.grad = None
    advantage = torch.tensor([2.], requires_grad=True)
    losses, _, reinforce = per_sequence_loss(q, p, old, answer, trace, advantage)
    torch.testing.assert_close(reinforce, torch.tensor([4.]))
    losses.sum().backward()
    assert q.grad[0, 0] == -2 and q.grad[0, 1] == 0
    assert advantage.grad is None


def test_single_rollout_length_pg_preserves_cost_and_masks_forced_halt_answer_padding():
    log_probs = torch.tensor([[-1., -2., -5., -4., -8.], [-1., -2., -3., -4., -8.]],
                             requires_grad=True)
    # Two content tokens, then respectively a forced halt and a natural halt.
    trace = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    lengths = torch.tensor([2., 2.], requires_grad=True)
    objective = length_policy_gradient(log_probs, trace, lengths, budget=8, penalty=.1)
    torch.testing.assert_close(objective, torch.tensor([.025, .025]))
    objective.sum().backward()
    torch.testing.assert_close(log_probs.grad,
        torch.tensor([[.025, .025, 0., 0., 0.], [.025, .025, .025, 0., 0.]]))
    assert lengths.grad is None


def test_length_pg_expected_gradient_matches_exact_expected_length_and_shortens_policy():
    # Enumerate a two-step continue/stop policy, including the hard-budget branch.
    logits = torch.tensor([.4, -.3], dtype=torch.float64, requires_grad=True)
    forced_halt_log_prob = torch.tensor(-5., dtype=torch.float64, requires_grad=True)
    p = logits.sigmoid()
    zero = logits.new_zeros(())
    log_probs = torch.stack([
        torch.stack([(1-p[0]).log(), zero, zero]),                   # stop immediately
        torch.stack([p[0].log(), (1-p[1]).log(), zero]),             # continue, natural stop
        torch.stack([p[0].log(), p[1].log(), forced_halt_log_prob]), # continue twice, forced stop
    ])
    trace = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    lengths = torch.tensor([0., 1., 2.], dtype=torch.float64)
    probabilities = torch.stack([1-p[0], p[0]*(1-p[1]), p[0]*p[1]])
    estimates = length_policy_gradient(log_probs, trace, lengths, budget=2, penalty=.1)
    actual = (probabilities.detach() * estimates).sum()
    expected = .1 / 2 * p[0] * (1+p[1])
    torch.testing.assert_close(actual, expected)
    actual_gradient, forced_gradient = torch.autograd.grad(
        actual, (logits, forced_halt_log_prob), retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert forced_gradient == 0
    shorter_p = (logits.detach() - .25 * actual_gradient).sigmoid()
    assert shorter_p[0] * (1+shorter_p[1]) < p[0] * (1+p[1])


def test_single_rollout_combines_answer_kl_with_length_gradient_and_lambda_zero_ablation():
    q = torch.tensor([[-2., -1., -3., -4., -5.]], requires_grad=True)
    teacher = torch.tensor([[-9., -9., -2., -4.5, -99.]], requires_grad=True)
    answer = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    trace = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    for penalty in [0., .2]:
        q.grad = None
        loss, kl, trace_objective = per_sequence_loss(q, teacher, q.detach(), answer, trace,
            trace_lengths=torch.tensor([1]), budget=4, length_penalty=penalty)
        torch.testing.assert_close(loss, torch.tensor([-.25 + penalty / 4]))
        torch.testing.assert_close(trace_objective, torch.tensor([penalty / 4]))
        loss.sum().backward()
        torch.testing.assert_close(q.grad, torch.tensor([[penalty/4, 0., -.5, .25, 0.]]))
        assert teacher.grad is None


def test_group_loss_does_not_add_single_rollout_length_penalty_again():
    q = torch.tensor([[-2., -1., -3.]], requires_grad=True)
    teacher = torch.tensor([[-9., -9., -2.]])
    answer = torch.tensor([[0, 0, 1]], dtype=torch.bool)
    trace = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    advantages = torch.tensor([.5], requires_grad=True)
    baseline = per_sequence_loss(q, teacher, q.detach(), answer, trace, advantages)[0]
    result = per_sequence_loss(q, teacher, q.detach(), answer, trace, advantages,
        trace_lengths=torch.tensor([1]), budget=8, length_penalty=.1)[0]
    torch.testing.assert_close(result, baseline)
    baseline_gradient, = torch.autograd.grad(baseline.sum(), q, retain_graph=True)
    result_gradient, = torch.autograd.grad(result.sum(), q)
    torch.testing.assert_close(result_gradient, baseline_gradient)
    assert advantages.grad is None


def test_group_advantages_are_global_uid_based_detached_and_zero_variance_safe():
    d = torch.tensor([.2, 9., .8, 9.], requires_grad=True)
    uids = ['a', 'b', 'a', 'b']
    lengths = torch.tensor([2, 1, 8, 1])
    a = group_advantages(d, lengths, uids, torch.ones(4, dtype=torch.bool), 2, 10, .1)
    assert a[0] > 0 and a[2] < 0 and a[1] == a[3] == 0
    assert not a.requires_grad
    perm = torch.tensor([3, 2, 0, 1])
    actual = group_advantages(d[perm], lengths[perm], [uids[i] for i in perm],
                              torch.ones(4, dtype=torch.bool), 2, 10, .1)
    torch.testing.assert_close(actual, a[perm])
    with pytest.raises(ValueError, match='Expected 8'):
        group_advantages(d, lengths, uids, torch.ones(4, dtype=torch.bool), 8, 10, .1)


def test_halt_masks_forced_natural_padding_and_answer_boundary():
    responses = torch.tensor([[4, 5, 99, 6, 0], [4, 99, 8, 9, 10]])
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    policy = valid.clone(); policy[0, 2] = 0
    answer, trace, length, forced = response_masks(responses, policy, valid, 99, 2)
    assert answer.tolist() == [[False, False, False, True, False], [False, False, True, True, True]]
    assert trace.tolist() == [[True, True, False, False, False], [True, True, False, False, False]]
    assert length.tolist() == [2, 1] and forced.tolist() == [True, False]
    with pytest.raises(ValueError, match='no </think>'):
        response_masks(torch.zeros(1, 3), torch.ones(1, 3), torch.ones(1, 3), 99, 2)


@pytest.fixture
def budget_tokenizer():
    def encode_messages(messages, **kwargs):
        return [101] + [ord(char) for item in messages for char in item['content']] + [102]
    return SimpleNamespace(apply_chat_template=encode_messages,
        encode=lambda *a, **k: [99], eos_token_id=0, bos_token_id=0,
        decode=lambda tokens, **kwargs: ''.join(str(token) for token in tokens))


@pytest.mark.parametrize('budget', [512, 8192])
def test_budget_prompt_preserves_question_and_uses_configured_budget(budget, budget_tokenizer):
    from src.adopd_prompt import prompt_pair_ids, student_messages
    messages = [{'role': 'system', 'content': 'Solve the problem.'},
                {'role': 'user', 'content': r'What is 1+1? Put the answer in \boxed{}.'}]
    original = deepcopy(messages)
    tagged = student_messages(messages, budget)
    assert tagged[0] == messages[0]
    assert tagged[1]['content'] == (f'<budget:{budget}>\n'
        f'Use at most {budget} tokens for reasoning. Finish early when possible.\n\n'
        + messages[1]['content'])
    assert messages == original
    teacher, student = prompt_pair_ids(budget_tokenizer, messages, budget)
    assert teacher == budget_tokenizer.apply_chat_template(messages)
    assert student == budget_tokenizer.apply_chat_template(tagged)
    assert teacher != student


def test_budget_prompt_length_filter_matches_teacher_cache_scan(tmp_path, budget_tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset
    from src.adopd_cache import training_prompts
    from src.adopd_dataset import AdopdDataset
    from src.adopd_prompt import prompt_pair_ids
    short = [{'role': 'user', 'content': 'Q'}]
    long = [{'role': 'user', 'content': 'x' * 80}]
    teacher, student = prompt_pair_ids(budget_tokenizer, short, 8192)
    limit = len(student)
    assert len(budget_tokenizer.apply_chat_template(long)) <= limit
    rows = [{'prompt': messages} for messages in (short, long, short)]
    path = tmp_path / 'prompts.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = object.__new__(AdopdDataset)
    dataset.processor, dataset.tokenizer = None, budget_tokenizer
    dataset.filter_overlong_prompts, dataset.prompt_key = True, 'prompt'
    dataset.config = dict(adopd_reasoning_budget=8192)
    dataset.max_prompt_length, dataset.num_workers = limit, 1
    filtered = dataset.maybe_filter_out_long_prompts(Dataset.from_list(rows))
    assert len(filtered) == 2 and all(row['prompt'] == short for row in filtered)
    cfg = dict(train_files=str(path), max_prompt_length=limit, adopd_reasoning_budget=8192)
    assert list(training_prompts(cfg, budget_tokenizer)) == [teacher]


@pytest.mark.parametrize('eval_only', [False, True])
@pytest.mark.parametrize('budget', [512, 8192])
def test_student_agent_budget_prompt_and_original_teacher_metadata(monkeypatch, budget_tokenizer, budget, eval_only):
    import src.reward_async
    from src.adopd_rollout import agent_class
    from src.adopd_prompt import prompt_pair_ids
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_reasoning_budget=budget,
        enabled=not eval_only, eval_only=eval_only)))
    messages = [{'role': 'user', 'content': 'What is 1+1?'}]
    original = deepcopy(messages)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(token_ids=[4, 99] if len(calls) == 1 else [7])

    async def apply_template(messages):
        return budget_tokenizer.apply_chat_template(messages)

    async def score(*args, **kwargs):
        return {'acc': 1.0}

    monkeypatch.setattr(src.reward_async, 'compute_score', score)
    agent = object.__new__(agent_class())
    agent.tokenizer, agent.response_length = budget_tokenizer, 16384
    agent.apply_chat_template = apply_template
    agent.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(agent.run({'temperature': 1.0}, raw_prompt=messages,
        data_source='math', reward_model={'ground_truth': '2'}))
    teacher, student = prompt_pair_ids(budget_tokenizer, messages, budget)
    assert output.prompt_ids == student and calls[0]['prompt_ids'] == student
    assert calls[0]['sampling_params']['max_tokens'] == budget
    assert calls[1]['prompt_ids'] == student + [4, 99]
    assert output.extra_fields['adopd_teacher_prompt_ids'] == teacher
    assert messages == original


@pytest.mark.parametrize('budgeted,plain', [
    ([4, 99, 7], [5, 6, 99, 7]),  # both rollouts close their traces and answer
    ([4, 4, 4, 4], [5, 99, 8]),   # the budgeted rollout never closes: no answer, and no forced halt
])
def test_eval_agent_scores_unforced_rollouts_with_and_without_budget_prompt(monkeypatch, budget_tokenizer,
                                                                             budgeted, plain):
    import src.reward_async
    from src.adopd_prompt import prompt_pair_ids
    from src.adopd_rollout import eval_agent_class
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_reasoning_budget=2, enabled=True,
                                                       eval_max_completion_length=6)))
    messages = [{'role': 'user', 'content': 'q'}]
    teacher, student = prompt_pair_ids(budget_tokenizer, messages, 2)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(token_ids=budgeted if kwargs['prompt_ids'] == student else plain)

    async def apply_template(messages):
        return budget_tokenizer.apply_chat_template(messages)

    async def score(source, text, truth):
        return {'acc': float(text == truth)}

    monkeypatch.setattr(src.reward_async, 'compute_score', score)
    agent = object.__new__(eval_agent_class())
    agent.tokenizer, agent.response_length = budget_tokenizer, 6
    agent.apply_chat_template = apply_template
    agent.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(agent.run({'temperature': 1.0}, raw_prompt=messages, adopd_eval_plain=True,
        data_source='math', reward_model={'ground_truth': '7'}))
    info = output.extra_fields['reward_extra_info']
    assert output.prompt_ids == student and output.response_ids == budgeted and all(output.response_mask)
    assert sorted(call['prompt_ids'] for call in calls) == sorted([student, teacher])
    for call in calls:  # free generation: no halt stop, no blocked EOS, full response budget
        assert call['sampling_params'] == {'temperature': 1.0, 'max_tokens': 6}
    for prefix, tokens in (('', budgeted), ('nobudget_', plain)):
        closed = 99 in tokens
        stop = tokens.index(99) if closed else len(tokens)
        assert info[prefix + 'acc'] == float(closed and tokens[stop + 1:] == [7])
        assert info[prefix + 'finished'] == float(closed)
        assert info[prefix + 'reasoning_length'] == stop
        assert info[prefix + 'over_budget'] == float(stop > 2)
    # verl records the evaluated response's length itself; a second copy breaks its validation lists.
    assert 'response_length' not in info and info['nobudget_response_length'] == len(plain)


def test_budget_only_eval_skips_the_plain_prompt_rollout(monkeypatch, budget_tokenizer):
    import src.reward_async
    from src.adopd_prompt import prompt_pair_ids
    from src.adopd_rollout import eval_agent_class
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_reasoning_budget=2, enabled=True,
                                                       eval_max_completion_length=6)))
    messages = [{'role': 'user', 'content': 'q'}]
    _, student = prompt_pair_ids(budget_tokenizer, messages, 2)
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs['prompt_ids'])
        return SimpleNamespace(token_ids=[4, 99, 7])

    async def apply_template(messages):
        return budget_tokenizer.apply_chat_template(messages)

    async def score(source, text, truth):
        return {'acc': float(text == truth)}

    monkeypatch.setattr(src.reward_async, 'compute_score', score)
    agent = object.__new__(eval_agent_class())
    agent.tokenizer, agent.response_length = budget_tokenizer, 6
    agent.apply_chat_template = apply_template
    agent.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(agent.run({'temperature': 1.0}, raw_prompt=messages, adopd_eval_plain=False,
        data_source='math', reward_model={'ground_truth': '7'}))
    info = output.extra_fields['reward_extra_info']
    assert calls == [student] and output.response_ids == [4, 99, 7]
    assert info['acc'] == 1. and info['reasoning_length'] == 1
    assert not any(key.startswith('nobudget_') for key in info)


def test_only_validation_rows_use_the_eval_agent(monkeypatch):
    from src.eval_data import with_eval_prompt_limit
    from src.adopd_dataset import AdopdDataset
    from verl.utils.dataset.rl_dataset import RLHFDataset
    seen = {}
    def create(paths, data_config, tokenizer, processor, is_train=True, max_samples=-1):
        dataset = object.__new__(AdopdDataset)
        dataset.config = data_config
        seen[is_train] = dataset
    from omegaconf import OmegaConf
    config = OmegaConf.create({'val_agent_name': 'adopd_eval_agent'})  # verl's struct-free data config
    with_eval_prompt_limit(create)([], config, None, None, is_train=True)
    with_eval_prompt_limit(create)([], config, None, None, is_train=False)
    monkeypatch.setattr(RLHFDataset, '__getitem__', lambda self, item: {'index': item})
    assert 'agent_name' not in seen[True][0]
    assert seen[False][0]['agent_name'] == 'adopd_eval_agent'
    assert config == {'val_agent_name': 'adopd_eval_agent'}


def test_eval_reports_budget_and_no_budget_metrics_per_benchmark(tmp_path):
    class Parent:
        def _val_metrics_update(self, *args): return {'eval/aime25/avg@2': .5}
    class Trainer(AdopdTrainerMixin, Parent): pass
    trainer = Trainer()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=2))))
    trainer.global_steps = 3
    fields = dict(reasoning_length=[2, 8, 4, 4], answer_length=[1, 2, 3, 3],
                  finished=[1., 0., 1., 1.], over_budget=[0., 1., 0., 0.])
    extras = {**fields, **{f'nobudget_{k}': v for k, v in fields.items()},
              'nobudget_acc': [1., 0., 0., 0.], 'nobudget_response_length': [10, 20, 5, 5],
              'full_response_length': [3, 10, 7, 7]}
    metrics = trainer._val_metrics_update(['aime25'] * 2 + ['amc23'] * 2, ['a', 'a', 'b', 'b'], extras, [])
    assert metrics['eval/aime25/avg@2'] == .5
    assert metrics['eval/aime25/reasoning_length'] == 5 and metrics['eval/aime25/over_budget'] == .5
    assert metrics['eval_nobudget/aime25/avg@2'] == .5 and metrics['eval_nobudget/aime25/pass@2'] == 1.
    assert metrics['eval_nobudget/amc23/avg@2'] == 0.
    assert metrics['eval_nobudget/aime25/response-length'] == 15
    assert metrics['eval_nobudget/aime25/finished'] == .5
    assert metrics['eval_nobudget/amc23/answer_length'] == 3
    assert json.loads((tmp_path / 'eval_metrics/3.json').read_text()) == metrics


def test_budget_only_eval_reports_only_budget_metrics(tmp_path):
    class Parent:
        def _val_metrics_update(self, *args): return {'eval/aime25/avg@2': .5}
    class Trainer(AdopdTrainerMixin, Parent): pass
    trainer = Trainer()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=2))))
    trainer.global_steps = 0
    extras = dict(reasoning_length=[2, 8], answer_length=[1, 2], finished=[1., 0.], over_budget=[0., 1.],
                  full_response_length=[3, 10])
    metrics = trainer._val_metrics_update(['aime25'] * 2, ['a', 'a'], extras, [])
    assert metrics == {'eval/aime25/avg@2': .5, 'eval/aime25/reasoning_length': 5, 'eval/aime25/answer_length': 1.5,
                       'eval/aime25/finished': .5, 'eval/aime25/over_budget': .5}


@pytest.mark.parametrize('trace,expected,mask', [
    ([4, 5], [4, 5, 99, 7, 8, 9], [1, 1, 0, 1, 1, 1]),
    ([99], [99, 7, 8, 9], [1, 1, 1, 1]),
    ([4, 99], [4, 99, 7, 8, 9], [1, 1, 1, 1, 1]),
])
def test_two_phase_rollout_budget_and_no_sampling_param_mutation(trace, expected, mask):
    calls = []
    async def generate(*, request_id, prompt_ids, sampling_params):
        calls.append((request_id, prompt_ids, dict(sampling_params)))
        return SimpleNamespace(token_ids=trace if len(calls) == 1 else [7, 8, 9])
    params = {'temperature': 1., 'top_p': 1.}
    actual, chosen = asyncio.run(budget_rollout(generate, [1, 2], params, 99, 2, 6, ['EOS']))
    assert actual == expected and chosen == mask
    assert calls[0][0] == calls[1][0]
    assert calls[0][2]['max_tokens'] == 2 and calls[0][2]['bad_words'] == ['EOS']
    assert calls[1][2]['max_tokens'] == 6 - len(calls[1][1]) + 2
    assert 'bad_words' not in calls[1][2]
    assert params == {'temperature': 1., 'top_p': 1.}


def test_cache_reuse_mismatch_corruption_and_budget(tmp_path):
    identity = dict(model='fixed', teacher_budget=4, seed=42)
    cache = TraceCache(tmp_path, identity)
    cache.write([1, 2], [5, 99], 99, False)
    assert cache.read([1, 2], 99)['trace_ids'] == [5, 99]
    with pytest.raises(FileNotFoundError):
        TraceCache(tmp_path, {**identity, 'seed': 43}).read([1, 2], 99)
    with pytest.raises(FileNotFoundError):
        cache.read([1, 3], 99)
    entry = json.loads(cache.path([1, 2]).read_text()); entry['trace_ids'] = [1, 2, 3, 4, 99]
    cache.path([1, 2]).write_text(json.dumps(entry))
    with pytest.raises(ValueError, match='incompatible'):
        cache.read([1, 2], 99)


@pytest.mark.parametrize('mode', ['sample', 'full_vocab'])
def test_teacher_scores_student_answer_on_cached_teacher_trace_only(tmp_path, monkeypatch, mode):
    from verl.protocol import DataProto
    from src.adopd_runtime import teacher_batch
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_reasoning_budget=4, opd_loss_token_mode=mode)))
    feature_root = tmp_path / 'features'
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(dict(feature_cache=str(feature_root))))
    cache = TraceCache(tmp_path, dict(teacher_budget=8))
    cache.write([1, 2], [40, 41, 99], 99, False)
    cache.write([3], [42, 99], 99, False)
    prompts = torch.tensor([[81, 82], [0, 83]])  # student IDs differ after budget prompting
    responses = torch.tensor([[10, 99, 20, 21, 0], [11, 12, 99, 22, 23]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]])
    policy = attention[:, 2:].clone(); policy[1, 2] = 0
    batch = DataProto(batch=TensorDict(dict(prompts=prompts, responses=responses,
        input_ids=torch.cat([prompts, responses], -1), attention_mask=attention,
        response_mask=policy), batch_size=2),
        non_tensor_batch={'adopd_teacher_prompt_ids': np.array([[1, 2], [3]], dtype=object)})
    calls = []
    async def generate(*, request_id, prompt_ids, sampling_params):
        calls.append(prompt_ids)
        expected = dict(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        if mode == 'full_vocab':
            from src.opd_vllm_capture import atomic_save, sequence_key
            length = 2
            expected['_opd_response_length'] = length
            key = sequence_key(prompt_ids, length)
            atomic_save(dict(hidden_states=torch.zeros(length, 3),
                input_ids=torch.tensor(prompt_ids), response_tokens=torch.tensor(prompt_ids[-length:]),
                prompt_length=len(prompt_ids) - length), feature_root / key[:2] / (key + '.pt'))
        assert sampling_params == expected
        return SimpleNamespace(extra_fields=dict(prompt_ids=[[t] for t in prompt_ids[1:]] + [[0]],
            prompt_logprobs=[[float(i)] for i in range(len(prompt_ids))]))
    manager = SimpleNamespace(_adopd_cache=cache, _adopd_halt_id=99, generate=generate)
    output = asyncio.run(teacher_batch(manager, batch))
    assert calls == [[1, 2, 40, 41, 99, 20, 21], [3, 42, 99, 22, 23]]
    torch.testing.assert_close(output.batch['adopd_teacher_log_probs'],
                              torch.tensor([[0., 0., 4., 5., 0.], [0., 0., 0., 2., 3.]]))
    assert not output.batch['adopd_trace_mask'][1, 2]
    if mode == 'sample':
        assert 'opd_teacher_features' not in output.non_tensor_batch
    else:
        references = output.non_tensor_batch['opd_teacher_features']
        assert [reference['response_offset'] for reference in references] == [2, 3]
        assert [reference['response_length'] for reference in references] == [4, 5]
        assert [reference['teacher_prompt_length'] for reference in references] == [5, 3]
    del batch.non_tensor_batch['adopd_teacher_prompt_ids']
    with pytest.raises(ValueError, match='original teacher prompt IDs'):
        asyncio.run(teacher_batch(manager, batch))


def training_batch():
    from verl.protocol import DataProto
    return DataProto(batch=TensorDict(dict(
        old_log_probs=torch.tensor([[-1., -2., -3., 0.], [-1., -2., -3., -4.]]),
        adopd_teacher_log_probs=torch.tensor([[0., 0., -4., 0.], [0., 0., -2., -3.]]),
        adopd_answer_mask=torch.tensor([[0, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.bool),
        adopd_trace_mask=torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
        adopd_trace_length=torch.tensor([1, 1]), adopd_forced_halt=torch.tensor([True, False])),
        batch_size=2), non_tensor_batch={'uid': np.array(['p', 'p'])})


def full_vocab_batch(tmp_path, monkeypatch, chunk_size=2, group=False):
    from src.opd_options import options as feature_options
    from src.opd_vllm_capture import atomic_save, sequence_key
    batch = training_batch()
    batch.batch['prompts'] = torch.tensor([[1, 2], [0, 3]])
    batch.batch['responses'] = torch.tensor([[4, 99, 5, 0], [4, 99, 6, 7]])
    batch.batch['input_ids'] = torch.cat([batch.batch['prompts'], batch.batch['responses']], -1)
    batch.batch['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]])
    batch.batch['position_ids'] = batch.batch['attention_mask'].cumsum(-1) - 1
    batch.batch['response_mask'] = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]])
    batch.batch['adopd_normalizer'] = torch.full((2,), 2.)
    if group:
        batch.batch['adopd_advantage'] = torch.tensor([.5, -.5])
    head = torch.tensor([[.1, .4], [-.3, .8], [.7, -.2], [-.5, -.6]])
    hidden = torch.tensor([[.3, -.1], [.6, .2], [-.4, .5]])
    atomic_save(dict(weight=head, bias=None), tmp_path / 'output_head.pt')
    references = []
    for i, (student_ids, teacher_ids, p, r, a, rows) in enumerate([
        ([1, 2, 4, 99, 5], [8, 40, 41, 99, 5], 2, 3, 1, hidden[:1]),
        ([3, 4, 99, 6, 7], [9, 42, 99, 6, 7], 1, 4, 2, hidden[1:]),
    ]):
        key = sequence_key(teacher_ids, a)
        path = tmp_path / key[:2] / (key + '.pt')
        atomic_save(dict(hidden_states=rows, response_tokens=torch.tensor(teacher_ids[-a:]),
            input_ids=torch.tensor(teacher_ids), prompt_length=len(teacher_ids) - a), path)
        references.append(dict(path=str(path), head_path=str(tmp_path / 'output_head.pt'),
            prompt_length=p, response_length=r, response_offset=2,
            student_sequence_key=sequence_key(student_ids, r), teacher_sequence_key=key,
            teacher_prompt_length=len(teacher_ids) - a))
    batch.non_tensor_batch['opd_teacher_features'] = np.array(references, dtype=object)
    opts = feature_options(dict(opd_loss_token_mode='full_vocab', opd_token_top_k=2,
                               opd_distribution_chunk_size=chunk_size))
    opts['feature_cache'] = str(tmp_path)
    monkeypatch.setenv('OPD_EXTENSIONS_CONFIG', json.dumps(opts))
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(opd_loss_token_mode='full_vocab',
        adopd_use_group_rollouts=group, group_size=2, adopd_reasoning_budget=8, adopd_length_penalty=0.)))
    logits = torch.linspace(-1., 1.5, 40).reshape(1, 10, 4).clone().requires_grad_()
    return batch, logits, hidden @ head.T


@pytest.mark.parametrize('group', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 3])
def test_full_vocab_answer_values_gradients_and_microbatch_normalization(tmp_path, monkeypatch, group, chunk_size):
    from src.adopd_runtime import answer_loss
    from verl.workers.utils.padding import left_right_2_no_padding
    from verl.utils import tensordict_utils as tu
    batch, logits, teacher_logits = full_vocab_batch(tmp_path, monkeypatch, chunk_size, group)
    packed = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(packed, dp_size=1)
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(loss_max_clamp=1e-5, log_prob_min_clamp=-.01))
    positions = [3, 7, 8]
    q = logits[0, positions].float().log_softmax(-1)
    p = teacher_logits.float().log_softmax(-1)
    exact = (q.exp() * (q - p)).sum(-1)
    expected = (exact[0] + exact[1:].mean()) / 2
    expected_gradient, = torch.autograd.grad(expected, logits, retain_graph=True)
    chosen = torch.zeros(10, requires_grad=True)

    def loss(data, current_logits, current_log_probs):
        outputs = answer_loss(None, distill, student_logits=current_logits, data=data)
        model_output = dict(log_probs=torch.nested.nested_tensor_from_jagged(current_log_probs,
                               data['input_ids'].offsets()),
                            opd_kl=torch.nested.nested_tensor_from_jagged(outputs['opd_kl'][0],
                               data['input_ids'].offsets()))
        return answer_loss(None, distill, model_output=model_output, data=data)[0]

    actual = loss(packed, logits, chosen)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad, expected_gradient)
    if group:
        torch.testing.assert_close(chosen.grad[[1, 5, 6]], torch.tensor([-.25, .125, .125]))
    else:
        assert chosen.grad is None
    assert logits.grad[0, [0, 1, 2, 4, 5, 6, 9]].count_nonzero() == 0
    logits.grad, chosen.grad = None, None
    parts = []
    for i in range(2):
        micro = left_right_2_no_padding(batch[i:i + 1].to_tensordict())
        tu.assign_non_tensor(micro, dp_size=2)
        parts.append(loss(micro, logits[:, 5 * i:5 * i + 5], chosen[5 * i:5 * i + 5]))
    (sum(parts) / 2).backward()
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_full_vocab_old_policy_forward_supplies_group_cost(tmp_path, monkeypatch):
    from src.adopd_runtime import answer_loss
    from verl.utils import tensordict_utils as tu
    batch, logits, teacher_logits = full_vocab_batch(tmp_path, monkeypatch, group=True)
    del batch.batch['adopd_advantage']
    del batch.batch['adopd_normalizer']
    del batch.batch['old_log_probs']
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(loss_max_clamp=1e-5))

    def infer(packed):
        for flag in ('compute_loss', 'distillation_use_topk', 'adopd_scoring'):
            assert tu.get_non_tensor_data(packed, flag, False)
        with torch.no_grad():
            outputs = answer_loss(None, distill, student_logits=logits.detach(), data=packed)
            model = {key: torch.nested.nested_tensor_from_jagged(value[0], packed['input_ids'].offsets())
                     for key, value in outputs.items()}
            model['log_probs'] = torch.nested.nested_tensor_from_jagged(torch.zeros(10), packed['input_ids'].offsets())
            model['entropy'] = torch.nested.nested_tensor_from_jagged(torch.ones(10), packed['input_ids'].offsets())
            zero, metrics = answer_loss(None, distill, model_output=model, data=packed)
            assert zero == 0 and metrics == {}
            return tu.get_tensordict(model, non_tensor_dict={'metrics': {'mfu': .25}})

    trainer = object.__new__(AdopdTrainerMixin)
    trainer.actor_rollout_wg = SimpleNamespace(compute_log_prob=infer)
    old, mfu = trainer._compute_old_log_prob(batch)
    assert mfu == .25 and old.batch['adopd_old_answer_kl'].shape == (2, 4)
    batch = batch.union(old)
    # Corrupt sampled-token scores: the full-vocab branch must not read them.
    batch.batch['adopd_teacher_log_probs'].fill_(float('nan'))
    result = trainer._compute_advantage(batch)
    q = logits[0, [3, 7, 8]].detach().log_softmax(-1)
    p = teacher_logits.log_softmax(-1)
    kl = (q.exp() * (q - p)).sum(-1)
    costs = torch.stack([kl[0], kl[1:].mean()])
    expected = -(costs - costs.mean()) / (costs.std() + 1e-6)
    torch.testing.assert_close(result.batch['adopd_advantage'], expected)


def test_loss_choice_preserves_teacher_trace_cache_identity():
    cfg = recipe()
    sample = cache_identity({**cfg, 'opd_loss_token_mode': 'sample'})
    full = cache_identity({**cfg, 'opd_loss_token_mode': 'full_vocab'})
    assert sample == full
    assert run_baseline.run_directory_name({**cfg, 'opd_loss_token_mode': 'sample'}, 'time') != run_baseline.run_directory_name({**cfg, 'opd_loss_token_mode': 'full_vocab'}, 'time')


@pytest.mark.parametrize('field,value', [('response_offset', 1), ('student_sequence_key', 'wrong'),
                                       ('teacher_sequence_key', 'wrong')])
def test_full_vocab_rejects_misaligned_answer_features(tmp_path, monkeypatch, field, value):
    from src.adopd_runtime import answer_loss
    from verl.workers.utils.padding import left_right_2_no_padding
    batch, logits, _ = full_vocab_batch(tmp_path, monkeypatch)
    batch.non_tensor_batch['opd_teacher_features'][0][field] = value
    packed = left_right_2_no_padding(batch.to_tensordict())
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(loss_max_clamp=None))
    with pytest.raises(ValueError, match='do not match|misaligned'):
        answer_loss(None, distill, student_logits=logits, data=packed)


def test_full_vocab_empty_answers_keep_zero_gradient_without_features(tmp_path, monkeypatch):
    from src.adopd_runtime import answer_loss
    from verl.workers.utils.padding import left_right_2_no_padding
    from verl.utils import tensordict_utils as tu
    batch, logits, _ = full_vocab_batch(tmp_path, monkeypatch)
    batch.non_tensor_batch['opd_teacher_features'][:] = None
    batch.batch['adopd_answer_mask'].zero_()
    batch.batch['adopd_normalizer'].zero_()
    packed = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(packed, dp_size=1)
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(loss_max_clamp=None, log_prob_min_clamp=None))
    outputs = answer_loss(None, distill, student_logits=logits, data=packed)
    assert outputs['opd_kl'].count_nonzero() == 0
    model = dict(log_probs=torch.nested.nested_tensor_from_jagged(torch.zeros(10), packed['input_ids'].offsets()),
        opd_kl=torch.nested.nested_tensor_from_jagged(outputs['opd_kl'][0], packed['input_ids'].offsets()))
    loss, _ = answer_loss(None, distill, model_output=model, data=packed)
    assert loss == 0
    loss.backward()
    assert logits.grad.count_nonzero() == 0


@pytest.mark.parametrize('h100', [False, True])
def test_full_vocab_setup_keeps_adopd_answer_and_teacher_hooks(monkeypatch, h100):
    from src import adopd_runtime, opd_runtime, opd_h100_runtime
    from verl.workers import engine_workers
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(enabled=True, opd_loss_token_mode='full_vocab',
                                                     h100_runtime=h100)))
    calls = []

    def features():
        calls.append('features')
        monkeypatch.setattr(engine_workers, 'distillation_ppo_loss', opd_runtime.vocabulary_loss)
        monkeypatch.setattr(AsyncTeacherLLMServerManager, 'compute_teacher_logprobs_batch', opd_runtime.teacher_batch)

    monkeypatch.setattr(opd_runtime, 'install', features)
    monkeypatch.setattr(opd_h100_runtime, 'install', lambda: calls.append('h100'))
    adopd_runtime.install()
    assert calls == (['features', 'h100'] if h100 else ['features'])
    assert engine_workers.distillation_ppo_loss is adopd_runtime.answer_loss
    assert AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch is adopd_runtime.teacher_batch


def test_no_group_bypasses_eq5_and_all_invalid_skips_parent_update(monkeypatch):
    import src.adopd_trainer as module
    class Parent:
        def _update_actor(self, batch):
            raise AssertionError('optimizer must not run for an empty loss')
    class Trainer(AdopdTrainerMixin, Parent):
        pass
    monkeypatch.setattr(module, 'runtime_options', lambda: dict(adopd_use_group_rollouts=False))
    monkeypatch.setattr(module, 'group_advantages', lambda *a, **k: pytest.fail('Eq. (5) called without group'))
    batch = training_batch()
    trainer = Trainer()
    trainer._compute_advantage(batch)
    assert 'adopd_advantage' not in batch.batch
    assert batch.batch['adopd_normalizer'].tolist() == [2., 2.]
    batch.batch['adopd_answer_mask'].zero_()
    result = trainer._update_actor(batch)
    assert result.meta_info['metrics']['actor/adopd/update_skipped'] == 1


@pytest.mark.parametrize('penalty', [0., .1])
def test_sampled_loss_native_padding_and_microbatch_normalization(monkeypatch, penalty):
    from verl.protocol import DataProto
    from verl.workers.utils.padding import left_right_2_no_padding
    from src.adopd_runtime import sampled_answer_loss
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_use_group_rollouts=False,
        adopd_reasoning_budget=2, adopd_length_penalty=penalty)))
    batch = training_batch()
    batch.batch['prompts'] = torch.tensor([[1, 2], [0, 3]])
    batch.batch['responses'] = torch.tensor([[4, 99, 5, 0], [4, 99, 6, 7]])
    batch.batch['input_ids'] = torch.cat([batch.batch['prompts'], batch.batch['responses']], -1)
    batch.batch['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1]])
    batch.batch['response_mask'] = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]])
    batch.batch['position_ids'] = batch.batch['attention_mask'].cumsum(-1) - 1
    batch.batch['adopd_normalizer'] = torch.tensor([2., 2.])
    # Aligned next-token values: one padded sample has a two-token prompt; the other one token.
    logits = torch.tensor([0., -1., -2., -3., 0., -1., -2., -3., -4., 0.], requires_grad=True)
    expected_per_seq = torch.tensor([1., -1.])
    packed = left_right_2_no_padding(batch.to_tensordict())
    from verl.utils import tensordict_utils as tu
    tu.assign_non_tensor(packed, dp_size=1)
    log_probs = torch.nested.nested_tensor_from_jagged(logits, packed['input_ids'].offsets())
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(
        log_prob_min_clamp=-10., loss_max_clamp=10.))
    loss, metrics = sampled_answer_loss(None, distill, {'log_probs': log_probs}, packed)
    torch.testing.assert_close(loss, expected_per_seq.mean() + penalty / 2)
    assert metrics['adopd/length_cost'].aggregate() == pytest.approx(penalty / 2)
    assert metrics['adopd/regularized_distortion'].aggregate() == pytest.approx(penalty / 2)
    loss.backward()
    # Mean per ANSWER, then mean per TRAJECTORY, not a global token mean.
    assert logits.grad[3] == .5 and logits.grad[7] == -.25 and logits.grad[8] == -.25
    torch.testing.assert_close(logits.grad[[1, 5, 6]], torch.full((3,), penalty / 4))
    assert logits.grad[[0, 2, 4, 9]].count_nonzero() == 0
    expected_gradient = logits.grad.clone()
    logits.grad = None
    partial_losses = []
    for i in range(2):
        micro = left_right_2_no_padding(batch[i:i+1].to_tensordict())
        tu.assign_non_tensor(micro, dp_size=2)
        values = logits[5*i:5*i+5]
        output = torch.nested.nested_tensor_from_jagged(values, micro['input_ids'].offsets())
        partial, _ = sampled_answer_loss(None, distill, {'log_probs': output}, micro)
        partial_losses.append(partial)
    # Simulate FSDP's gradient averaging across two ranks, each with one microbatch.
    (sum(partial_losses) / 2).backward()
    torch.testing.assert_close(logits.grad, expected_gradient)


def test_adopd_cli_overrides_are_consumed_by_launcher(tmp_path):
    cfg = recipe()
    extra = ['adopd_use_group_rollouts=true', 'adopd_length_penalty=0.2', 'opd_loss_token_mode=full_vocab']
    run_baseline.apply_opd_overrides(cfg, extra)
    native = compose(cfg, tmp_path, extra)
    env = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.ADOPD_CONFIG)
    assert env['adopd_use_group_rollouts'] and env['adopd_length_penalty'] == .2
    assert env['opd_loss_token_mode'] == 'full_vocab'
    assert json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.OPD_EXTENSIONS_CONFIG)['opd_loss_token_mode'] == 'full_vocab'
    assert 'adopd_use_group_rollouts' not in native


@pytest.mark.parametrize('eval_only', [False, True])
@pytest.mark.parametrize('mode', ['sample', 'full_vocab'])
def test_main_passes_same_config_to_driver_and_ray(tmp_path, monkeypatch, eval_only, mode):
    import src.adopd_cache as cache_module
    cfg = recipe()
    cfg['opd_loss_token_mode'] = mode
    cfg['output_dir'] = str(tmp_path)
    cfg['use_wandb'] = False
    monkeypatch.setattr(run_baseline, 'load_config', lambda _: deepcopy(cfg))
    monkeypatch.setattr(run_baseline, 'load_runtime_env', lambda: {})
    monkeypatch.setattr(run_baseline, 'preflight', lambda _: None)
    monkeypatch.setattr(run_baseline, 'prepare_model', lambda _: None)
    monkeypatch.setattr(run_baseline, 'configure_wandb_paths', lambda *a, **k: None)
    monkeypatch.setattr(cache_module, 'validate_training_cache', lambda _: 100)
    captured = {}
    def run(command, run_dir, env, keep):
        captured.update(command=command, env=env)
        return 0
    monkeypatch.setattr(run_baseline, 'run_training', run)
    args = [str(ROOT / 'configs/qwen3_4b_adopd_v2.yaml')]
    if eval_only:
        args += ['--eval-only']
    assert run_baseline.main(args) == 0
    runtime = json.loads(captured['env']['ADOPD_CONFIG'])
    assert runtime['enabled'] is (not eval_only)
    assert runtime['eval_only'] is eval_only
    assert runtime['adopd_reasoning_budget'] == 8192
    features = json.loads(captured['env']['OPD_EXTENSIONS_CONFIG'])
    assert features == {} if eval_only or mode == 'sample' else features['opd_loss_token_mode'] == mode


@pytest.mark.parametrize('scenario', ['train', 'eval', 'cache_only', 'dry_run', 'cache_failure'])
def test_auto_cache_workflow_order_failure_and_no_gpu_modes(tmp_path, monkeypatch, scenario):
    import yaml
    import src.adopd_cache as cache_module
    cfg = recipe()
    cfg.update(output_dir=str(tmp_path), use_wandb=False)
    monkeypatch.setattr(run_baseline, 'load_config', lambda _: deepcopy(cfg))
    monkeypatch.setattr(run_baseline, 'load_runtime_env', lambda: {})
    monkeypatch.setattr(run_baseline, 'preflight', lambda _: None)
    monkeypatch.setattr(run_baseline, 'prepare_model', lambda _: None)
    monkeypatch.setattr(run_baseline, 'configure_wandb_paths', lambda *a, **k: None)
    events, child_configs = [], []

    def child(command, *, cwd, env, check):
        events.append('cache')
        assert command[0] == sys.executable and command[-1] == '--prepare-teacher-cache'
        assert check is True and cwd == ROOT
        path = Path(command[-2])
        child_configs.append(path)
        prepared = yaml.safe_load(path.read_text())
        # Serialize the effective settings so CLI overrides also reach the cache child.
        assert prepared['adopd_use_group_rollouts'] is True
        assert prepared['adopd_length_penalty'] == .2
        assert prepared['teacher_model_path'] == cfg['teacher_model_path']
        if scenario == 'cache_failure':
            raise subprocess.CalledProcessError(7, command)
        return subprocess.CompletedProcess(command, 0)

    def validate(_):
        events.append('validate')
        return 100

    def train(*args):
        events.append('train')
        return 0

    def cache_only(config, dry_run=False):
        events.append('cache_only')
        assert config['adopd_use_group_rollouts'] is True

    monkeypatch.setattr(run_baseline.subprocess, 'run', child)
    monkeypatch.setattr(run_baseline, 'run_training', train)
    monkeypatch.setattr(cache_module, 'validate_training_cache', validate)
    monkeypatch.setattr(cache_module, 'prepare', cache_only)
    args = [str(ROOT / 'configs/qwen3_4b_adopd_v2.yaml'), '--auto-prepare-teacher-cache']
    flags = {'eval': '--eval-only', 'cache_only': '--prepare-teacher-cache', 'dry_run': '--dry-run'}
    if scenario in flags:
        args.append(flags[scenario])
    args += ['--', 'adopd_use_group_rollouts=true', 'adopd_length_penalty=0.2']
    assert run_baseline.main(args) == (7 if scenario == 'cache_failure' else 0)
    assert events == {'train': ['cache', 'validate', 'train'], 'eval': ['train'],
                      'cache_only': ['cache_only'], 'dry_run': [], 'cache_failure': ['cache']}[scenario]
    assert all(not path.exists() for path in child_configs)
    if scenario in {'cache_only', 'dry_run', 'cache_failure'}:
        assert not list(tmp_path.iterdir())


def test_offline_cache_preparation_resumes_and_closes_within_teacher_budget(tmp_path, monkeypatch):
    from src import adopd_cache
    import transformers
    model = tmp_path / 'model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'frozen')
    cfg = {**options({}), 'teacher_model_path': str(model), 'teacher_tp_size': 1,
           'max_prompt_length': 10, 'adopd_teacher_max_completion_length': 5,
           'adopd_teacher_trace_cache': str(tmp_path / 'cache')}
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [99], eos_token_id=0, bos_token_id=0)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    monkeypatch.setattr(adopd_cache, 'training_prompts', lambda *a: iter([[1, 2], [3]]))
    calls = []
    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def generate(self, prompts, params, **kwargs):
            assert all(p.max_tokens == 5 for p in params)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids)])
                    for ids in [[4, 99], [5, 6, 7, 8, 9]]]
    class SamplingParams:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    adopd_cache.prepare(cfg, dry_run=True)
    assert not calls and not Path(cfg['adopd_teacher_trace_cache']).exists()
    adopd_cache.prepare(cfg)
    cache = TraceCache(cfg['adopd_teacher_trace_cache'], cache_identity(cfg))
    assert cache.read([1, 2], 99)['trace_ids'] == [4, 99]
    assert cache.read([3], 99)['trace_ids'] == [5, 6, 7, 8, 99]
    adopd_cache.prepare(cfg)
    assert len(calls) == 1


def test_a_generated_batch_is_stored_concurrently_on_slow_storage(tmp_path, monkeypatch):
    import time
    from src import adopd_cache
    import transformers
    model = tmp_path / 'model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'frozen')
    cfg = {**options({}), 'teacher_model_path': str(model), 'teacher_tp_size': 1,
           'max_prompt_length': 10, 'adopd_teacher_max_completion_length': 5,
           'adopd_teacher_trace_cache': str(tmp_path / 'cache')}
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [99], eos_token_id=0, bos_token_id=0)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    class LLM:
        def __init__(self, **kwargs): pass
        def generate(self, prompts, params, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 99])]) for _ in prompts]
    class SamplingParams:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    write = TraceCache.write
    def slow_write(self, *args):  # Network filesystems take seconds per small file.
        time.sleep(0.2)
        write(self, *args)
    monkeypatch.setattr(TraceCache, 'write', slow_write)
    prompts = [[i] for i in range(16)]
    worker = adopd_cache.TeacherCacheWorker(cfg)
    cache = TraceCache(cfg['adopd_teacher_trace_cache'], cache_identity(cfg))
    start = time.monotonic()
    adopd_cache.store(cache, prompts, worker.generate(prompts), 99)
    assert time.monotonic() - start < 16 * 0.2 / 4
    assert all(cache.read(p, 99)['trace_ids'] == [4, 99] for p in prompts)


def test_cache_scan_and_validation_read_slow_storage_concurrently(tmp_path, monkeypatch):
    import time
    from src import adopd_cache
    import transformers
    model = tmp_path / 'model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'frozen')
    cfg = {**options({}), 'teacher_model_path': str(model), 'teacher_tp_size': 1,
           'max_prompt_length': 10, 'adopd_teacher_max_completion_length': 5,
           'adopd_teacher_trace_cache': str(tmp_path / 'cache')}
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [99], eos_token_id=0, bos_token_id=0)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    prompts = [[i] for i in range(32)]
    monkeypatch.setattr(adopd_cache, 'training_prompts', lambda *a: iter(prompts))
    cache = TraceCache(cfg['adopd_teacher_trace_cache'], cache_identity(cfg))
    for prompt in prompts[:16]:
        cache.write(prompt, [4, 99], 99, False)
    read = TraceCache.read
    def slow_read(self, *args):  # Network filesystems take a fraction of a second per lookup.
        time.sleep(0.1)
        return read(self, *args)
    monkeypatch.setattr(TraceCache, 'read', slow_read)
    reported = []
    monkeypatch.setattr(adopd_cache, 'prepare_distributed', lambda c, missing, reused: reported.append((missing, reused)))
    start = time.monotonic()
    adopd_cache.prepare({**cfg, 'h100_nnodes': 2, 'n_gpus_per_node': 1})
    assert time.monotonic() - start < 32 * 0.1 / 4
    assert reported == [(prompts[16:], 16)]
    with pytest.raises(FileNotFoundError):
        adopd_cache.validate_training_cache(cfg)
    for prompt in prompts[16:]:
        cache.write(prompt, [4, 99], 99, False)
    start = time.monotonic()
    assert adopd_cache.validate_training_cache(cfg) == 32
    assert time.monotonic() - start < 32 * 0.1 / 4


@pytest.mark.parametrize('nnodes,gpus,tp', [(2, 4, 4), (4, 4, 4), (4, 4, 1)])
def test_cache_layout_uses_every_configured_gpu(nnodes, gpus, tp):
    from collections import Counter
    from src.adopd_cache import cache_worker_layout
    nodes = [{'Alive': True, 'NodeID': str(i), 'Resources': {'GPU': gpus}}
             for i in range(nnodes)]
    nodes.append({'Alive': False, 'NodeID': 'dead', 'Resources': {'GPU': gpus}})
    cfg = dict(h100_nnodes=nnodes, n_gpus_per_node=gpus, teacher_tp_size=tp)
    layout = cache_worker_layout(cfg, nodes)
    assert len(layout) * tp == nnodes * gpus
    assert Counter(layout) == {str(i): gpus // tp for i in range(nnodes)}
    with pytest.raises(RuntimeError, match='matching nodes'):
        cache_worker_layout(cfg, nodes[1:])
    with pytest.raises(ValueError, match='divide'):
        cache_worker_layout({**cfg, 'teacher_tp_size': 3}, nodes)


@pytest.mark.parametrize('nnodes', [2, 4])
@pytest.mark.parametrize('fail', [False, True])
def test_distributed_cache_batches_are_unique_resume_and_release_workers(tmp_path, monkeypatch, nnodes, fail):
    import ray
    import transformers
    from src import adopd_cache
    model = tmp_path / 'model'; model.mkdir()
    (model / 'model.safetensors').write_bytes(b'frozen')
    cfg = {**options({}), 'teacher_model_path': str(model), 'teacher_tp_size': 4,
           'h100_nnodes': nnodes, 'n_gpus_per_node': 4, 'max_prompt_length': 10,
           'adopd_teacher_trace_cache': str(tmp_path / 'cache'), 'adopd_cache_batch_size': 2}
    cache = TraceCache(cfg['adopd_teacher_trace_cache'], cache_identity(cfg))
    cache.write([0], [7, 99], 99, False)
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [99])
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    monkeypatch.setattr(adopd_cache, 'training_prompts', lambda *a: iter([[i] for i in range(21)]))
    created, generated, killed, closed, events = [], [], [], [], []

    class Ref:
        def __init__(self, callback): self.callback = callback

    class Worker:
        def __init__(self, node):
            self.node = node
            self.ready = SimpleNamespace(remote=lambda: Ref(lambda: {'node': node, 'gpus': '0,1,2,3'}))
            self.generate = SimpleNamespace(remote=lambda prompts: Ref(lambda: self.run(prompts)))
            self.close = SimpleNamespace(remote=lambda: Ref(lambda: closed.append(self)))

        def run(self, prompts):
            if fail:
                raise RuntimeError('generation failed')
            generated.extend(tuple(p) for p in prompts)
            return [([7, prompt[0], 99], False) for prompt in prompts]  # the driver's node stores them

    class Factory:
        def options(self, *, scheduling_strategy):
            node = scheduling_strategy.node_id
            def create(config):
                worker = Worker(node)
                created.append(worker)
                return worker
            return SimpleNamespace(remote=create)

    def remote(*, num_cpus, num_gpus):
        assert num_cpus == 1 and num_gpus == 4
        return lambda cls: Factory()

    def get(refs, **kwargs):
        return [ref.callback() for ref in refs] if isinstance(refs, list) else refs.callback()

    def wait(refs, **kwargs):
        # Complete out of submission order, after all nodes have work.
        assert len(created) == nnodes
        return [refs[-1]], refs[:-1]

    def init(**kwargs):
        assert kwargs['runtime_env']['env_vars']['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] == '0'
        events.append('init')

    monkeypatch.setattr(ray, 'init', init)
    monkeypatch.setattr(ray, 'nodes', lambda: [
        {'Alive': True, 'NodeID': f'{i + 1:056x}', 'Resources': {'GPU': 4}} for i in range(nnodes)])
    monkeypatch.setattr(ray, 'remote', remote)
    monkeypatch.setattr(ray, 'get', get)
    monkeypatch.setattr(ray, 'wait', wait)
    monkeypatch.setattr(ray, 'kill', lambda worker, **kwargs: killed.append(worker))
    monkeypatch.setattr(ray, 'shutdown', lambda: events.append('shutdown'))
    if fail:
        with pytest.raises(RuntimeError, match='generation failed'):
            adopd_cache.prepare(cfg)
    else:
        adopd_cache.prepare(cfg)
        assert set(generated) == {(i,) for i in range(1, 21)}
        assert len(generated) == 20  # no duplicates and no re-generation of cached prompt 0
        assert len(closed) == nnodes
        assert all(cache.read([i], 99)['trace_ids'] == [7, i, 99] for i in range(1, 21))
        adopd_cache.prepare(cfg)
        assert len(created) == nnodes  # fully cached runs start no engines
    assert {worker.node for worker in created} == {f'{i + 1:056x}' for i in range(nnodes)}
    assert killed == created and events == ['init', 'shutdown']


@pytest.mark.parametrize('nnodes', [2, 4])
def test_sbatch_reads_yaml_allocation_and_launches_all_nodes(tmp_path, nnodes):
    import shlex
    import yaml
    (tmp_path / 'scripts/sbatch').mkdir(parents=True)
    (tmp_path / 'configs').mkdir()
    cfg_path = tmp_path / 'configs/qwen3_4b_adopd_v2.yaml'
    cfg_path.write_text(yaml.safe_dump(dict(h100_nnodes=nnodes, n_gpus_per_node=4)))
    source = (ROOT / 'scripts/sbatch/qwen3_4b_adopd_v2.sbatch').read_text()
    script = tmp_path / 'scripts/sbatch/qwen3_4b_adopd_v2.sbatch'
    script.write_text(source.replace(f'REPO_ROOT={ROOT}', f'REPO_ROOT={shlex.quote(str(tmp_path))}'))
    binary = tmp_path / 'bin'; binary.mkdir()
    python = binary / 'python'
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    python.chmod(0o755)
    (tmp_path / 'scripts/h100_8gpu_env.sh').write_text(
        f'export OPD_PYTHON={shlex.quote(str(python))}\nexport OPD_RAY=ray\n')
    mock = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open(os.environ['MOCK_LOG'], 'a') as stream:
    stream.write(json.dumps([name] + sys.argv[1:]) + '\\n')
if name == 'scontrol':
    print('\\n'.join('node' + str(i) for i in range(int(os.environ['MOCK_NNODES']))))
if name == 'srun' and '--nodes=1' in sys.argv:
    print('RAY_HEAD_IP=10.0.0.1')
'''
    # -S avoids unrelated site startup hooks; these stubs never contact Slurm.
    for name in ('sbatch', 'scontrol', 'srun'):
        path = binary / name
        path.write_text(mock.replace('#!/usr/bin/env python3', f'#!{sys.executable} -S'))
        path.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = {key: value for key, value in os.environ.items() if not key.startswith('SLURM_')}
    env.update(PATH=str(binary) + os.pathsep + env['PATH'], MOCK_LOG=str(log), MOCK_NNODES=str(nnodes),
               PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(['bash', str(script), '--prepare-teacher-cache'], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1 and calls[0][0] == 'sbatch'
    assert f'--nodes={nnodes}' in calls[0] and f'--ntasks={nnodes}' in calls[0]
    assert '--gpus-per-node=h100:4' in calls[0] and calls[0][-1] == '--prepare-teacher-cache'
    log.unlink()
    env.update(SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES=str(nnodes), SLURM_GPUS_ON_NODE='4',
               SLURM_JOB_NODELIST='mock', SLURM_CPUS_PER_TASK='48')
    result = subprocess.run(['bash', str(script), '--prepare-teacher-cache'], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    launch = calls[-1]
    assert launch[0] == 'srun' and f'--nodes={nnodes}' in launch and f'--ntasks={nnodes}' in launch
    assert launch[launch.index('--min-nodes') + 1] == str(nnodes)
    assert launch[launch.index('--num-gpus') + 1] == '4'
    assert launch[-1] == '--prepare-teacher-cache'
    log.unlink()
    env['SLURM_JOB_NUM_NODES'] = str(nnodes + 1)
    result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'differ from h100_nnodes' in result.stderr
    assert not log.exists()  # allocation mismatches never start Ray or generation


@pytest.mark.parametrize('eval_only,group', [(False, False), (False, True), (True, True)])
def test_trainer_native_init_guards_and_cache_setup(tmp_path, monkeypatch, eval_only, group):
    import src.adopd_trainer as module
    cfg = recipe(); cfg['adopd_use_group_rollouts'] = group
    cfg['_adopd_eval_only'] = eval_only
    native = compose(cfg, tmp_path)
    opts = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.ADOPD_CONFIG)
    monkeypatch.setattr(module, 'runtime_options', lambda: opts)
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 1}, encode=lambda *a, **k: [99])
    server = SimpleNamespace()
    class Parent:
        def init_workers(self):
            self.initialized = True
            self.teacher_model_manager = SimpleNamespace(server_manager=server,
                rollout_replicas=[SimpleNamespace(model_config=SimpleNamespace(tokenizer=tokenizer))])
    class Trainer(AdopdTrainerMixin, Parent): pass
    trainer = Trainer(); trainer.config = native; trainer.tokenizer = tokenizer
    trainer.init_workers()
    assert trainer.initialized
    assert hasattr(server, '_adopd_cache') is (not eval_only)
    native.actor_rollout_ref.actor.ppo_epochs = 2
    with pytest.raises(ValueError, match='one optimizer'):
        trainer.init_workers()


class _GenBatch(SimpleNamespace):
    def __len__(self):
        return len(self.non_tensor_batch['raw_prompt'])


@pytest.mark.parametrize('mode,expected', [(True, [True, True]), (False, [False, False]), ('initial', [True, False])])
def test_validation_rows_carry_the_plain_prompt_decision_for_their_step(monkeypatch, mode, expected):
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps({'adopd_eval_plain_prompt': mode}))
    class Parent:  # mirrors verl: _validate builds its generation batch through _get_gen_batch
        def _get_gen_batch(self, batch):
            return _GenBatch(non_tensor_batch=dict(batch))
        def _validate(self):
            return self._get_gen_batch({'raw_prompt': np.array(['a', 'b'], dtype=object)}).non_tensor_batch
    class Trainer(AdopdTrainerMixin, Parent): pass
    trainer = Trainer()
    for step, plain in zip((0, 20), expected):
        trainer.global_steps = step
        assert list(trainer._validate()['adopd_eval_plain']) == [plain, plain]
    training = trainer._get_gen_batch({'raw_prompt': np.array(['a'], dtype=object)})
    assert 'adopd_eval_plain' not in training.non_tensor_batch


def test_adopd_evaluates_with_a_longer_response_limit_than_training(tmp_path):
    cfg = recipe()
    cfg.update(val_max_completion_length=32768)
    native = compose(cfg, tmp_path, [])
    assert native.actor_rollout_ref.rollout.response_length == 16384
    assert native.actor_rollout_ref.rollout.max_model_len == 4096 + 32768
    opts = json.loads(native.ray_kwargs.ray_init.runtime_env.env_vars.ADOPD_CONFIG)
    assert opts['eval_max_completion_length'] == 32768


def test_eval_agent_scores_the_full_long_rollout_but_returns_a_training_sized_response(monkeypatch, budget_tokenizer):
    import src.reward_async
    from src.adopd_rollout import eval_agent_class
    monkeypatch.setenv('ADOPD_CONFIG', json.dumps(dict(adopd_reasoning_budget=2, enabled=True,
                                                       eval_max_completion_length=8)))
    long_answer = [4, 4, 4, 4, 4, 99, 7]  # reasoning longer than the training response limit
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs['sampling_params']['max_tokens'])
        return SimpleNamespace(token_ids=long_answer)

    async def apply_template(messages):
        return budget_tokenizer.apply_chat_template(messages)

    async def score(source, text, truth):
        return {'acc': float(text == truth)}

    monkeypatch.setattr(src.reward_async, 'compute_score', score)
    agent = object.__new__(eval_agent_class())
    agent.tokenizer, agent.response_length = budget_tokenizer, 4
    agent.apply_chat_template = apply_template
    agent.server_manager = SimpleNamespace(generate=generate)
    output = asyncio.run(agent.run({'temperature': 1.0}, raw_prompt=[{'role': 'user', 'content': 'q'}],
        adopd_eval_plain=True, data_source='math', reward_model={'ground_truth': '7'}))
    info = output.extra_fields['reward_extra_info']
    assert seen == [8, 8]
    assert info['acc'] == 1.0 and info['nobudget_acc'] == 1.0
    assert output.response_ids == long_answer[:4] and output.response_mask == [1] * 4
    assert info['full_response_length'] == 7 and info['nobudget_response_length'] == 7


def test_eval_response_length_metrics_use_the_full_rollouts(tmp_path):
    class Parent:
        def _val_metrics_update(self, data_sources, sample_uids, extras, sample_turns):
            return {'eval/amc23/response-length': sum(extras['response_length']) / len(extras['response_length'])}
    class Trainer(AdopdTrainerMixin, Parent): pass
    trainer = Trainer()
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=1))))
    trainer.global_steps = 0
    extras = dict(response_length=[4, 4], full_response_length=[7, 9], reasoning_length=[5, 6],
                  answer_length=[1, 2], finished=[1., 1.], over_budget=[1., 1.])
    metrics = trainer._val_metrics_update(['amc23'] * 2, ['a', 'b'], extras, [])
    assert metrics['eval/amc23/response-length'] == 8
