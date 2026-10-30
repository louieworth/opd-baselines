"""Configuration for answer-only V2 self-distillation."""

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'adopd_use_group_rollouts': False,
    'adopd_reasoning_budget': 8192,
    'adopd_teacher_max_completion_length': 16384,
    'adopd_length_penalty': 0.1,
    'adopd_teacher_trace_cache': str(ROOT / 'data/.cache/adopd_v2'),
    'adopd_cache_seed': 42,
    'adopd_cache_batch_size': 16,
    'adopd_eval_plain_prompt': True,  # True, False, or 'initial' (only the step-0 evaluation)
}
KEYS = tuple(DEFAULTS)
LOSS_KEYS = ('opd_loss_token_mode', 'opd_distribution_chunk_size')


def options(config):
    result = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    result.update(opd_loss_token_mode=config.get('opd_loss_token_mode', 'sample'),
                  opd_distribution_chunk_size=config.get('opd_distribution_chunk_size', 32))
    if result['opd_loss_token_mode'] not in ('sample', 'full_vocab'):
        raise ValueError('ADOPD opd_loss_token_mode must be sample or full_vocab')
    if type(result['opd_distribution_chunk_size']) is not int or result['opd_distribution_chunk_size'] < 1:
        raise ValueError('opd_distribution_chunk_size must be a positive integer')
    if type(result['adopd_use_group_rollouts']) is not bool:
        raise ValueError('adopd_use_group_rollouts must be a boolean')
    if type(result['adopd_eval_plain_prompt']) is not bool and result['adopd_eval_plain_prompt'] != 'initial':
        raise ValueError("adopd_eval_plain_prompt must be a boolean or 'initial'")
    for key in ('adopd_reasoning_budget', 'adopd_teacher_max_completion_length', 'adopd_cache_batch_size'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['adopd_cache_seed']) is not int or result['adopd_cache_seed'] < 0:
        raise ValueError('adopd_cache_seed must be a nonnegative integer')
    penalty = result['adopd_length_penalty']
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError('adopd_length_penalty must be finite and nonnegative')
    cache = result['adopd_teacher_trace_cache']
    if not isinstance(cache, str) or not Path(cache).is_absolute():
        raise ValueError('adopd_teacher_trace_cache must be an absolute path')
    response_budget = config.get('max_completion_length', 16384)
    if type(response_budget) is not int or not 0 < result['adopd_reasoning_budget'] < response_budget - 1:
        raise ValueError('ADOPD reasoning budget must leave room for </think> and the answer')
    if result['adopd_teacher_max_completion_length'] < 2:
        raise ValueError('Teacher completion budget must leave room for a closing token')
    if config.get('mode', 'thinking') != 'thinking':
        raise ValueError('ADOPD requires mode: thinking')
    if result['adopd_use_group_rollouts']:
        size = config.get('group_size', 8)
        if type(size) is not int or size < 2:
            raise ValueError('Group ADOPD requires group_size >= 2')
    return result


def runtime_options():
    return json.loads(os.environ.get('ADOPD_CONFIG', '{}'))


def enabled():
    return runtime_options().get('enabled', False)


def objective_name(config):
    prefix = 'full_vocab' if config.get('opd_loss_token_mode') == 'full_vocab' else 'sampled'
    trace = 'group_reinforce' if config['adopd_use_group_rollouts'] else 'length_pg'
    return f'{prefix}_answer_kl_plus_{trace}'
