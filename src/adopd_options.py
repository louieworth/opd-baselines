"""Configuration for answer-only V2 self-distillation."""

import json
import math
import os
import re
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
    # None: one unforced evaluation at val_max_completion_length. Otherwise a list of
    # {name, budget, limit, forced}: every question runs every variant (forced: </think> at the budget).
    # budget may be omitted: it is then the run's adopd_reasoning_budget.
    'adopd_eval_variants': None,
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
    variants = result['adopd_eval_variants']
    if variants is not None:
        if not isinstance(variants, list) or not variants:
            raise ValueError('adopd_eval_variants must be a non-empty list')
        names = set()
        # Validated as written: a missing budget is filled in only where it is used (eval_variants), so a
        # later override of adopd_reasoning_budget still reaches it.
        for variant in variants:
            if not isinstance(variant, dict) or not {'name', 'limit', 'forced'} <= set(variant) \
                    or not set(variant) <= {'name', 'budget', 'limit', 'forced'}:
                raise ValueError('Each eval variant needs name, limit and forced, and optionally budget')
            variant = dict(variant, budget=variant.get('budget', result['adopd_reasoning_budget']))
            if not isinstance(variant['name'], str) or not re.fullmatch(r'[a-z0-9_]+', variant['name']) \
                    or variant['name'] in names:
                raise ValueError('Eval variant names must be unique and match [a-z0-9_]+')
            names.add(variant['name'])
            if type(variant['budget']) is not int or type(variant['limit']) is not int \
                    or variant['budget'] < 1 or variant['limit'] < 1 or type(variant['forced']) is not bool:
                raise ValueError('Eval variant budget and limit must be positive integers and forced a boolean')
            if variant['forced'] and variant['budget'] >= variant['limit'] - 1:
                raise ValueError('A forced eval variant must leave room for </think> and the answer')
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


def eval_variants(opts):
    """The explicit eval variants with each missing budget set to the run's reasoning budget, or None."""
    if opts.get('adopd_eval_variants') is None:
        return None
    return [dict(v, budget=v.get('budget', opts['adopd_reasoning_budget'])) for v in opts['adopd_eval_variants']]


def runtime_options():
    return json.loads(os.environ.get('ADOPD_CONFIG', '{}'))


def enabled():
    return runtime_options().get('enabled', False)


def objective_name(config):
    prefix = 'full_vocab' if config.get('opd_loss_token_mode') == 'full_vocab' else 'sampled'
    trace = 'group_reinforce' if config['adopd_use_group_rollouts'] else 'length_pg'
    return f'{prefix}_answer_kl_plus_{trace}'
