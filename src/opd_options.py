"""New OPD support options, separate from the baseline training recipe."""

import json
import os

from src.teacher_prompt import KEYS as TEACHER_PROMPT_KEYS, options as teacher_prompt_options

MODES = ('sample', 'top_k_student', 'top_k_teacher', 'top_k_diff', 'full_vocab')
KEYS = ('opd_loss_token_mode', 'opd_token_top_k', 'opd_enable_metrics', 'opd_distribution_chunk_size',
        'opd_loss_last_n_tokens', 'opd_require_verified', 'opd_loss_top_k_divergence',
        'opd_teacher_source', 'opd_teacher_refresh_steps') + TEACHER_PROMPT_KEYS
# vllm: a separate teacher server (default). self: the reference copy of the student in the actor worker,
# refreshed to the student every opd_teacher_refresh_steps updates (0 = frozen); CRISP, arXiv 2603.05433.
TEACHER_SOURCES = ('vllm', 'self')


def options(config):
    result = dict(opd_loss_token_mode=config.get('opd_loss_token_mode', 'sample'),
                  opd_token_top_k=config.get('opd_token_top_k', 32),
                  opd_enable_metrics=config.get('opd_enable_metrics', False),
                  opd_distribution_chunk_size=config.get('opd_distribution_chunk_size', 32),
                  opd_loss_last_n_tokens=config.get('opd_loss_last_n_tokens'),
                  opd_loss_top_k_divergence=config.get('opd_loss_top_k_divergence'),
                  opd_require_verified=config.get('opd_require_verified', False),
                  opd_teacher_source=config.get('opd_teacher_source', 'vllm'),
                  opd_teacher_refresh_steps=config.get('opd_teacher_refresh_steps', 0))
    if result['opd_loss_token_mode'] not in MODES:
        raise ValueError(f'opd_loss_token_mode must be one of {MODES}')
    for key in ('opd_token_top_k', 'opd_distribution_chunk_size'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['opd_enable_metrics']) is not bool:
        raise ValueError('opd_enable_metrics must be a boolean')
    tail = result['opd_loss_last_n_tokens']
    if tail is not None and (type(tail) is not int or tail < 1):
        raise ValueError('opd_loss_last_n_tokens must be a positive integer or null')
    if type(result['opd_require_verified']) is not bool:
        raise ValueError('opd_require_verified must be a boolean')
    divergence_k = result['opd_loss_top_k_divergence']
    if divergence_k is not None and (type(divergence_k) is not int or divergence_k < 1):
        raise ValueError('opd_loss_top_k_divergence must be a positive integer or null')
    if result['opd_teacher_source'] not in TEACHER_SOURCES:
        raise ValueError(f'opd_teacher_source must be one of {TEACHER_SOURCES}')
    refresh = result['opd_teacher_refresh_steps']
    if type(refresh) is not int or refresh < 0:
        raise ValueError('opd_teacher_refresh_steps must be a nonnegative integer')
    if result['opd_teacher_source'] == 'self' and result['opd_loss_token_mode'] != 'full_vocab':
        raise ValueError('A self-teacher scores the full vocabulary: use opd_loss_token_mode full_vocab')
    if refresh and result['opd_teacher_source'] != 'self':
        raise ValueError('Only a self-teacher (opd_teacher_source: self) can be refreshed')
    result.update(teacher_prompt_options(config))
    return result


def needs_features(config):
    return config['opd_loss_token_mode'] != 'sample' or config['opd_enable_metrics']


def selects_response_tokens(config):
    return (config.get('opd_loss_last_n_tokens') is not None or config.get('opd_require_verified', False)
            or config.get('opd_loss_top_k_divergence') is not None)


def runtime_options():
    return options(json.loads(os.environ.get('OPD_EXTENSIONS_CONFIG', '{}')))
