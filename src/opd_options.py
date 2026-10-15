"""New OPD support options, separate from the baseline training recipe."""

import json
import os

MODES = ('sample', 'top_k_student', 'top_k_teacher', 'top_k_diff', 'full_vocab')
KEYS = ('opd_loss_token_mode', 'opd_token_top_k', 'opd_enable_metrics', 'opd_distribution_chunk_size')


def options(config):
    result = dict(opd_loss_token_mode=config.get('opd_loss_token_mode', 'sample'),
                  opd_token_top_k=config.get('opd_token_top_k', 32),
                  opd_enable_metrics=config.get('opd_enable_metrics', False),
                  opd_distribution_chunk_size=config.get('opd_distribution_chunk_size', 32))
    if result['opd_loss_token_mode'] not in MODES:
        raise ValueError(f'opd_loss_token_mode must be one of {MODES}')
    for key in ('opd_token_top_k', 'opd_distribution_chunk_size'):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(result['opd_enable_metrics']) is not bool:
        raise ValueError('opd_enable_metrics must be a boolean')
    return result


def needs_features(config):
    return config['opd_loss_token_mode'] != 'sample' or config['opd_enable_metrics']


def runtime_options():
    return options(json.loads(os.environ.get('OPD_EXTENSIONS_CONFIG', '{}')))
