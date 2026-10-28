"""Add comparable GRPO/OPD W&B keys without changing native training metrics."""

from copy import deepcopy
import json
import os


def comparable_metrics(metrics, step):
    result = dict(metrics)
    if any(key.startswith('actor/') for key in metrics):
        result['train/step'] = max(0, step - 1)  # Lightning uses zero-based update IDs.
        for old, new in [('actor/distillation/loss', 'train/loss'), ('actor/grad_norm', 'train/grad_norm'),
                         ('actor/lr', 'train/lr-pg_0'), ('actor/distillation/pg_loss', 'train/pg_loss'),
                         ('actor/distillation/pg_clipfrac', 'train/pg_clipfrac'),
                         ('actor/distillation/ppo_kl', 'train/ppo_kl')]:
            if old in metrics:
                result[new] = metrics[old]
        for name in ('selected_trajectories', 'filtered_trajectories', 'loss_tokens', 'update_skipped',
                     'selected_divergence_mean', 'selected_divergence_max'):
            key = 'actor/opd/' + name
            if key in metrics:
                result['train/opd/' + name] = metrics[key]
        denominator = metrics.get('actor/opd/valid_tokens', 0)
        for key, value in metrics.items():
            if key.startswith('actor/adopd/'):
                result['train/' + key[len('actor/'):]] = value
        if denominator:
            for key, value in metrics.items():
                if key.startswith('actor/opd/') and key.endswith('_pct_sum'):
                    result['train/' + key[len('actor/'):-4]] = value / denominator
    if any(key.startswith('eval/') for key in metrics):
        result['eval/step'] = step  # Evaluations follow this many completed updates.
    for old, new in [('timing_s/gen', 'perf/rollout_time'), ('timing_s/teacher', 'perf/teacher_time'),
                     ('timing_s/update_actor', 'perf/actor_train_time'),
                     ('timing_s/update_weights', 'perf/update_weights_time'), ('timing_s/step', 'perf/step_time')]:
        if old in metrics:
            result[new] = metrics[old]
    return result


def install_tracking():
    import verl.utils.tracking as tracking
    if getattr(tracking.Tracking, '_opd_comparable', False):
        return
    parent = tracking.Tracking

    class ComparableTracking(parent):
        _opd_comparable = True

        def __init__(self, *args, config=None, **kwargs):
            self._opd_batch = None
            if config is not None:
                config = deepcopy(config)
                if config.get('distillation', {}).get('enabled', False):
                    config['opd_extensions'] = json.loads(os.environ.get('OPD_EXTENSIONS_CONFIG', '{}'))
                if os.environ.get('ADOPD_CONFIG'):
                    config['adopd'] = json.loads(os.environ['ADOPD_CONFIG'])
                config['training_framework'] = 'verl'
                self._opd_batch = (config['data']['train_batch_size'],
                                   config['actor_rollout_ref']['rollout']['n'])
            super().__init__(*args, config=config, **kwargs)
            if 'wandb' in self.logger:
                wandb = self.logger['wandb']
                for namespace, axis in [('train', 'train/step'), ('eval', 'eval/step')]:
                    wandb.define_metric(axis)
                    wandb.define_metric(namespace+'/*', step_metric=axis)

        def log(self, data, step, *args, **kwargs):
            logged = comparable_metrics(data, step)
            if 'train/step' in logged and self._opd_batch is not None:
                prompts, responses_per_prompt = self._opd_batch
                logged.update({'train/prompts_per_step': prompts,
                               'train/responses_per_step': prompts * responses_per_prompt,
                               'train/cumulative_prompt_visits': step * prompts,
                               'train/cumulative_response_samples': step * prompts * responses_per_prompt})
            if 'eval/step' in logged and self._opd_batch is not None:
                prompts, responses_per_prompt = self._opd_batch
                logged.update({'eval/cumulative_prompt_visits': step * prompts,
                               'eval/cumulative_response_samples': step * prompts * responses_per_prompt})
            return super().log(logged, step, *args, **kwargs)

    tracking.Tracking = ComparableTracking
