"""Driver-side ADOPD mixin; all group statistics are fixed before DP sharding."""

import json
from pathlib import Path

import numpy as np
import torch

from src.adopd_cache import TraceCache, halt_id
from src.adopd_math import group_advantages, masked_sequence_mean, sampled_distortion
from src.adopd_options import runtime_options
from src.eval_metrics import compute_eval_metrics


class AdopdTrainerMixin:
    def _validate(self, *args, **kwargs):
        mode = runtime_options().get('adopd_eval_plain_prompt', True)
        self._adopd_eval_plain = mode is True or (mode == 'initial' and self.global_steps == 0)
        try:
            return super()._validate(*args, **kwargs)
        finally:
            del self._adopd_eval_plain

    def _get_gen_batch(self, batch):
        gen = super()._get_gen_batch(batch)
        if hasattr(self, '_adopd_eval_plain'):  # validation rows only; training rows never carry it
            gen.non_tensor_batch['adopd_eval_plain'] = np.full(len(gen), self._adopd_eval_plain, dtype=object)
        return gen

    def init_workers(self):
        opts = runtime_options()
        actor, rollout = self.config.actor_rollout_ref.actor, self.config.actor_rollout_ref.rollout
        expected = opts['group_size'] if opts['adopd_use_group_rollouts'] and not opts.get('eval_only') else 1
        if rollout.n != expected:
            raise ValueError('ADOPD group toggle and rollout.n disagree')
        if actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != self.config.data.train_batch_size:
            raise ValueError('ADOPD requires one optimizer update per complete rollout batch (ppo_epochs=1)')
        if self.config.algorithm.get('rollout_correction') is not None:
            raise ValueError('ADOPD requires recomputed old logprobs without rollout correction')
        if not opts.get('eval_only'):
            if self.config.distillation.teacher_model.enable_resource_pool:
                raise ValueError('ADOPD requires colocated teacher scoring')
            if self.config.distillation.distillation_loss.loss_mode != 'k1':
                raise ValueError('ADOPD uses the native k1 registration; select its answer loss with opd_loss_token_mode')
        super().init_workers()
        if opts.get('eval_only', False):
            return
        manager = self.teacher_model_manager
        teacher_tokenizer = manager.rollout_replicas[0].model_config.tokenizer
        if self.tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError('V2 self-teacher and student must use the same token vocabulary')
        manager.server_manager._adopd_halt_id = halt_id(teacher_tokenizer)
        manager.server_manager._adopd_cache = TraceCache(opts['adopd_teacher_trace_cache'], opts['cache_identity'])

    def _compute_advantage(self, batch):
        opts = runtime_options()
        answer = batch.batch['adopd_answer_mask'].bool()
        valid = answer.any(-1)
        count = int(valid.sum())
        batch.batch['adopd_normalizer'] = torch.full((len(batch),), float(count))
        # Native monitoring expects these two arrays. They are inert compatibility
        # fields: there is NO group Eq. (5) or native GRPO calculation in the no-group
        # path. Its length policy gradient is computed directly in the actor loss.
        batch.batch['advantages'] = torch.zeros_like(batch.batch['old_log_probs'])
        batch.batch['returns'] = torch.zeros_like(batch.batch['old_log_probs'])
        if opts['adopd_use_group_rollouts']:
            if opts.get('opd_loss_token_mode', 'sample') == 'full_vocab':
                difference = batch.batch['adopd_old_answer_kl']
            else:
                loss_config = self.config.distillation.distillation_loss
                difference = sampled_distortion(batch.batch['old_log_probs'],
                    batch.batch['adopd_teacher_log_probs'], loss_config.log_prob_min_clamp,
                    loss_config.loss_max_clamp)
            distortions = masked_sequence_mean(difference, answer)
            advantages = group_advantages(distortions, batch.batch['adopd_trace_length'],
                batch.non_tensor_batch['uid'], valid, opts['group_size'],
                opts['adopd_reasoning_budget'], opts['adopd_length_penalty'])
            batch.batch['adopd_advantage'] = advantages
            batch.batch['advantages'] = advantages[:, None] * batch.batch['adopd_trace_mask']
        return batch

    def _compute_old_log_prob(self, batch):
        opts = runtime_options()
        if opts.get('opd_loss_token_mode', 'sample') != 'full_vocab' or not opts['adopd_use_group_rollouts']:
            return super()._compute_old_log_prob(batch)
        from verl.protocol import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

        # Reuse the old-policy forward to evaluate exact answer KL before the
        # group advantage is computed. No optimizer update happens in this pass.
        packed = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(packed, calculate_entropy=True, compute_loss=True,
                             distillation_use_topk=True, adopd_scoring=True)
        output = self.actor_rollout_wg.compute_log_prob(packed)
        values = {
            'old_log_probs': no_padding_2_padding(tu.get(output, 'log_probs'), packed).float(),
            'entropys': no_padding_2_padding(tu.get(output, 'entropy'), packed).float(),
            'adopd_old_answer_kl': no_padding_2_padding(tu.get(output, 'opd_kl'), packed).float(),
        }
        routed = tu.get(output, 'routed_experts')
        if routed is not None:
            values['routed_experts'] = routed
        return DataProto.from_tensordict(tu.get_tensordict(values)), tu.get(output, 'metrics')['mfu']

    def _update_actor(self, batch):
        from verl.protocol import DataProto
        valid = batch.batch['adopd_answer_mask'].any(-1)
        metrics = {
            'actor/adopd/valid_trajectories': int(valid.sum()),
            'actor/adopd/invalid_trajectories': int((~valid).sum()),
            'actor/adopd/forced_halt_rate': float(batch.batch['adopd_forced_halt'].float().mean()),
            'actor/adopd/reasoning_length': float(batch.batch['adopd_trace_length'].float().mean()),
            'actor/adopd/answer_length': float(batch.batch['adopd_answer_mask'].sum(-1).float().mean()),
            'actor/adopd/update_skipped': int(not valid.any()),
        }
        if not valid.any():
            metrics['actor/distillation/loss'] = 0.0
            return DataProto.from_single_dict(data={}, meta_info={'metrics': metrics})
        output = super()._update_actor(batch)
        output.meta_info['metrics'].update(metrics)
        return output

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        # verl measures the returned, training-sized response; the length that was scored is the full rollout.
        full = dict(reward_extra_infos_dict, response_length=reward_extra_infos_dict['full_response_length'])
        metrics = super()._val_metrics_update(data_sources, sample_uids, full, sample_turns)
        info = reward_extra_infos_dict
        # The plain-prompt rollout and every named eval variant are scored like the main one, each in its own group.
        groups = [('', 'eval')] + ([('nobudget_', 'eval_nobudget')] if 'nobudget_acc' in info else [])
        groups += [(f"{v['name']}_", f"eval_{v['name']}") for v in runtime_options().get('adopd_eval_variants') or []]
        for prefix, group in groups[1:]:
            lengths = info['nobudget_response_length'] if prefix == 'nobudget_' else info[prefix + 'full_response_length']
            extra = compute_eval_metrics(data_sources, sample_uids, info[prefix + 'acc'],
                k=self.config.actor_rollout_ref.rollout.val_kwargs.n, response_lengths=lengths)
            metrics.update({key.replace('eval/', f'{group}/', 1): value for key, value in extra.items()})
        for source in set(data_sources):
            indices = [i for i, value in enumerate(data_sources) if value == source]
            for prefix, group in groups:
                fields = ('reasoning_length', 'answer_length', 'finished', 'over_budget')
                fields += ('forced_halt',) if prefix + 'forced_halt' in info else ()
                for field in fields:
                    values = info[prefix + field]
                    metrics[f'{group}/{source}/{field}'] = sum(float(values[i]) for i in indices) / len(indices)
        path = Path(self.config.trainer.default_local_dir) / 'eval_metrics'
        path.mkdir(parents=True, exist_ok=True)
        (path / f'{self.global_steps}.json').write_text(json.dumps(metrics, indent=2) + '\n')
        return metrics
