"""Answer-only sampled/full-vocabulary teacher scoring and native FSDP2 losses."""

import asyncio
from pathlib import Path

import numpy as np
import torch

from src.adopd_math import per_sequence_loss, response_masks
from src.adopd_options import runtime_options


async def teacher_batch(self, data):
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.experimental.teacher_loop.teacher_manager import _unpad_teacher_inputs

    opts = runtime_options()
    responses = data.batch['responses']
    teacher_prompts = data.non_tensor_batch.get('adopd_teacher_prompt_ids')
    if teacher_prompts is None or len(teacher_prompts) != len(data):
        raise ValueError('ADOPD rollout is missing original teacher prompt IDs')
    width = data.batch['prompts'].shape[1]
    answer_mask, trace_mask, lengths, forced = response_masks(
        responses, data.batch['response_mask'], data.batch['attention_mask'][:, width:],
        self._adopd_halt_id, opts['adopd_reasoning_budget'])
    targets = torch.zeros(responses.shape, dtype=torch.float32)
    full_vocab = opts.get('opd_loss_token_mode', 'sample') == 'full_vocab'
    references = [None] * len(data)

    async def score(i):
        answer = responses[i][answer_mask[i]].tolist()
        if not answer:
            return
        # The student's prompt contains the budget instruction. V2 references
        # and cache keys are always based on the original unbudgeted question.
        prompt = list(teacher_prompts[i])
        cached = self._adopd_cache.read(prompt, self._adopd_halt_id)
        prefix = prompt + cached['trace_ids']
        # The answer prefix is the STUDENT'S answer, never the cached teacher answer.
        params = dict(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        if full_vocab:
            params['_opd_response_length'] = len(answer)
        output = await self.generate(request_id=self._adopd_cache.path(prompt).stem,
            prompt_ids=prefix + answer, sampling_params=params)
        if full_vocab:
            from src.opd_runtime import cache_root
            from src.opd_vllm_capture import sequence_key
            student_ids, student_p, student_r = _unpad_teacher_inputs(data[i:i + 1])
            key = sequence_key(prefix + answer, len(answer))
            root = cache_root()
            reference = dict(path=str(root / key[:2] / (key + '.pt')),
                head_path=str(root / 'output_head.pt'), prompt_length=student_p,
                response_length=student_r, response_offset=student_r - len(answer),
                student_sequence_key=sequence_key(student_ids, student_r),
                teacher_sequence_key=key, teacher_prompt_length=len(prefix))
            if not Path(reference['path']).is_file():
                raise RuntimeError('vLLM did not capture ADOPD teacher answer features')
            references[i] = reference
        tokens = torch.tensor(output.extra_fields['prompt_ids'], dtype=torch.long)
        log_probs = torch.tensor(output.extra_fields['prompt_logprobs'], dtype=torch.float32)
        start, end = len(prefix) - 1, len(prefix) + len(answer) - 1
        if tokens.shape != log_probs.shape or tokens.shape != (len(prefix) + len(answer), 1):
            raise ValueError('Unexpected sampled-token teacher logprob shape')
        if tokens[start:end, 0].tolist() != answer:
            raise ValueError('V2 teacher answer predictions are causally misaligned')
        values = log_probs[start:end, 0]
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite teacher answer log probabilities')
        targets[i, answer_mask[i]] = values

    await asyncio.gather(*(score(i) for i in range(len(data))))
    return DataProto(batch=TensorDict({
        'adopd_teacher_log_probs': targets,
        'adopd_answer_mask': answer_mask,
        'adopd_trace_mask': trace_mask,
        'adopd_trace_length': lengths,
        'adopd_forced_halt': forced,
    }, batch_size=len(data)), non_tensor_batch=(
        {'opd_teacher_features': np.array(references, dtype=object)} if full_vocab else {}))


def answer_loss(config, distillation_config, model_output=None, data=None, dp_group=None,
                student_logits=None, **kwargs):
    from verl.workers.utils.padding import no_padding_2_padding
    from verl.utils.metric import AggregationType, Metric
    from verl.utils import tensordict_utils as tu
    opts = runtime_options()
    if student_logits is not None:
        from src.opd_runtime import packed_outputs
        return packed_outputs(student_logits, data, distillation_config, answer_only=True)
    if tu.get_non_tensor_data(data, 'adopd_scoring', default=False):
        values = model_output['log_probs']
        values = values.values() if values.is_nested else values
        return values.sum() * 0.0, {}
    current = no_padding_2_padding(model_output['log_probs'], data).float()
    loss_config = distillation_config.distillation_loss
    answer_mask, trace_mask = data['adopd_answer_mask'], data['adopd_trace_mask']
    valid = answer_mask.bool().any(-1)
    advantages = data['adopd_advantage'] if opts['adopd_use_group_rollouts'] else None
    full_kl = (no_padding_2_padding(model_output['opd_kl'], data).float()
               if opts.get('opd_loss_token_mode', 'sample') == 'full_vocab' else None)
    losses, distortions, reinforce = per_sequence_loss(current, data['adopd_teacher_log_probs'],
        data['old_log_probs'], answer_mask, trace_mask, advantages,
        trace_lengths=data['adopd_trace_length'], budget=opts['adopd_reasoning_budget'],
        length_penalty=opts['adopd_length_penalty'],
        log_prob_min_clamp=loss_config.log_prob_min_clamp,
        loss_max_clamp=loss_config.loss_max_clamp, answer_kl=full_kl)
    length_cost = (opts['adopd_length_penalty'] * data['adopd_trace_length'].detach().to(current)
                   / opts['adopd_reasoning_budget'])
    # FSDP averages gradients over DP ranks. Sum microbatch sequence means and
    # normalize by the FULL batch's valid trajectory count, never local token counts.
    scale = data['dp_size'] / data['adopd_normalizer'][0].clamp_min(1)
    loss = torch.where(valid, losses, 0.0).sum() * scale
    metrics = {
        'distillation/loss': Metric(AggregationType.SUM, loss.detach()),
        'adopd/answer_kl': Metric(AggregationType.SUM,
            (torch.where(valid, distortions, 0.0).sum() * scale).detach()),
        'adopd/trace_reinforce': Metric(AggregationType.SUM,
            (torch.where(valid, reinforce, 0.0).sum() * scale).detach()),
        'adopd/length_cost': Metric(AggregationType.SUM,
            (torch.where(valid, length_cost, 0.0).sum() * scale).detach()),
        'adopd/regularized_distortion': Metric(AggregationType.SUM,
            (torch.where(valid, distortions + length_cost, 0.0).sum() * scale).detach()),
    }
    return loss, metrics


# Keep the existing import available for sampled-mode callers/tests.
sampled_answer_loss = answer_loss


def install():
    if not runtime_options().get('enabled', False):
        return
    if runtime_options().get('opd_loss_token_mode') == 'full_vocab':
        from src.opd_runtime import install as install_features
        install_features()
        if runtime_options().get('h100_runtime'):
            from src.opd_h100_runtime import install as install_h100
            install_h100()
    from verl.workers import engine_workers
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    engine_workers.distillation_ppo_loss = answer_loss
    AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch = teacher_batch


install()
