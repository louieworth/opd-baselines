"""Opt-in verl extensions for current-student vocabulary losses and diagnostics."""

import asyncio
from functools import lru_cache, wraps
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.opd_math import METRIC_NAMES, distribution_metrics, reverse_kl_loss
from src.opd_options import needs_features, runtime_options
from src.opd_vllm_capture import install_on_worker, sequence_key

_HEADS = {}


def cache_root():
    return Path(json.loads(os.environ['OPD_EXTENSIONS_CONFIG'])['feature_cache'])


@lru_cache(maxsize=2)
def cpu_head(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def packed_outputs(student_logits, data, config):
    """Align each response with its preceding next-token prediction position."""
    opts = runtime_options()
    tokens = data['input_ids']
    references = list(data['opd_teacher_features'])
    if not tokens.is_nested:
        raise ValueError('OPD vocabulary extensions require the existing remove-padding FSDP2 path.')
    offsets = tokens.offsets().tolist()
    outputs = {name: [] for name in METRIC_NAMES} if opts['opd_enable_metrics'] else {}
    mode = opts['opd_loss_token_mode']
    if mode != 'sample':
        outputs['opd_kl'] = []
    for i, reference in enumerate(references):
        ids = tokens[i]
        artifact = torch.load(reference['path'], map_location='cpu', weights_only=True, mmap=True)
        p, r = reference['prompt_length'], reference['response_length']
        if len(ids) != p + r or not torch.equal(ids.detach().cpu(), artifact['input_ids']):
            raise ValueError('Teacher feature tokens/prompt boundary do not match this training sample.')
        head_path = reference['head_path']
        key = (head_path, str(student_logits.device))
        if key not in _HEADS:
            head = cpu_head(head_path)
            _HEADS[key] = (head['weight'].to(student_logits.device),
                           None if head['bias'] is None else head['bias'].to(student_logits.device))
        weight, bias = _HEADS[key]
        if student_logits.size(-1) != weight.size(0):
            raise ValueError('OPD teacher/student vocabulary sizes differ.')
        hidden = artifact['hidden_states']
        if hidden.size(0) != r or not torch.equal(artifact['response_tokens'], ids[p:].detach().cpu()):
            raise ValueError('Teacher response hidden states are misaligned.')
        chunks = {name: [] for name in outputs}

        def run(student, features, weight=weight, bias=bias):
            teacher = F.linear(features.to(weight.dtype), weight, bias)
            values = []
            if opts['opd_enable_metrics']:
                values.extend(distribution_metrics(student, teacher, opts['opd_token_top_k']).values())
            if mode != 'sample':
                values.append(reverse_kl_loss(student, teacher, mode, opts['opd_token_top_k'],
                                              config.distillation_loss.loss_max_clamp))
            return tuple(values)

        for start in range(0, r, opts['opd_distribution_chunk_size']):
            end = min(r, start + opts['opd_distribution_chunk_size'])
            student = student_logits[0, offsets[i] + p - 1 + start:offsets[i] + p - 1 + end]
            features = hidden[start:end].to(student.device)
            if mode == 'sample':
                with torch.no_grad():
                    values = run(student.detach(), features)
            else:
                values = checkpoint(run, student, features, use_reentrant=False)
            for name, value in zip(chunks, values, strict=True):
                chunks[name].append(value)
        for name in outputs:
            outputs[name].append(F.pad(torch.cat(chunks[name]), (p - 1, 1)))
    return {name: torch.cat(values).unsqueeze(0) for name, values in outputs.items()}


def vocabulary_loss(config, distillation_config, model_output=None, data=None, dp_group=None,
                    student_logits=None, data_format='thd'):
    from verl.trainer.distillation import losses as native
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding
    from verl.workers.utils.losses import ppo_loss

    opts = runtime_options()
    if student_logits is not None:
        return packed_outputs(student_logits, data, distillation_config)
    if opts['opd_loss_token_mode'] == 'sample':
        # Preserve k1/PPO, task-reward and clipping behavior exactly.
        loss, metrics = native.distillation_ppo_loss(config, distillation_config, model_output, data, dp_group)
    else:
        policy_loss, metrics = ppo_loss(config, model_output, data, dp_group)
        settings = distillation_config.distillation_loss
        values = no_padding_2_padding(model_output['opd_kl'], data)
        kl = agg_loss(values, data['response_mask'], config.loss_agg_mode, **config.global_batch_info)
        loss = (policy_loss + settings.distillation_loss_coef * kl) if settings.use_task_rewards else kl
        metrics['distillation/loss'] = Metric(AggregationType.SUM, kl.detach())
    if opts['opd_enable_metrics']:
        mask = data['response_mask'].bool()
        metrics['opd/valid_tokens'] = Metric(AggregationType.SUM, mask.sum().detach())
        for name in METRIC_NAMES:
            values = no_padding_2_padding(model_output[name], data)
            metrics[name + '_sum'] = Metric(AggregationType.SUM, values[mask].sum().detach())
    return loss, metrics


async def teacher_batch(self, data):
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.experimental.teacher_loop.teacher_manager import _unpad_teacher_inputs, _pad_teacher_outputs

    multimodal = data.non_tensor_batch.get('teacher_multi_modal_data')
    prompt_width, response_width = data.batch['prompts'].shape[1], data.batch['responses'].shape[1]
    root = cache_root()

    async def score(i):
        ids, p, r = _unpad_teacher_inputs(data[i:i+1])
        if multimodal is not None and multimodal[i]:
            raise ValueError('OPD vocabulary extensions currently support text-only teacher scoring.')
        # Keep native sampled-token teacher outputs. Vocabulary targets come from captured features.
        params = dict(max_tokens=1, temperature=1.0, prompt_logprobs=0, _opd_response_length=r)
        result = await self.generate(request_id=sequence_key(ids, r), prompt_ids=ids, sampling_params=params)
        key = sequence_key(ids, r)
        reference = dict(path=str(root/key[:2]/(key+'.pt')), head_path=str(root/'output_head.pt'),
                         prompt_length=p, response_length=r)
        if not Path(reference['path']).is_file():
            raise RuntimeError('vLLM did not produce the required teacher hidden-state cache.')
        teacher_ids = torch.tensor(result.extra_fields['prompt_ids'], dtype=torch.int32)
        logprobs = torch.tensor(result.extra_fields['prompt_logprobs'])
        padded = _pad_teacher_outputs(teacher_ids, logprobs, prompt_width, response_width, p, r, self.pad_token_id)
        return padded, reference

    results = await asyncio.gather(*(score(i) for i in range(len(data))))
    return DataProto(batch=TensorDict(dict(teacher_ids=torch.cat([item[0][0] for item in results]),
                                         teacher_logprobs=torch.cat([item[0][1] for item in results])),
                                    batch_size=len(data)),
                     non_tensor_batch={'opd_teacher_features': np.array([item[1] for item in results], dtype=object)})


def install():
    if not needs_features(runtime_options()):
        return
    from verl.workers import engine_workers
    if getattr(engine_workers, '_opd_extensions_installed', False):
        return
    from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    engine_workers.distillation_ppo_loss = vocabulary_loss
    AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch = teacher_batch
    original_generate = vLLMHttpServer.generate

    @wraps(original_generate)
    async def generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        params = dict(sampling_params)
        response_length = params.pop('_opd_response_length', None)
        if response_length is not None:
            if not hasattr(self, '_opd_install_lock'):
                self._opd_install_lock = asyncio.Lock()
            async with self._opd_install_lock:
                if not getattr(self, '_opd_capture_installed', False):
                    await self.engine.collective_rpc(install_on_worker, args=(str(cache_root()),))
                    self._opd_capture_installed = True
            request_id = f'opdfeat-{sequence_key(prompt_ids, response_length)}-{response_length}-{uuid4().hex}'
        return await original_generate(self, prompt_ids=prompt_ids, sampling_params=params,
                                       request_id=request_id, **kwargs)

    vLLMHttpServer.generate = generate
    original_update = engine_workers.ActorRolloutRefWorker.update_actor

    @wraps(original_update)
    def update_actor(self, *args, **kwargs):
        try:
            return original_update(self, *args, **kwargs)
        finally:
            _HEADS.clear()  # Release teacher heads before vLLM is woken on these GPUs.

    engine_workers.ActorRolloutRefWorker.update_actor = update_actor
    engine_workers._opd_extensions_installed = True


install()
