"""Opt-in verl extensions for current-student vocabulary losses and diagnostics."""

import asyncio
from functools import lru_cache, wraps
import gc
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from src import opd_self_teacher as self_teacher
from src.opd_math import METRIC_NAMES, distribution_metrics, reverse_kl_loss
from src.opd_options import needs_features, runtime_options, selects_response_tokens
from src.opd_vllm_capture import sequence_key
from src.teacher_prompt import has_teacher_prompt, teacher_prompt_ids

_HEADS = {}


def _release_teacher_heads(opts):
    """Return unused actor memory before the colocated vLLM restores its weights."""
    _HEADS.clear()
    if not needs_features(opts):
        return
    gc.collect()
    if torch.cuda.is_available():
        # Wait for work on every stream before returning cached blocks to CUDA.
        # Removing the head reference alone leaves its storage in PyTorch's pool.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cache_root():
    return Path(json.loads(os.environ['OPD_EXTENSIONS_CONFIG'])['feature_cache'])


@lru_cache(maxsize=2)
def cpu_head(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


class _ChunkedDistributionOutputs(torch.autograd.Function):
    """Recompute chunk gradients into one buffer instead of full-size SliceBackward buffers."""

    @staticmethod
    def forward(ctx, logits, hidden, weight, bias, start, opts, maximum):
        ctx.save_for_backward(logits, hidden, weight, bias)
        ctx.start, ctx.opts, ctx.maximum = start, dict(opts), maximum
        ctx.set_materialize_grads(False)
        rows = [[] for _ in range((len(METRIC_NAMES) if opts['opd_enable_metrics'] else 0) + 1)]
        for first in range(0, len(hidden), opts['opd_distribution_chunk_size']):
            last = min(len(hidden), first + opts['opd_distribution_chunk_size'])
            student = logits[0, start + first:start + last]
            features = hidden[first:last].to(device=weight.device, dtype=weight.dtype)
            teacher = F.linear(features, weight, bias)
            values = []
            if opts['opd_enable_metrics']:
                values.extend(distribution_metrics(student, teacher, opts['opd_token_top_k']).values())
            values.append(reverse_kl_loss(student, teacher, opts['opd_loss_token_mode'],
                                          opts['opd_token_top_k'], maximum))
            for row, value in zip(rows, values, strict=True):
                row.append(value)
        outputs = tuple(torch.cat(row) for row in rows)
        ctx.mark_non_differentiable(*outputs[:-1])
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        loss_gradient = grad_outputs[-1]
        if loss_gradient is None:
            return (None,) * 7
        logits, hidden, weight, bias = ctx.saved_tensors
        opts, start = ctx.opts, ctx.start
        # Each response chunk used to send a dense, sequence-sized gradient
        # through SliceBackward/SelectBackward and their accumulation. Allocate
        # this buffer once, and detach each chunk before building its local graph.
        gradient = torch.zeros_like(logits)
        for first in range(0, len(hidden), opts['opd_distribution_chunk_size']):
            last = min(len(hidden), first + opts['opd_distribution_chunk_size'])
            with torch.no_grad():
                features = hidden[first:last].to(device=weight.device, dtype=weight.dtype)
                teacher = F.linear(features, weight, bias)
            with torch.enable_grad():
                student = logits[0, start + first:start + last].detach().requires_grad_(True)
                losses = reverse_kl_loss(student, teacher, opts['opd_loss_token_mode'],
                                         opts['opd_token_top_k'], ctx.maximum)
                chunk_gradient, = torch.autograd.grad(losses, student, loss_gradient[first:last])
            gradient[0, start + first:start + last] = chunk_gradient
        return gradient, None, None, None, None, None, None


def packed_outputs(student_logits, data, config, *, answer_only=False):
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
        if reference is None:
            # ADOPD can retain an empty-answer trajectory for DP collectives.
            zero = student_logits[0, offsets[i]:offsets[i + 1], 0] * 0.0
            for row in outputs.values():
                row.append(zero)
            continue
        if 'teacher_input_ids' in reference:
            p, r, response_offset = reference['prompt_length'], reference['response_length'], 0
            hidden, weight, bias = _self_teacher_targets(reference, ids, student_logits.device)
        else:
            hidden, weight, bias, p, r, response_offset = _cached_teacher_targets(reference, ids, student_logits)
        if student_logits.size(-1) != weight.size(0):
            raise ValueError('OPD teacher/student vocabulary sizes differ.')
        if mode != 'sample':
            values = _ChunkedDistributionOutputs.apply(
                student_logits, hidden, weight, bias, offsets[i] + p + response_offset - 1, opts,
                None if answer_only else config.distillation_loss.loss_max_clamp,
            )
            for name, value in zip(outputs, values, strict=True):
                outputs[name].append(F.pad(value, (p + response_offset - 1, 1)))
            continue
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
            with torch.no_grad():
                values = run(student.detach(), features)
            for name, value in zip(chunks, values, strict=True):
                chunks[name].append(value)
        for name in outputs:
            outputs[name].append(F.pad(torch.cat(chunks[name]), (p - 1, 1)))
    return {name: torch.cat(values).unsqueeze(0) for name, values in outputs.items()}


def _self_teacher_targets(reference, ids, device):
    """Score the student's own response now, after the teacher prompt, with the refreshed student copy."""
    p, r = reference['prompt_length'], reference['response_length']
    teacher_ids, teacher_p = reference['teacher_input_ids'], reference['teacher_prompt_length']
    if len(ids) != p + r or list(teacher_ids[teacher_p:]) != ids[p:].detach().cpu().tolist():
        raise ValueError('Self-teacher tokens do not match this training sample.')
    hidden = self_teacher.response_hidden(self_teacher.teacher_module(), teacher_ids, teacher_p, r, device)
    weight, bias = self_teacher.output_head(device)
    return hidden, weight, bias


def _cached_teacher_targets(reference, ids, student_logits):
    """Teacher hidden states captured by the teacher server, and its saved output head."""
    artifact = torch.load(reference['path'], map_location='cpu', weights_only=True, mmap=True)
    p, r = reference['prompt_length'], reference['response_length']
    response_offset = reference.get('response_offset', 0)
    if not 0 <= response_offset < r:
        raise ValueError('Invalid teacher feature response offset')
    feature_length = r - response_offset
    if 'student_sequence_key' in reference:
        matches = (sequence_key(ids.detach().cpu().tolist(), r) == reference['student_sequence_key']
                   and sequence_key(artifact['input_ids'].tolist(), feature_length) == reference['teacher_sequence_key']
                   and artifact['prompt_length'] == reference['teacher_prompt_length'])
    else:
        matches = torch.equal(ids.detach().cpu(), artifact['input_ids'])
    if len(ids) != p + r or not matches:
        raise ValueError('Teacher feature tokens/prompt boundary do not match this training sample.')
    head_path = reference['head_path']
    key = (head_path, str(student_logits.device))
    if key not in _HEADS:
        head = cpu_head(head_path)
        _HEADS[key] = (head['weight'].to(student_logits.device),
                       None if head['bias'] is None else head['bias'].to(student_logits.device))
    weight, bias = _HEADS[key]
    hidden = artifact['hidden_states']
    if hidden.size(0) != feature_length or not torch.equal(artifact['response_tokens'], ids[p + response_offset:].detach().cpu()):
        raise ValueError('Teacher response hidden states are misaligned.')
    return hidden, weight, bias, p, r, response_offset


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
    if selects_response_tokens(opts):
        # Native distillation runs before ppo_loss initializes these statistics.
        # Refresh them for every microbatch so the selected loss is normalized by
        # current global selected-token count or trajectory count, according to
        # the actor's aggregation mode, including on step one.
        config.global_batch_info.update(dp_size=data['dp_size'],
                                        batch_num_tokens=data['batch_num_tokens'],
                                        global_batch_size=data['global_batch_size'],
                                        loss_scale_factor=config.loss_scale_factor)
        if not data['response_mask'].bool().any():
            # A DP rank/microbatch may contain only rejected trajectories.
            # Keep its forward/backward collectives, with no token gradients;
            # native min/max reductions cannot handle an empty selection.
            # Obtain the native metric schema on a throwaway mask so every DP
            # rank reports the same metric keys and accumulation counts.
            probe = data.clone(recurse=False)
            probe_mask = torch.zeros_like(data['response_mask'])
            probe_mask[0, 0] = 1
            probe['response_mask'] = probe_mask
            with torch.no_grad():
                if opts['opd_loss_token_mode'] == 'sample':
                    _, metrics = native.distillation_ppo_loss(config, distillation_config, model_output, probe, dp_group)
                else:
                    _, metrics = ppo_loss(config, model_output, probe, dp_group)
                    metrics['distillation/loss'] = Metric(AggregationType.SUM, 0.0)
            for name, metric in metrics.items():
                if isinstance(metric, Metric):
                    neutral = {AggregationType.MIN: float('inf'), AggregationType.MAX: -float('inf')}
                    metrics[name] = Metric(metric.aggregation, neutral.get(metric.aggregation, 0.0))
                else:
                    metrics[name] = 0.0
            log_probs = model_output['log_probs']
            values = log_probs.values() if log_probs.is_nested else log_probs
            zero = values.sum() * 0.0
            if opts['opd_enable_metrics']:
                metrics['opd/valid_tokens'] = Metric(AggregationType.SUM, zero.detach())
                metrics.update({name + '_sum': Metric(AggregationType.SUM, zero.detach())
                                for name in METRIC_NAMES})
            return zero, metrics
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
    opts = runtime_options()
    features = needs_features(opts)
    root = cache_root() if features else None

    async def score(i):
        ids, p, r = _unpad_teacher_inputs(data[i:i+1])
        if multimodal is not None and multimodal[i]:
            raise ValueError('OPD vocabulary extensions currently support text-only teacher scoring.')
        teacher_input, teacher_p = ids, p
        if has_teacher_prompt(opts):
            prompt = teacher_prompt_ids(self._opd_tokenizer, data.non_tensor_batch['raw_prompt'][i], opts)
            teacher_input, teacher_p = prompt + ids[p:], len(prompt)
        if opts['opd_teacher_source'] == 'self':
            # The teacher is the reference copy of the student in the actor worker; it scores this
            # response during the update. Native teacher outputs are unused by the full-vocabulary KL.
            reference = dict(prompt_length=p, response_length=r, teacher_prompt_length=teacher_p,
                             teacher_input_ids=list(teacher_input))
            unused = torch.zeros((p + r, 1))
            return _pad_teacher_outputs(unused.int(), unused, prompt_width, response_width, p, r,
                                        self.pad_token_id), reference
        # Keep native sampled-token teacher outputs. Vocabulary targets come from captured features.
        params = dict(max_tokens=1, temperature=1.0, prompt_logprobs=0)
        if features:
            params['_opd_response_length'] = r
        result = await self.generate(request_id=sequence_key(teacher_input, r), prompt_ids=teacher_input,
                                     sampling_params=params)
        reference = None
        if features:
            key = sequence_key(teacher_input, r)
            reference = dict(path=str(root/key[:2]/(key+'.pt')), head_path=str(root/'output_head.pt'),
                             prompt_length=p, response_length=r)
            if has_teacher_prompt(opts):
                reference.update(student_sequence_key=sequence_key(ids, r), teacher_sequence_key=key,
                                 teacher_prompt_length=teacher_p)
            if not Path(reference['path']).is_file():
                raise RuntimeError('vLLM did not produce the required teacher hidden-state cache.')
        teacher_ids = torch.tensor(result.extra_fields['prompt_ids'], dtype=torch.int32)
        logprobs = torch.tensor(result.extra_fields['prompt_logprobs'])
        if teacher_ids.shape[0] != len(teacher_input) or logprobs.shape[0] != len(teacher_input):
            raise ValueError('Teacher prompt-logprob output length does not match the scoring input')
        if has_teacher_prompt(opts):
            # vLLM row j predicts token j+1. Map only response predictions back
            # into the student's prompt/response frame before native padding.
            aligned_ids = teacher_ids.new_zeros((p + r, teacher_ids.size(-1)))
            aligned_logprobs = logprobs.new_zeros((p + r, logprobs.size(-1)))
            aligned_ids[p-1:p+r-1] = teacher_ids[teacher_p-1:teacher_p+r-1]
            aligned_logprobs[p-1:p+r-1] = logprobs[teacher_p-1:teacher_p+r-1]
            teacher_ids, logprobs = aligned_ids, aligned_logprobs
        padded = _pad_teacher_outputs(teacher_ids, logprobs, prompt_width, response_width, p, r, self.pad_token_id)
        return padded, reference

    results = await asyncio.gather(*(score(i) for i in range(len(data))))
    return DataProto(batch=TensorDict(dict(teacher_ids=torch.cat([item[0][0] for item in results]),
                                         teacher_logprobs=torch.cat([item[0][1] for item in results])),
                                    batch_size=len(data)),
                     non_tensor_batch=({'opd_teacher_features': np.array([item[1] for item in results], dtype=object)}
                                       if features else {}))


def install():
    opts = runtime_options()
    if has_teacher_prompt(opts):
        from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
        AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch = teacher_batch
    if selects_response_tokens(opts):
        from verl.workers import engine_workers
        engine_workers.distillation_ppo_loss = vocabulary_loss
    if not needs_features(opts):
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
                    # vLLM 0.12 serializes this RPC; use the worker extension's named method.
                    await self.engine.collective_rpc('install_on_worker', args=(str(cache_root()),))
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
            _release_teacher_heads(opts)

    engine_workers.ActorRolloutRefWorker.update_actor = update_actor
    if opts['opd_teacher_source'] == 'self':
        _install_self_teacher(engine_workers.ActorRolloutRefWorker, opts['opd_teacher_refresh_steps'])
    engine_workers._opd_extensions_installed = True


def _install_self_teacher(worker_class, every):
    """The reference model is the teacher; it becomes the student before every `every`-th update."""
    original_init, original_update = worker_class.init_model, worker_class.update_actor

    @wraps(original_init)
    def init_model(self, *args, **kwargs):
        output = original_init(self, *args, **kwargs)
        if self.actor is None or self.ref is None:
            raise RuntimeError('A self-teacher needs the reference model in the actor worker (use_kl_loss)')
        self_teacher.register(self.actor.engine.module, self.ref.engine.module)
        return output

    @wraps(original_update)
    def update_actor(self, *args, **kwargs):
        if self_teacher.before_update(every):
            print(f'OPD self-teacher refreshed to the student before update {self_teacher._STATE["updates"]}',
                  flush=True)
        try:
            return original_update(self, *args, **kwargs)
        finally:
            self_teacher.release_head()

    worker_class.init_model = init_model
    worker_class.update_actor = update_actor


install()
