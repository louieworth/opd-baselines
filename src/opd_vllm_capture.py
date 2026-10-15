"""Capture response hidden states during vLLM's existing prompt-logprob pass.

No second teacher forward, no full-vocabulary probability transport. The hook is
scoped to tagged OPD requests. vLLM still owns batching, TP and GPU sleep/wake.
"""

import hashlib
import json
from pathlib import Path
import types
import uuid

import torch


def sequence_key(ids, response_length):
    return hashlib.sha256(json.dumps([list(ids), int(response_length)]).encode()).hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_chunk(runner, hidden_states, scheduled):
    for req_id in list(runner.num_prompt_logprobs):
        if not req_id.startswith('opdfeat-') or req_id not in scheduled:
            continue
        _, key, response_text, _unique = req_id.split('-')
        response_length = int(response_text)
        request = runner.requests[req_id]
        ids = request.prompt_token_ids
        if sequence_key(ids, response_length) != key:
            raise ValueError('OPD feature request token identity changed.')
        target = runner._opd_cache_root / key[:2] / (key + '.pt')
        if target.is_file():
            continue
        prompt_length = len(ids) - response_length
        if not 0 < response_length < len(ids):
            raise ValueError('OPD teacher requires a nonempty prompt and response.')
        start = request.num_computed_tokens
        lower, upper = max(start, prompt_length - 1), min(start + scheduled[req_id], len(ids) - 1)
        if upper > lower:
            buffer = runner._opd_buffers.get(req_id)
            if buffer is None:
                buffer = (torch.empty(response_length, hidden_states.size(-1), dtype=hidden_states.dtype),
                          torch.zeros(response_length, dtype=torch.bool))
                runner._opd_buffers[req_id] = buffer
            offset = int(runner.query_start_loc.np[runner.input_batch.req_id_to_index[req_id]])
            a, b = lower - prompt_length + 1, upper - prompt_length + 1
            buffer[0][a:b] = hidden_states[offset + lower - start:offset + upper - start].detach().cpu()
            buffer[1][a:b] = True
        buffer = runner._opd_buffers.get(req_id)
        if buffer is not None and buffer[1].all():
            atomic_save({'hidden_states': buffer[0], 'response_tokens': torch.tensor(ids[-response_length:]),
                         'input_ids': torch.tensor(ids), 'prompt_length': prompt_length}, target)
            del runner._opd_buffers[req_id]
        elif start + scheduled[req_id] >= len(ids) - 1:
            raise RuntimeError('vLLM skipped teacher response hidden states; OPD feature capture is incomplete.')


def _prompt_logprobs(self, hidden_states, scheduled):
    if self._opd_capture_rank == 0:
        capture_chunk(self, hidden_states, scheduled)
    return self._opd_original_prompt_logprobs(hidden_states, scheduled)


def _update_states(self, output):
    for req_id in output.finished_req_ids:
        self._opd_buffers.pop(req_id, None)
    return self._opd_original_update_states(output)


def install_on_worker(worker, cache_root):
    """vLLM collective RPC: every TP rank participates in the head gather once."""
    from vllm.distributed import get_tp_group

    runner = worker.model_runner
    root = Path(cache_root)
    if getattr(runner, '_opd_cache_root', None) == root:
        return
    if not hasattr(runner, '_get_prompt_logprobs_dict'):
        raise RuntimeError('OPD capture requires the vLLM 0.12 GPUModelRunner prompt-logprob interface.')
    group = get_tp_group()
    model = runner.model
    if not hasattr(model, 'lm_head') or not hasattr(model, 'config'):
        raise RuntimeError('OPD teacher feature capture requires a causal LM with lm_head and config.')
    head = model.lm_head
    if head.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError('Quantized output heads are not supported by OPD feature reconstruction.')
    weight = group.all_gather(head.weight.detach().contiguous(), dim=0)
    bias = getattr(head, 'bias', None)
    if bias is not None:
        bias = group.all_gather(bias.detach().contiguous(), dim=0)
    vocab_size = model.config.vocab_size
    if group.rank_in_group == 0:
        target = root / 'output_head.pt'
        if not target.exists():
            atomic_save({'weight': weight[:vocab_size].cpu(),
                         'bias': None if bias is None else bias[:vocab_size].cpu()}, target)
    runner._opd_cache_root = root
    runner._opd_capture_rank = group.rank_in_group
    runner._opd_buffers = {}
    runner._opd_original_prompt_logprobs = runner._get_prompt_logprobs_dict
    runner._opd_original_update_states = runner._update_states
    runner._get_prompt_logprobs_dict = types.MethodType(_prompt_logprobs, runner)
    runner._update_states = types.MethodType(_update_states, runner)
