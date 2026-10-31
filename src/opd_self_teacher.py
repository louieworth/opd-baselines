"""Self-teacher for CRISP-style on-policy self-distillation (arXiv 2603.05433).

The reference model inside the actor worker is the teacher: a copy of the student that scores the
student's own response after the teacher prompt. Every `opd_teacher_refresh_steps` actor updates its
parameters, and the cached full output head used for the full-vocabulary KL, become the student's.
"""

import torch

_STATE = {}  # actor and teacher modules, the teacher's head (CPU and per-update device copy), update count


def refresh_due(update, every):
    """True before 1-based actor update `update` when the teacher becomes the student after `every` updates."""
    return every > 0 and update > 1 and (update - 1) % every == 0


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, 'to_local') else tensor


def copy_parameters(source, target):
    """Overwrite the teacher's (local shards of) parameters with the student's."""
    pairs = list(zip(source.named_parameters(), target.named_parameters(), strict=True))
    for (name, s), (other, t) in pairs:
        if name != other or _local(s).shape != _local(t).shape:
            raise ValueError(f'Teacher parameters do not match the student: {name} {tuple(_local(s).shape)} '
                             f'vs {other} {tuple(_local(t).shape)}')
    with torch.no_grad():
        for (_, s), (_, t) in pairs:
            _local(t).copy_(_local(s))


def gather_head(module, device, dtype=torch.bfloat16):
    """The full output layer (weight, bias); sharded parameters are gathered (a collective on all ranks)."""
    head = module.get_output_embeddings()
    weight, bias = head.weight, getattr(head, 'bias', None)
    full = lambda value: (value.full_tensor() if hasattr(value, 'full_tensor') else value).detach()
    return (full(weight).to(device=device, dtype=dtype),
            None if bias is None else full(bias).to(device=device, dtype=dtype))


def response_hidden(module, input_ids, prompt_length, response_length, device):
    """Final-norm hidden states at the positions that predict each response token (no gradients).

    `device` is where the forward runs; FSDP2 brings CPU-offloaded teacher parameters there itself.
    """
    captured = []
    handle = module.get_decoder().norm.register_forward_hook(lambda _m, _i, output: captured.append(output))
    try:
        ids = torch.as_tensor(list(input_ids), device=device).unsqueeze(0)
        with torch.no_grad():
            module(input_ids=ids, position_ids=torch.arange(ids.size(1), device=device).unsqueeze(0),
                   use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    return captured[-1][0, prompt_length - 1:prompt_length - 1 + response_length].detach()


def register(actor, teacher):
    """Called on every actor rank after the models are built; the teacher starts as the initial student."""
    _STATE.update(actor=actor, teacher=teacher, updates=0,
                  head_cpu=gather_head(actor, 'cpu'))
    _STATE.pop('head', None)


def teacher_module():
    if 'teacher' not in _STATE:
        raise RuntimeError('The self-teacher is not registered in this actor worker')
    return _STATE['teacher']


def before_update(every):
    """Count actor updates and refresh the teacher when due (all ranks call this before each update)."""
    _STATE['updates'] += 1
    if refresh_due(_STATE['updates'], every):
        copy_parameters(_STATE['actor'], _STATE['teacher'])
        _STATE['head_cpu'] = gather_head(_STATE['actor'], 'cpu')
        return True
    return False


def output_head(device):
    """The teacher's head on this device for the current update (released by release_head)."""
    if 'head' not in _STATE:
        weight, bias = _STATE['head_cpu']
        _STATE['head'] = (weight.to(device), None if bias is None else bias.to(device))
    return _STATE['head']


def release_head():
    _STATE.pop('head', None)
