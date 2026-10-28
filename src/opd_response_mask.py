"""Select OPD loss tokens while preserving the complete causal context."""

import torch


def response_loss_mask(response_mask, last_n_tokens=None, accuracy=None,
                       divergence_scores=None, top_k_divergence=None):
    """Select valid response positions; earlier tokens remain causal context."""
    if response_mask.ndim != 2:
        raise ValueError('response_mask must have shape [batch, response_length]')
    valid = response_mask.bool()
    selected = valid
    if last_n_tokens is not None:
        if type(last_n_tokens) is not int or last_n_tokens < 1:
            raise ValueError('last_n_tokens must be a positive integer or null')
        # Count from each response's end, ignoring right padding and masked tokens.
        remaining = valid.flip(-1).cumsum(-1).flip(-1)
        selected = selected & (remaining <= last_n_tokens)
    if accuracy is not None:
        accuracy = torch.as_tensor(accuracy, device=response_mask.device)
        if accuracy.shape != (response_mask.shape[0],):
            raise ValueError('verified filtering requires one acc value per trajectory')
        selected = selected & accuracy.eq(1).unsqueeze(-1)
    if top_k_divergence is not None:
        if type(top_k_divergence) is not int or top_k_divergence < 1:
            raise ValueError('top_k_divergence must be a positive integer or null')
        if divergence_scores is None or divergence_scores.shape != response_mask.shape:
            raise ValueError('divergence_scores must match response_mask shape')
        scores = divergence_scores.detach().to(device=response_mask.device, dtype=torch.float32)
        if not torch.isfinite(scores[selected]).all():
            raise ValueError('divergence scores must be finite at eligible response positions')
        if response_mask.shape[1]:
            scores = scores.masked_fill(~selected, -torch.inf)
            count = min(top_k_divergence, scores.shape[1])
            # argmax breaks top-1 ties at the earliest valid position.
            indices = (scores.argmax(-1, keepdim=True) if count == 1
                       else scores.argsort(dim=-1, descending=True, stable=True)[:, :count])
            top = torch.zeros_like(selected).scatter_(1, indices, True)
            selected = selected & top
    return selected.to(response_mask.dtype)


def sampled_token_divergence(batch):
    """Absolute, unclipped teacher/rollout-student log-prob gap (HDL Eq. 2 adapted to OPD).

    Rank once before the actor update, using old_log_probs rather than the
    changing optimization policy. Teacher row j predicts input token j+1.
    """
    data = batch.batch
    if 'old_log_probs' not in data or 'teacher_logprobs' not in data:
        raise ValueError('divergence selection requires old_log_probs and teacher_logprobs')
    student = data['old_log_probs']
    teacher = data['teacher_logprobs']
    mask = data['response_mask']
    prompt_width = data['prompts'].shape[1]
    if prompt_width < 1 or student.shape != mask.shape:
        raise ValueError('old_log_probs must align with response_mask and prompts must be nonempty')
    expected = (mask.shape[0], prompt_width + mask.shape[1], 1)
    if teacher.shape != expected:
        raise ValueError('divergence selection requires one sampled-token teacher log-prob per input position')
    teacher = teacher[:, prompt_width - 1:prompt_width - 1 + mask.shape[1], 0]
    return (teacher.detach().float() - student.detach().float()).abs()


def select_actor_loss(batch, opts):
    """Use a shallow batch copy so rollout lengths, rewards and eval stay intact."""
    from verl.protocol import DataProto

    accuracy = None
    if opts['opd_require_verified']:
        if 'acc' not in batch.non_tensor_batch:
            raise ValueError('opd_require_verified requires the reward verifier acc result')
        accuracy = batch.non_tensor_batch['acc']
    divergence_k = opts.get('opd_loss_top_k_divergence')
    scores = sampled_token_divergence(batch) if divergence_k is not None else None
    mask = response_loss_mask(batch.batch['response_mask'], opts['opd_loss_last_n_tokens'], accuracy,
                              scores, divergence_k)
    selected = mask.bool().any(-1)
    selected_count = int(selected.sum().item())
    metrics = {
        'actor/opd/selected_trajectories': selected_count,
        'actor/opd/filtered_trajectories': len(batch) - selected_count,
        'actor/opd/loss_tokens': int(mask.sum().item()),
        'actor/opd/update_skipped': int(selected_count == 0),
    }
    if scores is not None:
        values = scores[mask.bool()]
        metrics['actor/opd/selected_divergence_mean'] = values.mean().item() if values.numel() else 0.0
        metrics['actor/opd/selected_divergence_max'] = values.max().item() if values.numel() else 0.0
    actor_batch = DataProto(batch=batch.batch.clone(recurse=False),
                           non_tensor_batch=batch.non_tensor_batch.copy(),
                           meta_info=batch.meta_info.copy())
    actor_batch.batch['response_mask'] = mask
    return actor_batch, metrics
