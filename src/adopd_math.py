"""Sampled reverse KL and Eq. (5)/(6), independent of verl and GPU engines."""

from collections import defaultdict

import torch


def response_masks(responses, policy_mask, attention_mask, halt_token_id, reasoning_budget):
    """Split at the FIRST halt; a forced halt remains context, never a policy choice."""
    valid = attention_mask.bool()
    chosen = policy_mask.bool() & valid
    answer = torch.zeros_like(valid)
    trace = torch.zeros_like(valid)
    lengths = torch.zeros(len(responses), dtype=torch.long, device=responses.device)
    forced = torch.zeros(len(responses), dtype=torch.bool, device=responses.device)
    for i in range(len(responses)):
        stops = ((responses[i] == halt_token_id) & valid[i]).nonzero().flatten()
        if not len(stops):
            raise ValueError('ADOPD response has no </think>; use the budget agent loop')
        stop = int(stops[0])
        if stop > reasoning_budget:
            raise ValueError('ADOPD reasoning exceeded its hard budget')
        lengths[i] = stop  # response tokens before the delimiter, including a generated <think>
        trace[i, :stop + 1] = chosen[i, :stop + 1]
        answer[i, stop + 1:] = chosen[i, stop + 1:]
        forced[i] = not bool(chosen[i, stop])
    return answer, trace, lengths, forced


def masked_sequence_mean(values, mask):
    mask = mask.bool()
    return torch.where(mask, values, 0.0).sum(-1) / mask.sum(-1).clamp_min(1)


def sampled_distortion(student_log_probs, teacher_log_probs,
                       log_prob_min_clamp=None, loss_max_clamp=None):
    """Sampled k1 cost, with the same optional floor/clipping as Lightning OPD."""
    student, teacher = student_log_probs.float(), teacher_log_probs.detach().float()
    if log_prob_min_clamp is not None:
        student = student.clamp_min(log_prob_min_clamp)
        teacher = teacher.clamp_min(log_prob_min_clamp)
    cost = student - teacher
    return cost if loss_max_clamp is None else cost.clamp(-loss_max_clamp, loss_max_clamp)


def sampled_kl(current_log_probs, teacher_log_probs, old_log_probs,
               log_prob_min_clamp=None, loss_max_clamp=None):
    """Importance-sampled k1 reverse KL, optionally using Lightning OPD clipping.

    At the rollout policy its VALUE is log q - log p and its GRADIENT is
    (log q - log p) * grad(log q), the standard sampled-token OPD estimator.
    With clipping, use the detached clipped cost as the score-function weight;
    keep the importance ratio UNFLOORED so low-probability choices still train.
    With neither clamp, retain the exact-KL estimator's unit control variate.
    """
    current = current_log_probs.float()
    teacher, old = teacher_log_probs.detach().float(), old_log_probs.detach().float()
    ratio = (current - old).exp()
    if log_prob_min_clamp is not None or loss_max_clamp is not None:
        cost = sampled_distortion(current, teacher, log_prob_min_clamp, loss_max_clamp)
        return ratio * cost.detach()
    return ratio * (current - teacher - 1.0) + 1.0


@torch.no_grad()
def group_advantages(distortions, trace_lengths, prompt_uids, valid, group_size, budget, penalty):
    """Eq. (5) on complete prompt groups, BEFORE DP dispatch or mini-batching."""
    groups = defaultdict(list)
    for i, uid in enumerate(prompt_uids):
        groups[str(uid)].append(i)
    result = torch.zeros_like(distortions)
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError(f'Expected {group_size} rollouts per prompt, got {len(indices)}')
        ids = torch.tensor(indices, device=distortions.device)
        ids = ids[valid[ids]]
        if not len(ids):
            continue
        costs = distortions[ids].float() + penalty * trace_lengths[ids].float() / budget
        # Matches the PDF's torch.std() sketch. Handle zero/one valid rollout explicitly.
        std = costs.std(correction=1) if len(ids) > 1 else costs.new_zeros(())
        result[ids] = -(costs - costs.mean()) / (std + 1e-6)
    return result.detach()


def length_policy_gradient(log_probs, trace_mask, trace_lengths, budget, penalty):
    """Single-trajectory REINFORCE for E[penalty * length / budget].

    The sampled length is detached. SUM the log probabilities of chosen trace
    tokens, including a natural halt and excluding forced tokens/padding. A
    token mean would divide out the length signal. No group baseline is used.

    Keep the forward value equal to the actual length cost while retaining
    the score-function gradient: cost + pg - stop_gradient(pg).
    """
    cost = penalty * trace_lengths.detach().to(log_probs) / budget
    trace_log_prob = torch.where(trace_mask.bool(), log_probs, 0.0).sum(-1)
    pg = cost * trace_log_prob
    return cost + (pg - pg.detach())


def per_sequence_loss(current, teacher, old, answer_mask, trace_mask, advantages=None,
                      *, trace_lengths=None, budget=8192, length_penalty=0.0,
                      log_prob_min_clamp=None, loss_max_clamp=None, answer_kl=None):
    token_losses = (sampled_kl(current, teacher, old, log_prob_min_clamp, loss_max_clamp)
                    if answer_kl is None else answer_kl)
    distortion = masked_sequence_mean(token_losses, answer_mask)
    if advantages is None:
        if length_penalty == 0:
            length_objective = torch.zeros_like(distortion)
        else:
            if trace_lengths is None:
                raise ValueError('Single-rollout length policy gradient requires trace_lengths')
            length_objective = length_policy_gradient(
                current, trace_mask, trace_lengths, budget, length_penalty)
        return distortion + length_objective, distortion, length_objective
    # Group Eq. (5) already includes the length cost. Do not add it a second time.
    reinforce = -advantages.detach() * masked_sequence_mean(current, trace_mask)
    return distortion + reinforce, distortion, reinforce
