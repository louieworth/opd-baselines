"""Reverse-KL objectives and exact vocabulary diagnostics, independent of Megatron."""

import torch


METRIC_NAMES = (
    "opd/teacher_student_topk_overlap_pct",
    "opd/teacher_difference_topk_overlap_pct",
    "opd/student_difference_topk_overlap_pct",
)


def sampled_advantage(teacher_log_probs, student_log_probs, minimum=-10.0, maximum=10.0):
    """Negative k1 reverse-KL estimate; clipping never changes diagnostic probabilities."""
    teacher, student = teacher_log_probs.float(), student_log_probs.float()
    if minimum is not None:
        teacher, student = teacher.clamp_min(minimum), student.clamp_min(minimum)
    advantage = teacher - student
    return advantage if maximum is None else advantage.clamp(-maximum, maximum)


def reverse_kl_loss(student_logits, teacher_logits, mode, top_k=32, maximum=10.0):
    """Per-position reverse KL. Top-k modes renormalize on the SAME selected support.

    Teacher logits are fixed targets. Full-vocab uses every real vocabulary entry.
    Float32 log-softmax avoids probability underflow; no logprob floor distorts these distributions.
    """
    student, teacher = student_logits.float(), teacher_logits.detach().float()
    if student.shape != teacher.shape:
        raise ValueError("Teacher and student vocabulary logits must have the same shape.")
    if not 1 <= top_k <= student.size(-1):
        raise ValueError("opd_token_top_k must not exceed the real vocabulary size.")
    if mode in ("top_k_student", "top_k_teacher", "top_k_diff"):
        if mode == "top_k_diff":
            selector = teacher.softmax(-1) - student.detach().softmax(-1)
        else:
            selector = student.detach() if mode == "top_k_student" else teacher
        ids = selector.topk(top_k, dim=-1).indices
        student, teacher = student.gather(-1, ids), teacher.gather(-1, ids)
    elif mode != "full_vocab":
        raise ValueError(f"Distribution reverse KL does not support mode {mode!r}.")
    log_student, log_teacher = student.log_softmax(-1), teacher.log_softmax(-1)
    losses = (log_student.exp() * (log_student - log_teacher)).sum(-1).clamp_min(0.0)
    return losses if maximum is None else losses.clamp_max(maximum)


def _overlap(left, right):
    """Intersection as a percentage of K; both sets have K unique token IDs."""
    return (left.unsqueeze(-1) == right.unsqueeze(-2)).any(-1).float().mean(-1) * 100.0


@torch.no_grad()
def distribution_metrics(student_logits, teacher_logits, top_k=32):
    """Three top-k overlaps; difference tokens maximize full-vocab p_teacher - p_student."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Teacher and student vocabulary logits must have the same shape.")
    if not 1 <= top_k <= student_logits.size(-1):
        raise ValueError("opd_token_top_k must not exceed the real vocabulary size.")
    student, teacher = student_logits.float().softmax(-1), teacher_logits.float().softmax(-1)
    teacher_ids = teacher_logits.topk(top_k, dim=-1).indices
    student_ids = student_logits.topk(top_k, dim=-1).indices
    difference_ids = (teacher - student).topk(top_k, dim=-1).indices
    return dict(zip(METRIC_NAMES, (
        _overlap(teacher_ids, student_ids),
        _overlap(teacher_ids, difference_ids),
        _overlap(student_ids, difference_ids),
    ), strict=True))
