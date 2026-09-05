"""Exact per-question metrics for k sampled answers (no bootstrap resampling)."""

from collections import defaultdict
from math import isfinite


def compute_eval_metrics(data_sources, sample_uids, scores, k=12):
    if k <= 0 or not (len(data_sources) == len(sample_uids) == len(scores)):
        raise ValueError("Invalid k or misaligned evaluation results")
    groups = defaultdict(lambda: defaultdict(list))
    for source, uid, score in zip(data_sources, sample_uids, scores, strict=True):
        score = float(score)
        if not isfinite(score) or score not in (0.0, 1.0):
            raise ValueError(f"pass@{k} requires binary correctness, got {score}")
        groups[str(source)][str(uid)].append(score)
    if not groups:
        raise ValueError("Empty evaluation results")
    metrics = {}
    for source, questions in groups.items():
        for uid, values in questions.items():
            if len(values) != k:
                raise ValueError(f"{source}/{uid}: expected {k} samples, got {len(values)}")
        n = len(questions)
        metrics[f"eval/{source}/avg@{k}"] = sum(sum(v) / k for v in questions.values()) / n
        metrics[f"eval/{source}/pass@{k}"] = sum(max(v) for v in questions.values()) / n
        metrics[f"eval/{source}/num_questions"] = n
    return metrics
