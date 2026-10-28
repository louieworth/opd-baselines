"""Exact per-question metrics for k sampled answers (no bootstrap resampling)."""

from collections import defaultdict
from math import isfinite


def compute_eval_metrics(data_sources, sample_uids, scores, k=12, response_lengths=None):
    if k <= 0 or not (len(data_sources) == len(sample_uids) == len(scores)):
        raise ValueError("Invalid k or misaligned evaluation results")
    groups = defaultdict(lambda: defaultdict(list))
    src_lengths = defaultdict(list)
    for i, (source, uid, score) in enumerate(zip(data_sources, sample_uids, scores, strict=True)):
        score = float(score)
        if not isfinite(score) or score not in (0.0, 1.0):
            raise ValueError(f"pass@{k} requires binary correctness, got {score}")
        groups[str(source)][str(uid)].append(score)
        if response_lengths is not None:
            src_lengths[str(source)].append(response_lengths[i])
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
        if response_lengths is not None:
            lengths = src_lengths[source]
            metrics[f"eval/{source}/response-length"] = sum(lengths) / len(lengths)
    return metrics
