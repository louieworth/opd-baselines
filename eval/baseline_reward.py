"""Rule-based boxed-answer reward for math rollouts."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def extract_last_boxed(text: str | None) -> str | None:
    """Return the payload of the last balanced ``\\boxed{...}`` expression."""
    if not text:
        return None
    start = str(text).rfind("\\boxed{")
    if start < 0:
        return None
    cursor = start + len("\\boxed{")
    depth = 1
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return str(text)[start + len("\\boxed{") : cursor].strip()
        cursor += 1
    return None


def _simple_number(text: str) -> Decimal | None:
    value = str(text).strip().strip("$").replace(",", "").replace(" ", "")
    if value.startswith("\\(") and value.endswith("\\)"):
        value = value[2:-2]
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def answers_match(predicted: str, ground_truth: str) -> bool:
    pred_number = _simple_number(predicted)
    truth_number = _simple_number(ground_truth)
    if pred_number is not None and truth_number is not None:
        return pred_number == truth_number

    try:
        from math_verify import parse, verify

        pred_text = predicted if "$" in predicted else f"${predicted}$"
        truth_text = ground_truth if "$" in str(ground_truth) else f"${ground_truth}$"
        return bool(
            verify(
                parse(truth_text, fallback_mode="no_fallback"),
                parse(pred_text, fallback_mode="no_fallback"),
                timeout_seconds=5,
            )
        )
    except Exception:
        normalize = lambda value: str(value).replace("$", "").replace(" ", "").lower().strip()
        return normalize(predicted) == normalize(ground_truth)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """verl custom-reward entrypoint."""
    del extra_info, kwargs
    if data_source == "amobench":
        from eval.amo_reward import compute_score as amo_score

        score = amo_score(solution_str, ground_truth)
        return {"score": score, "acc": score, "formatted": float(extract_last_boxed(solution_str) is not None)}
    predicted = extract_last_boxed(solution_str)
    formatted = float(predicted is not None)
    correct = predicted is not None and answers_match(predicted, str(ground_truth))
    score = float(correct)
    return {"score": score, "acc": score, "formatted": formatted}
