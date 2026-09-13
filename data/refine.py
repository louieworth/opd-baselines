"""Build the math rewrite prompt used by the TRD trajectory stage."""

from __future__ import annotations

from copy import copy
from uuid import uuid4

from data.prompt_modes import ANSWER_INSTRUCTION


# Adapted from recipe/opd/generation/y_r_prepare.py (PROMPT_TEMPLATE_OPD_REFINE).
REFINE_TEMPLATE = """Your task is to rewrite your mathematical solution.

**Problem:**
{problem}

**Your Initial Solution:**
{initial_response}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution"""


def refine_prompt(messages, initial_response: str) -> str:
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("TRD refinement expects one user question")
    problem = messages[0].get("content")
    if not isinstance(problem, str):
        raise ValueError("TRD refinement requires a text question")
    problem = problem.strip()
    if problem.endswith(ANSWER_INSTRUCTION):
        problem = problem[: -len(ANSWER_INSTRUCTION)].rstrip()
    return REFINE_TEMPLATE.format(
        problem=problem, initial_response=initial_response.strip(),
    ) + "\n\n" + ANSWER_INSTRUCTION


def refine_token_ids_batch(
    tokenizer, messages_batch, initial_responses: list[str], mode: str, max_prompt_length: int, *, device=None,
) -> list[list[int]]:
    """Batch-tokenize and clip answers while retaining rewrite instructions."""
    import torch

    marker = f"TRD_RESPONSE_{uuid4().hex}"
    prompts = [refine_prompt(messages, marker) for messages in messages_batch]
    if mode == "plaint":
        rendered = prompts
    elif mode in {"thinking", "non-thinking"}:
        rendered = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in prompts], tokenize=False,
            add_generation_prompt=True, enable_thinking=mode == "thinking",
        )
    else:
        raise ValueError(f"Unsupported TRD prompt mode: {mode!r}")

    prefixes, suffixes = zip(*(text.split(marker) for text in rendered), strict=True)
    batch_tokenizer = copy(tokenizer)
    batch_tokenizer.padding_side = batch_tokenizer.truncation_side = "right"
    encoded = batch_tokenizer(
        [*prefixes, *(text.strip() for text in initial_responses), *suffixes],
        add_special_tokens=False, padding=True, truncation=True,
        max_length=max_prompt_length, return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device=device)
    lengths = encoded["attention_mask"].to(device=device).sum(dim=1)
    prefix_ids, response_ids, suffix_ids = input_ids.chunk(3)
    prefix_lengths, response_lengths, suffix_lengths = lengths.chunk(3)

    # Reserve the suffix, then the question, and give y_o the remaining space.
    # All row budgets and gather indices are computed as batched tensor operations.
    prefix_lengths = torch.minimum(prefix_lengths, max_prompt_length - suffix_lengths)
    response_lengths = torch.minimum(response_lengths, max_prompt_length - prefix_lengths - suffix_lengths)
    total_lengths = prefix_lengths + response_lengths + suffix_lengths
    width = input_ids.shape[1]
    positions = torch.arange(min(max_prompt_length, 3 * width), device=input_ids.device)[None, :]
    response_start = prefix_lengths[:, None]
    suffix_start = (prefix_lengths + response_lengths)[:, None]
    indices = torch.where(
        positions < response_start, positions,
        torch.where(positions < suffix_start, width + positions - response_start,
                    2 * width + positions - suffix_start),
    )
    packed = torch.cat([prefix_ids, response_ids, suffix_ids], dim=1)
    clipped = packed.gather(1, indices.clamp(max=3 * width - 1))
    # Remove padding only when serializing the variable-length vLLM requests.
    return [row[:length] for row, length in zip(clipped.tolist(), total_lengths.tolist(), strict=True)]


def refine_token_ids(tokenizer, messages, initial_response: str, mode: str, max_prompt_length: int) -> list[int]:
    return refine_token_ids_batch(tokenizer, [messages], [initial_response], mode, max_prompt_length)[0]
