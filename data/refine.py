"""Build the math rewrite prompt used by the TRD trajectory stage."""

from __future__ import annotations

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


def refine_token_ids(tokenizer, messages, initial_response: str, mode: str, max_prompt_length: int) -> list[int]:
    prompt = refine_prompt(messages, initial_response)
    if mode == "plaint":
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    elif mode in {"thinking", "non-thinking"}:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=mode == "thinking",
        )
    else:
        raise ValueError(f"Unsupported TRD prompt mode: {mode!r}")
    if len(ids) > max_prompt_length:
        raise ValueError(
            f"TRD rewrite prompt has {len(ids)} tokens, exceeding refine_max_prompt_length="
            f"{max_prompt_length}; increase that limit to retain the whole question and y_o"
        )
    return list(ids)
