"""Plain-text prompt encoding shared by dataset filtering and rollout."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ANSWER_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
CODE_ANSWER_INSTRUCTION = "Return a complete Python solution in a single ```python code block."
CHOICE_ANSWER_INSTRUCTION = (
    r"Please reason step by step, and put only the final answer letter (A, B, C, or D) within \boxed{}."
)


def dataset_prompt_config(config, *, is_train: bool):
    """Give validation its own prompt limit without changing training filtering."""
    if is_train or config.get("val_max_prompt_length") is None:
        return config
    result = deepcopy(config)
    result["max_prompt_length"] = config["val_max_prompt_length"]
    return result


def plaint_prompt(messages: list[dict[str, Any]]) -> str:
    """Keep the question/instructions, with one standard answer hint on a new line."""
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("plaint mode expects one user message")
    content = messages[0].get("content")
    if not isinstance(content, str):
        raise ValueError("plaint mode requires a text question")
    question = content.strip()
    if question.endswith((CODE_ANSWER_INSTRUCTION, CHOICE_ANSWER_INSTRUCTION)):
        return question
    if question.endswith(ANSWER_INSTRUCTION):
        question = question[: -len(ANSWER_INSTRUCTION)].rstrip()
    return f"{question}\n{ANSWER_INSTRUCTION}"


def plaint_token_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    # No chat template, assistant prefix, BOS, or EOS is added to the prompt.
    return tokenizer.encode(plaint_prompt(messages), add_special_tokens=False)


class PlaintDatasetMixin:
    """Filter with exactly the same prompt tokens that the rollout will consume."""

    def maybe_filter_out_long_prompts(self, dataframe):
        if self.processor is not None:
            raise ValueError("plaint mode supports text-only models")
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        max_length = self.max_prompt_length

        def within_limit(row):
            return len(plaint_token_ids(tokenizer, row[prompt_key])) <= max_length

        dataframe = dataframe.filter(
            within_limit,
            num_proc=self.num_workers,
            desc=f"Filtering plain prompts longer than {max_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe


class PlaintAgentMixin:
    """Replace the single-turn agent's input formatting with direct text encoding."""

    async def apply_chat_template(
        self, messages, tools=None, images=None, videos=None, remove_system_prompt=False,
    ) -> list[int]:
        # The method name is the verl extension point; it does not call a template.
        if self.processor is not None or tools or images or videos:
            raise ValueError("plaint mode supports text-only prompts without tools")
        return plaint_token_ids(self.tokenizer, messages)
