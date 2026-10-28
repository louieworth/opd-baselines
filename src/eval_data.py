"""Use a separate prompt limit when creating validation datasets."""

from copy import deepcopy
from functools import wraps


def with_eval_prompt_limit(create_dataset):
    @wraps(create_dataset)
    def create(data_paths, data_config, tokenizer, processor, is_train=True, max_samples=-1):
        val_limit = data_config.get("val_max_prompt_length")
        if not is_train and (val_limit is not None or data_config.get("teacher_prompt_filter")):
            # The shared training config must retain its original prompt limit.
            data_config = deepcopy(data_config)
            if val_limit is not None:
                data_config["max_prompt_length"] = int(val_limit)
            if data_config.get("teacher_prompt_filter"):
                # Validation runs only the student, without teacher scoring.
                data_config["teacher_prompt_filter"] = None
        return create_dataset(
            data_paths, data_config, tokenizer, processor,
            is_train=is_train, max_samples=max_samples,
        )

    return create
