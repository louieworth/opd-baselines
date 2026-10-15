"""Use a separate prompt limit when creating validation datasets."""

from copy import deepcopy
from functools import wraps


def with_eval_prompt_limit(create_dataset):
    @wraps(create_dataset)
    def create(data_paths, data_config, tokenizer, processor, is_train=True, max_samples=-1):
        val_limit = data_config.get("val_max_prompt_length")
        if not is_train and val_limit is not None:
            # The shared training config must retain its original prompt limit.
            data_config = deepcopy(data_config)
            data_config["max_prompt_length"] = int(val_limit)
        return create_dataset(
            data_paths, data_config, tokenizer, processor,
            is_train=is_train, max_samples=max_samples,
        )

    return create
