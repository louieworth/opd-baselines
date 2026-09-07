"""Shared recipe fields, validation helpers, and native verl override encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROMPT_MODES = {"thinking", "non-thinking", "plaint"}
COMMON_KEYS = {
    "method",
    "model_path",
    "auto_prepare_model",
    "project_name",
    "experiment_name",
    "output_dir",
    "use_wandb",
    "wandb_entity",
    "wandb_api_key",
    "n_gpus_per_node",
    "vllm_tensor_parallel_size",
    "per_device_batch_size",
    "group_size",
    "ppo_mini_batch_size",
    "total_training_steps",
    "num_epochs",
    "learning_rate",
    "warmup_steps",
    "weight_decay",
    "max_grad_norm",
    "kl_loss_coef",
    "max_train_samples",
    "max_prompt_length",
    "max_completion_length",
    "actor_max_token_len_per_gpu",
    "temperature",
    "top_p",
    "top_k_sampling",
    "mode",
    "vllm_gpu_memory_utilization",
    "train_files",
    "val_files",
    "val_before_train",
    "eval_steps",
    "val_max_completion_length",
    "val_max_prompt_length",
    "val_do_sample",
    "val_n",
    "val_temperature",
    "val_top_p",
    "val_top_k",
    "val_batch_size",
    "save_steps",
}


def prompt_mode(cfg: dict[str, Any]) -> str:
    mode = cfg.get("mode", "non-thinking")
    if not isinstance(mode, str) or mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {sorted(PROMPT_MODES)}, got {mode!r}")
    return mode


def positive_int(cfg: dict[str, Any], key: str, default: int | None = None) -> int:
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a positive integer, got {raw!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value}")
    return value


def strict_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be YAML true/false, got {value!r}")
    return value


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def override(key: str, value: Any) -> str:
    return f"{key}={hydra_value(value)}"


def local_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def model_reference(value: str) -> str:
    if value.startswith(("./", "../", "/", "~")):
        return str(local_path(value))
    return value


def data_references(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("At least one data file is required")
    result = []
    for value in values:
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
            raise ValueError(f"Data paths must be repo-relative under data/: {value}")
        result.append(path.as_posix())
    return result


def effective_model(cfg: dict[str, Any]) -> str:
    source = str(cfg["model_path"])
    if strict_bool(cfg, "auto_prepare_model", True) and source == "Qwen/Qwen3-4B-Base":
        return "./models/Qwen3-4B-Base-chatml"
    return model_reference(source)
