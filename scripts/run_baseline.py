#!/usr/bin/env python3
"""Launch the packaged GRPO or plain OPD baseline through native verl."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
METHODS = {"grpo", "opd"}
PROMPT_MODES = {"thinking", "non-thinking", "plaint"}
ALLOWED_KEYS = {
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
    "val_do_sample",
    "val_n",
    "val_temperature",
    "val_top_p",
    "val_top_k",
    "val_batch_size",
    "save_steps",
    "teacher_model_path",
    "teacher_enable_resource_pool",
    "teacher_n_gpus_per_node",
    "teacher_tp_size",
    "teacher_gpu_memory_utilization",
    "distillation_num_workers",
    "distillation_loss_mode",
    "distillation_topk",
    "distillation_use_policy_gradient",
    "distillation_use_task_rewards",
    "distillation_loss_coef",
    "distillation_loss_max_clamp",
    "distillation_log_prob_min_clamp",
}
OPD_ONLY_KEYS = {
    "teacher_model_path",
    "teacher_enable_resource_pool",
    "teacher_n_gpus_per_node",
    "teacher_tp_size",
    "teacher_gpu_memory_utilization",
    "distillation_num_workers",
    "distillation_loss_mode",
    "distillation_topk",
    "distillation_use_policy_gradient",
    "distillation_use_task_rewards",
    "distillation_loss_coef",
    "distillation_loss_max_clamp",
    "distillation_log_prob_min_clamp",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    unknown = sorted(set(payload) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"{path}: unsupported keys: {unknown}")
    entity = payload.get("wandb_entity")
    if entity is not None:
        if not isinstance(entity, str):
            raise ValueError(f"{path}: wandb_entity must be a string or null")
        payload["wandb_entity"] = entity.strip() or None
        if entity.strip().startswith("wandb_v1_"):
            raise ValueError(f"{path}: put the API key in wandb_api_key; wandb_entity is an account/team name")
    api_key = payload.get("wandb_api_key")
    if api_key is not None:
        if not isinstance(api_key, str):
            raise ValueError(f"{path}: wandb_api_key must be a string or null")
        payload["wandb_api_key"] = api_key.strip() or None
    method = str(payload.get("method", "")).lower()
    if method not in METHODS:
        raise ValueError(f"{path}: method must be one of {sorted(METHODS)}, got {method!r}")
    if method == "grpo":
        stray = sorted(OPD_ONLY_KEYS.intersection(payload))
        if stray:
            raise ValueError(f"{path}: GRPO config contains OPD-only keys: {stray}")
    elif not payload.get("teacher_model_path"):
        raise ValueError(f"{path}: OPD requires teacher_model_path")
    payload["mode"] = prompt_mode(payload)
    return payload


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


def build_command(cfg: dict[str, Any], run_dir: Path, extra: list[str]) -> list[str]:
    method = str(cfg["method"]).lower()
    mode = prompt_mode(cfg)
    n_gpus = positive_int(cfg, "n_gpus_per_node", 8)
    tp_size = positive_int(cfg, "vllm_tensor_parallel_size", 1)
    if n_gpus % tp_size:
        raise ValueError(f"n_gpus_per_node={n_gpus} is not divisible by tensor parallel size={tp_size}")

    per_device = positive_int(cfg, "per_device_batch_size")
    group_size = positive_int(cfg, "group_size")
    rollout_batch = per_device * (n_gpus // tp_size)
    rollout_samples = rollout_batch * group_size
    mini_batch = positive_int(cfg, "ppo_mini_batch_size", rollout_batch)
    if rollout_samples % mini_batch:
        raise ValueError(
            f"rollout samples ({rollout_samples}) must be divisible by ppo_mini_batch_size ({mini_batch})"
        )

    max_prompt = positive_int(cfg, "max_prompt_length", 2048)
    max_response = positive_int(cfg, "max_completion_length", 8192)
    val_max_response = positive_int(cfg, "val_max_completion_length", max_response)
    if val_max_response != max_response:
        raise ValueError("This native verl rollout requires equal train and validation response lengths")
    train_max_seq = max_prompt + max_response
    rollout_max_len = max(train_max_seq, max_prompt + val_max_response)
    actor_token_budget = positive_int(cfg, "actor_max_token_len_per_gpu", train_max_seq)

    train_files = data_references(cfg["train_files"])
    val_files = data_references(cfg["val_files"])
    model_path = effective_model(cfg)
    reward_path = "src/reward_async.py"
    kl_coef = float(cfg.get("kl_loss_coef", 0.0))
    use_kl = kl_coef != 0.0
    use_wandb = strict_bool(cfg, "use_wandb", True)
    logger = ["console", "wandb"] if use_wandb else ["console"]

    command = [
        sys.executable,
        "-m",
        "src.train",
        override("hydra.job.chdir", False),
        override("hydra.run.dir", run_dir / "hydra"),
        override("algorithm.adv_estimator", "grpo"),
        override("algorithm.use_kl_in_reward", False),
        override("trainer.use_legacy_worker_impl", "disable"),
        override("trainer.critic_warmup", 0),
        override("trainer.logger", logger),
        override("trainer.project_name", cfg.get("project_name", "opd-grpo-baselines")),
        override("trainer.experiment_name", cfg.get("experiment_name", method)),
        override("trainer.default_local_dir", run_dir),
        override("trainer.n_gpus_per_node", n_gpus),
        override("trainer.nnodes", 1),
        override("trainer.save_freq", int(cfg.get("save_steps", 25))),
        override("trainer.test_freq", int(cfg.get("eval_steps", -1))),
        override("trainer.total_epochs", positive_int(cfg, "num_epochs", 1)),
        override("trainer.total_training_steps", positive_int(cfg, "total_training_steps")),
        override("trainer.val_before_train", strict_bool(cfg, "val_before_train", False)),
        override("trainer.validation_data_dir", run_dir / "validation"),
        override("data.train_files", train_files),
        override("data.val_files", val_files),
        override("+data.cache_dir", "data/.cache/verl"),
        override("data.train_batch_size", rollout_batch),
        override("data.val_batch_size", positive_int(cfg, "val_batch_size", 16)),
        override("data.train_max_samples", positive_int(cfg, "max_train_samples", 100000)),
        override("data.max_prompt_length", max_prompt),
        override("data.max_response_length", max_response),
        override("+data.prompt_mode", mode),
        override("data.filter_overlong_prompts", True),
        override("data.truncation", "error"),
        override("data.shuffle", False),
        override("actor_rollout_ref.model.path", model_path),
        override("actor_rollout_ref.model.use_remove_padding", True),
        override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        override("actor_rollout_ref.model.lora.merge", False),
        override("actor_rollout_ref.model.lora_rank", 0),
        override("actor_rollout_ref.actor.optim.lr", float(cfg.get("learning_rate", 1.0e-6))),
        override("actor_rollout_ref.actor.optim.weight_decay", float(cfg.get("weight_decay", 0.01))),
        override("actor_rollout_ref.actor.optim.lr_warmup_steps", int(cfg.get("warmup_steps", 0))),
        override("actor_rollout_ref.actor.grad_clip", float(cfg.get("max_grad_norm", 1.0))),
        override("actor_rollout_ref.actor.ppo_mini_batch_size", mini_batch),
        override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 4),
        override("actor_rollout_ref.actor.use_dynamic_bsz", True),
        override("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", actor_token_budget),
        override("actor_rollout_ref.actor.use_kl_loss", use_kl),
        override("actor_rollout_ref.actor.kl_loss_coef", kl_coef),
        override("actor_rollout_ref.actor.kl_loss_type", "low_var_kl"),
        override("actor_rollout_ref.actor.entropy_coeff", 0),
        override("actor_rollout_ref.actor.strategy", "fsdp2"),
        override("actor_rollout_ref.actor.fsdp_config.model_dtype", "bf16"),
        override("actor_rollout_ref.actor.fsdp_config.param_offload", False),
        override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", False),
        override("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"),
        override("actor_rollout_ref.rollout.temperature", float(cfg.get("temperature", 1.0))),
        override("actor_rollout_ref.rollout.top_p", float(cfg.get("top_p", 1.0))),
        override("actor_rollout_ref.rollout.top_k", int(cfg.get("top_k_sampling", -1))),
        override("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True),
        override("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", rollout_max_len),
        override("actor_rollout_ref.rollout.tensor_model_parallel_size", tp_size),
        override("actor_rollout_ref.rollout.name", "vllm"),
        override("actor_rollout_ref.rollout.gpu_memory_utilization", float(cfg.get("vllm_gpu_memory_utilization", 0.6))),
        override("actor_rollout_ref.rollout.n", group_size),
        override("actor_rollout_ref.rollout.agent.num_workers", n_gpus),
        override("actor_rollout_ref.rollout.load_format", "safetensors"),
        override("actor_rollout_ref.rollout.layered_summon", True),
        override("actor_rollout_ref.rollout.max_model_len", rollout_max_len),
        override("actor_rollout_ref.rollout.max_num_batched_tokens", max_prompt + val_max_response),
        override("actor_rollout_ref.rollout.val_kwargs.n", positive_int(cfg, "val_n", 1)),
        override("actor_rollout_ref.rollout.val_kwargs.do_sample", strict_bool(cfg, "val_do_sample", True)),
        override("actor_rollout_ref.rollout.val_kwargs.temperature", float(cfg.get("val_temperature", 0.7))),
        override("actor_rollout_ref.rollout.val_kwargs.top_p", float(cfg.get("val_top_p", 0.8))),
        override("actor_rollout_ref.rollout.val_kwargs.top_k", int(cfg.get("val_top_k", 20))),
        override("actor_rollout_ref.ref.log_prob_use_dynamic_bsz", True),
        override("actor_rollout_ref.ref.log_prob_max_token_len_per_gpu", rollout_max_len),
        override("actor_rollout_ref.ref.fsdp_config.param_offload", True),
        override("actor_rollout_ref.ref.strategy", "fsdp2"),
        override("actor_rollout_ref.ref.fsdp_config.model_dtype", "bf16"),
        override("reward.custom_reward_function.path", reward_path),
        override("reward.custom_reward_function.name", "compute_score"),
    ]

    if mode == "plaint":
        command.extend(
            [
                override("data.custom_cls.path", "pkg://src.plaint_runtime"),
                override("data.custom_cls.name", "PlaintDataset"),
                override("actor_rollout_ref.rollout.agent.default_agent_loop", "plaint_agent"),
                override("actor_rollout_ref.rollout.agent.agent_loop_config_path", "configs/plaint_agent.yaml"),
            ]
        )
    else:
        command.append(override("+data.apply_chat_template_kwargs.enable_thinking", mode == "thinking"))

    if method == "opd":
        command.extend(
            [
                override("distillation.enabled", True),
                override("distillation.num_workers", positive_int(cfg, "distillation_num_workers", 8)),
                override("distillation.teacher_model.enable_resource_pool", strict_bool(cfg, "teacher_enable_resource_pool", False)),
                override("distillation.teacher_model.n_gpus_per_node", int(cfg.get("teacher_n_gpus_per_node", 0))),
                override("distillation.teacher_model.nnodes", 1),
                override("distillation.teacher_model.model_path", model_reference(str(cfg["teacher_model_path"]))),
                override("distillation.teacher_model.inference.tensor_model_parallel_size", positive_int(cfg, "teacher_tp_size", 1)),
                override("distillation.teacher_model.inference.name", "vllm"),
                override("distillation.teacher_model.inference.gpu_memory_utilization", float(cfg.get("teacher_gpu_memory_utilization", 0.3))),
                override("distillation.teacher_model.inference.max_model_len", train_max_seq + 1),
                override("distillation.teacher_model.inference.max_num_batched_tokens", train_max_seq + 1),
                override("distillation.distillation_loss.loss_mode", cfg.get("distillation_loss_mode", "k1")),
                override("distillation.distillation_loss.topk", positive_int(cfg, "distillation_topk", 64)),
                override("distillation.distillation_loss.use_task_rewards", strict_bool(cfg, "distillation_use_task_rewards", False)),
                override("distillation.distillation_loss.use_policy_gradient", strict_bool(cfg, "distillation_use_policy_gradient", True)),
                override("distillation.distillation_loss.policy_loss_mode", "vanilla"),
                override("distillation.distillation_loss.distillation_loss_coef", float(cfg.get("distillation_loss_coef", 1.0))),
                override("distillation.distillation_loss.loss_max_clamp", float(cfg.get("distillation_loss_max_clamp", 10.0))),
                override("distillation.distillation_loss.log_prob_min_clamp", float(cfg.get("distillation_log_prob_min_clamp", -10.0))),
            ]
        )
    else:
        command.append(override("distillation.enabled", False))

    for item in extra:
        key, _, value = item.partition("=")
        if key.lstrip("+") in {"data.train_files", "data.val_files", "data.cache_dir"}:
            data_references(yaml.safe_load(value))
    command.extend(extra)
    return command


def preflight(cfg: dict[str, Any]) -> None:
    model = str(cfg["model_path"])
    if model.startswith(("./", "../", "/", "~")) and not local_path(model).is_dir():
        raise FileNotFoundError(f"model_path does not exist: {local_path(model)}")
    try:
        __import__("flash_attn")
    except ImportError as exc:
        raise RuntimeError("flash-attn is required; refusing to fall back to another attention backend") from exc
    __import__("math_verify")


def prepare_model(cfg: dict[str, Any]) -> None:
    if effective_model(cfg) == "./models/Qwen3-4B-Base-chatml" and cfg["model_path"] == "Qwen/Qwen3-4B-Base":
        if __package__:
            from .prepare_qwen3_base import ensure_prepared
        else:
            from prepare_qwen3_base import ensure_prepared
        ensure_prepared(ROOT / "models/Qwen3-4B-Base-chatml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the Hydra command without launching")
    parser.add_argument("--eval-only", action="store_true", help="evaluate the student without training or a teacher")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any("=" not in item for item in extra):
        parser.error("extra arguments after -- must be Hydra key=value overrides")

    config_path = args.config.resolve() if args.config.exists() else ROOT / args.config
    cfg = load_config(config_path)
    # Credentials never enter the saved recipe or the Hydra command/config.
    wandb_api_key = cfg.pop("wandb_api_key", None)
    wandb_entity = cfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
    cfg["model_path"] = os.environ.get("MODEL_PATH", str(cfg["model_path"]))
    if cfg["method"] == "opd" and os.environ.get("TEACHER_MODEL_PATH"):
        cfg["teacher_model_path"] = os.environ["TEACHER_MODEL_PATH"]
    if args.eval_only:
        cfg = {key: value for key, value in cfg.items() if key not in OPD_ONLY_KEYS}
        cfg["method"] = "grpo"
        cfg["experiment_name"] += "_eval"
        extra += ["trainer.val_before_train=true", "trainer.val_only=true"]
    output_root = Path(str(cfg.get("output_dir", f"./outputs/{cfg['method']}")))
    if output_root.is_absolute() or ".." in output_root.parts:
        raise ValueError("output_dir must be repo-relative")
    stamp = "DRY_RUN" if args.dry_run else datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S")
    run_dir = output_root / stamp
    command = build_command(cfg, run_dir, extra)

    summary = {
        "method": cfg["method"],
        "mode": cfg["mode"],
        "config": str(config_path),
        "run_dir": str(run_dir),
        "distillation_enabled": cfg["method"] == "opd",
        "teacher_model": cfg.get("teacher_model_path"),
        "policy_loss": "vanilla",
        "wandb_enabled": cfg.get("use_wandb", True),
        "wandb_entity": wandb_entity,
        "wandb_mode": os.environ.get("WANDB_MODE", "online"),
        "eval_only": args.eval_only,
    }
    print(json.dumps(summary, indent=2))
    print(shlex.join(command))
    if args.dry_run:
        return 0

    os.chdir(ROOT)
    preflight(cfg)
    prepare_model(cfg)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "launch_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    env = os.environ.copy()
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        # The input YAML contains a credential, so do not upload source/diffs.
        env["WANDB_DISABLE_CODE"] = "true"
        env["WANDB_DISABLE_GIT"] = "true"
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "third_party" / "verl"), str(ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["HYDRA_FULL_ERROR"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_DATASETS_CACHE"] = "data/.cache/huggingface/datasets"
    env["WANDB_DIR"] = str(run_dir)
    env["WANDB_DATA_DIR"] = str(run_dir / "wandb_staging")
    env["WANDB_CACHE_DIR"] = str(output_root / "wandb_cache")
    env["WANDB_ARTIFACT_DIR"] = str(run_dir / "wandb_artifacts")
    for name in ("WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_ARTIFACT_DIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)

    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
