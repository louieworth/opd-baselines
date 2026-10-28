#!/usr/bin/env bash
#SBATCH --job-name=qwen3-1.7b-th-er-top-k-diff-t4b2507
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/opd-baselines
#SBATCH --output=/scratch/l/luli/src/opd-baselines/tmp/slurm-%x-%j.log

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with: sbatch scripts/sbatch/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff.sbatch" >&2
    exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != 1 ]]; then
    echo "This OPD job requires one node with eight H200 GPUs." >&2
    exit 1
fi

REPO_ROOT=/scratch/l/luli/src/opd-baselines
VERL_ENV=/scratch/l/luli/conda/envs/verl
source /home/l/luli/miniconda3/etc/profile.d/conda.sh
conda activate "$VERL_ENV"
cd "$REPO_ROOT"

# Read only the W&B entity from .env; keep API credentials out of the job shell.
if [[ -f "$REPO_ROOT/.env" ]]; then
    WANDB_ENTITY="$(sed -n 's/^WANDB_ENTITY=//p' "$REPO_ROOT/.env" | head -n 1)"
    WANDB_ENTITY="${WANDB_ENTITY#\"}"
    WANDB_ENTITY="${WANDB_ENTITY%\"}"
    export WANDB_ENTITY
fi

# Ray services must inherit these before the launcher runs.
export PYTHONPATH="$REPO_ROOT/third_party/verl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/l/luli/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$REPO_ROOT/data/.cache/huggingface/datasets"
export WANDB_MODE=offline
export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"
export WANDB_DIR="$REPO_ROOT"
export WANDB_DATA_DIR="$REPO_ROOT/wandb/staging"
export WANDB_CACHE_DIR="$REPO_ROOT/wandb/cache"
export WANDB_ARTIFACT_DIR="$REPO_ROOT/wandb/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export RAY_USAGE_STATS_ENABLED=0
# The installed DeepGEMM toolkit is CUDA 12.4 while verl/PyTorch is CUDA
# 12.8; disabling the optional dense-model kernel avoids CUDA_HOME-dependent
# warmup failures. These Qwen recipes do not require MoE DeepGEMM.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=300
export RAY_TMPDIR="/tmp/opd-ray-${SLURM_JOB_ID}"
# Let verl assign each actor's local rank from Ray's accelerator IDs.  Ray's
# automatic CUDA_VISIBLE_DEVICES rewrite can collapse fractional-GPU actors
# onto GPU 0 and make NCCL report duplicate devices.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export SLURM_EXPORT_ENV=ALL

# This single-node layout matches the YAML's eight-GPU batch calculation.
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from run_baseline import load_config

cfg = load_config(Path("configs/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff.yaml"))
if cfg["mode"] != "thinking":
    raise SystemExit("This job requires thinking mode for student training and evaluation.")
if cfg["teacher_model_path"] != "Qwen/Qwen3-4B-Instruct-2507" or not cfg["teacher_prompt"].strip():
    raise SystemExit("This job requires the fixed 4B-Instruct-2507 teacher with concise prompting.")
if cfg["opd_loss_token_mode"] != "top_k_diff":
    raise SystemExit("This script requires opd_loss_token_mode: top_k_diff.")
if cfg["method"] != "opd" or cfg["n_gpus_per_node"] != 8:
    raise SystemExit("This sbatch script expects the current eight-GPU OPD recipe.")
if cfg.get("teacher_enable_resource_pool", False):
    raise SystemExit("Eight total GPUs require the teacher to share the student resource pool.")
for key in ("vllm_tensor_parallel_size", "teacher_tp_size"):
    if 8 % int(cfg.get(key, 1)):
        raise SystemExit(f"{key} must divide the eight GPUs available on this node.")
PY

mapfile -t JOB_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#JOB_NODES[@]}" != 1 ]]; then
    echo "Could not resolve the single allocated Slurm node." >&2
    exit 1
fi
HEAD_NODE="${JOB_NODES[0]}"
# symmetric-run identifies the head by comparing its address with local NIC IPs.
# Cluster DNS can resolve a node name to an address absent from those NICs.
HEAD_IP="$(srun --mpi=none --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --nodelist "$HEAD_NODE" --cpu-bind=none --gpu-bind=none --unbuffered \
    python -c 'import ipaddress; from ray.util import get_node_ip_address; ip = ipaddress.ip_address(get_node_ip_address()); assert not ip.is_loopback and not ip.is_unspecified, f"Invalid Ray head IP: {ip}"; print(f"RAY_HEAD_IP={ip}")' \
    | sed -n 's/^RAY_HEAD_IP=//p')"
if [[ -z "$HEAD_IP" || "$HEAD_IP" == *$'\n'* ]]; then
    echo "Could not resolve a unique network IP on the Ray head node." >&2
    exit 1
fi
RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

printf 'Job %s: node=%s, H200=8/node, Ray=%s, environment=verl\n' \
    "$SLURM_JOB_ID" "${JOB_NODES[0]}" "$RAY_ADDRESS"

# symmetric-run starts Ray and Python on the allocated node. Additional
# arguments (e.g. --resume) go to the launcher.
srun --mpi=none --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    ray symmetric-run --address "$RAY_ADDRESS" --min-nodes 1 \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 8 \
    --disable-usage-stats -- \
    python scripts/run_baseline.py configs/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff.yaml \
    trainer.nnodes=1 trainer.n_gpus_per_node=8 \
    distillation.teacher_model.nnodes=1 \
    distillation.teacher_model.n_gpus_per_node=8 \
    "+ray_kwargs.ray_init.address=$RAY_ADDRESS" "$@"
