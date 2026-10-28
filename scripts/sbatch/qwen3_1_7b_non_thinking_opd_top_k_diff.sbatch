#!/usr/bin/env bash
#SBATCH --job-name=qwen3-1.7b-nt-opd-top-k-diff-t4b2507
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=08:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/opd-baselines
#SBATCH --output=/scratch/l/luli/src/opd-baselines/tmp/slurm-%x-%j.log

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with: sbatch scripts/sbatch/qwen3_1_7b_non_thinking_opd_top_k_diff.sbatch" >&2
    exit 1
fi
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != 2 ]]; then
    echo "This OPD job requires two nodes with four H100 GPUs each." >&2
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

# Ray services on both nodes must inherit these before the launcher runs.
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

# The YAML's eight-GPU batch calculation stays intact. Only the physical
# resource layout is overridden below, from 1 x 8 GPUs to 2 x 4 GPUs.
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from run_baseline import load_config

cfg = load_config(Path("configs/qwen3_1_7b_non_thinking_opd.yaml"))
if cfg["mode"] != "non-thinking":
    raise SystemExit("This job requires the non-thinking chat template for training and evaluation.")
if cfg["method"] != "opd" or cfg["n_gpus_per_node"] != 8:
    raise SystemExit("This sbatch script expects the current eight-GPU OPD recipe.")
if cfg.get("teacher_enable_resource_pool", False):
    raise SystemExit("Eight total GPUs require the teacher to share the student resource pool.")
for key in ("vllm_tensor_parallel_size", "teacher_tp_size"):
    if 4 % int(cfg.get(key, 1)):
        raise SystemExit(f"{key} must divide the four GPUs available on each node.")
PY

mapfile -t JOB_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#JOB_NODES[@]}" != 2 ]]; then
    echo "Could not resolve the two allocated Slurm nodes." >&2
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

printf 'Job %s: nodes=%s,%s, H100=4/node, Ray=%s, environment=verl\n' \
    "$SLURM_JOB_ID" "${JOB_NODES[0]}" "${JOB_NODES[1]}" "$RAY_ADDRESS"

# symmetric-run starts Ray on both nodes, waits for them, and runs Python
# only on the head node. Additional arguments (e.g. --resume) go to the launcher.
# Its Ray start options are passed to both head and worker, so keep head-only
# options such as the dashboard toggle out of this command.
srun --mpi=none --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    ray symmetric-run --address "$RAY_ADDRESS" --min-nodes 2 \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 4 \
    --disable-usage-stats -- \
    python scripts/run_baseline.py configs/qwen3_1_7b_non_thinking_opd.yaml \
    trainer.nnodes=2 trainer.n_gpus_per_node=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    distillation.teacher_model.nnodes=2 \
    distillation.teacher_model.n_gpus_per_node=4 \
    "+ray_kwargs.ray_init.address=$RAY_ADDRESS" "$@" \
    opd_loss_token_mode=top_k_diff \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
