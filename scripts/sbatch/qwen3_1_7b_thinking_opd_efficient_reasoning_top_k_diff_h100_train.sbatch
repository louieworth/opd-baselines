#!/usr/bin/env bash
#SBATCH --job-name=opd-top-k-diff-h100-train
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/opd-baselines
#SBATCH --output=/scratch/l/luli/src/opd-baselines/tmp/slurm-%x-%j.log

# Submit normally, or append --resume /absolute/path/to/a/completed-run-root.
# Fixed production recipe; eval at 20/40/60/80/100 and the final training step.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NUM_NODES:-0}" != 1 ]]; then
    echo "Submit with sbatch; this recipe requires one node with four H100 GPUs." >&2
    exit 1
fi

REPO_ROOT=/scratch/l/luli/src/opd-baselines
PYTHON=/scratch/l/luli/conda/envs/verl/bin/python
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/third_party/verl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HOME=/scratch/l/luli/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR="/tmp/opd-h100-train-${SLURM_JOB_ID}"
export OPD_H100_RECORD_DIR="$REPO_ROOT/outputs/h100-opd/runtime/slurm-${SLURM_JOB_ID}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export SLURM_EXPORT_ENV=ALL
unset RAY_ADDRESS

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
srun --mpi=none --nodes=1 --ntasks=1 --cpus-per-task=48 \
    --gpus-per-node=h100:4 --cpu-bind=none --gpu-bind=none \
    --kill-on-bad-exit=1 --unbuffered \
    "$PYTHON" scripts/run_h100_opd.py \
    configs/qwen3_1_7b_thinking_opd_efficient_reasoning_top_k_diff_h100_train.yaml "$@"
