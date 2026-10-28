#!/usr/bin/env bash
# Source before Ray starts so both nodes inherit the same environment.
OPD_VERL_ENV="${OPD_VERL_ENV:-/scratch/l/luli/conda/envs/verl}"
source /home/l/luli/miniconda3/etc/profile.d/conda.sh
conda activate "$OPD_VERL_ENV"
export OPD_PYTHON="$OPD_VERL_ENV/bin/python"
export OPD_RAY="$OPD_VERL_ENV/bin/ray"
export PYTHONPATH="$REPO_ROOT/third_party/verl:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HF_HOME=/scratch/l/luli/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$REPO_ROOT/data/.cache/huggingface/datasets"
export WANDB_MODE=offline
export WANDB_DIR="$REPO_ROOT"
export WANDB_DATA_DIR="$REPO_ROOT/wandb/staging"
export WANDB_CACHE_DIR="$REPO_ROOT/wandb/cache"
export WANDB_ARTIFACT_DIR="$REPO_ROOT/wandb/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
if [[ -f "$REPO_ROOT/.env" ]]; then
    WANDB_ENTITY="$(sed -n 's/^WANDB_ENTITY=//p' "$REPO_ROOT/.env" | head -n 1)"
    WANDB_ENTITY="${WANDB_ENTITY#\"}"
    WANDB_ENTITY="${WANDB_ENTITY%\"}"
    export WANDB_ENTITY
fi
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=300
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export SLURM_EXPORT_ENV=ALL
