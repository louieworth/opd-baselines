#!/usr/bin/env bash
# Environment for the long-lived two-node Ray cluster (2 x 8 H100) in the tiger pods.
# Source it on the head node before scripts/run_baseline.py; the launcher
# forwards the relevant variables to every Ray worker.
REPO_ROOT="${REPO_ROOT:-/opt/tiger/opd-baselines}"
export OPD_MODEL_ROOT="${OPD_MODEL_ROOT:-/opt/tiger/models}"

# The pod exports MODEL_PATH etc. for another job; the launcher reads MODEL_PATH and
# TEACHER_MODEL_PATH as overrides, so clear them (set them per command if needed).
unset MODEL_PATH TEACHER_MODEL_PATH

# Ray: attach to the running cluster and keep the driver on the head node.
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export OPD_DRIVER_RESOURCE=head
# As in h100_8gpu_env.sh: the ADOPD worker setup hook initializes CUDA before Ray would mask
# devices, so let verl select each worker's GPU by its Ray accelerator id instead.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Python: vendored verl, this repository, then repo-local wandb (the image only has byted-wandb).
export PYTHONPATH="$REPO_ROOT/third_party/verl:$REPO_ROOT:$REPO_ROOT/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN

# Models are local on both nodes; datasets are prepared parquet files.
export HF_HOME="$OPD_MODEL_ROOT/hf"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0

# Outbound HTTPS (online wandb) goes through the TLS-intercepting proxy from ~/.bashrc and its CA.
export WANDB_HTTPS_PROXY="${https_proxy:-}"
export SSL_CERT_FILE="$HOME/.claude/certs/bytedance-ca-bundle.pem"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
