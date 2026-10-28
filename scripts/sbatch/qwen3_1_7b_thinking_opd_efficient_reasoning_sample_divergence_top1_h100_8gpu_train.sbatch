#!/usr/bin/env bash
#SBATCH --job-name=opd-sample-divergence-top1-h100-8gpu
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/opd-baselines
#SBATCH --output=/scratch/l/luli/src/opd-baselines/tmp/slurm-%x-%j.log

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_NUM_NODES:-0}" != 2 ]]; then
    echo "Submit with sbatch: this recipe requires two nodes, four H100 GPUs each." >&2
    exit 1
fi
REPO_ROOT=/scratch/l/luli/src/opd-baselines
cd "$REPO_ROOT"
source scripts/h100_8gpu_env.sh
export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/opd-h100-8gpu-${SLURM_JOB_ID}"
export OPD_H100_RECORD_DIR="$REPO_ROOT/outputs/h100-8gpu-opd-divergence-top1/runtime/slurm-${SLURM_JOB_ID}"
mapfile -t JOB_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#JOB_NODES[@]}" == 2 ]]
HEAD_IP="$(srun --mpi=none --nodes=1 --ntasks=1 --nodelist "${JOB_NODES[0]}" \
    --cpu-bind=none --gpu-bind=none --unbuffered \
    "$OPD_PYTHON" -c 'import ipaddress; from ray.util import get_node_ip_address; ip=ipaddress.ip_address(get_node_ip_address()); assert not ip.is_loopback and not ip.is_unspecified; print("RAY_HEAD_IP="+str(ip))' \
    | sed -n 's/^RAY_HEAD_IP=//p')"
[[ -n "$HEAD_IP" && "$HEAD_IP" != *$'\n'* ]]
export RAY_ADDRESS="${HEAD_IP}:$((20000 + SLURM_JOB_ID % 20000))"
printf 'Job %s: nodes=%s, H100=4/node, total=8, Ray=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$RAY_ADDRESS"
srun --mpi=none --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    "$OPD_RAY" symmetric-run --address "$RAY_ADDRESS" --min-nodes 2 \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 4 --disable-usage-stats -- \
    "$OPD_PYTHON" scripts/run_h100_8gpu_opd.py \
    configs/qwen3_1_7b_thinking_opd_efficient_reasoning_sample_divergence_top1_h100_8gpu_train.yaml "$@"
