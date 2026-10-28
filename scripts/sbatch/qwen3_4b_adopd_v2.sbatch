#!/usr/bin/env bash
#SBATCH --job-name=adopd-qwen3-4b-v2
#SBATCH --account=aip-plbacon
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --chdir=/scratch/l/luli/src/opd-baselines
#SBATCH --output=/scratch/l/luli/src/opd-baselines/tmp/slurm-%x-%j.log

set -euo pipefail
REPO_ROOT=/scratch/l/luli/src/opd-baselines
cd "$REPO_ROOT"
source scripts/h100_8gpu_env.sh
read -r CONFIG_NNODES CONFIG_GPUS_PER_NODE < <("$OPD_PYTHON" -c '
import yaml
with open("configs/qwen3_4b_adopd_v2.yaml") as stream:
    config = yaml.safe_load(stream)
values = [config[key] for key in ("h100_nnodes", "n_gpus_per_node")]
assert all(type(value) is int and value > 0 for value in values)
print(*values)
')
# Run with bash to have sbatch request exactly the resources in the YAML.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch --nodes="$CONFIG_NNODES" --ntasks="$CONFIG_NNODES" \
        --gpus-per-node="h100:$CONFIG_GPUS_PER_NODE" \
        "$REPO_ROOT/scripts/sbatch/qwen3_4b_adopd_v2.sbatch" "$@"
fi
if [[ "${SLURM_JOB_NUM_NODES:-0}" != "$CONFIG_NNODES" ]]; then
    echo "Allocated nodes (${SLURM_JOB_NUM_NODES:-0}) differ from h100_nnodes=$CONFIG_NNODES. Submit this script with bash to read the YAML allocation." >&2
    exit 1
fi
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" != "$CONFIG_GPUS_PER_NODE" ]]; then
    echo "Allocated GPUs/node ($SLURM_GPUS_ON_NODE) differ from n_gpus_per_node=$CONFIG_GPUS_PER_NODE." >&2
    exit 1
fi
export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"
export RAY_TMPDIR="/tmp/adopd-v2-${SLURM_JOB_ID}"
mapfile -t JOB_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#JOB_NODES[@]}" == "$CONFIG_NNODES" ]]
HEAD_IP="$(srun --mpi=none --nodes=1 --ntasks=1 --nodelist "${JOB_NODES[0]}" \
    --cpu-bind=none --gpu-bind=none --unbuffered \
    "$OPD_PYTHON" -c 'import ipaddress; from ray.util import get_node_ip_address; ip=ipaddress.ip_address(get_node_ip_address()); assert not ip.is_loopback and not ip.is_unspecified; print("RAY_HEAD_IP="+str(ip))' \
    | sed -n 's/^RAY_HEAD_IP=//p')"
[[ -n "$HEAD_IP" && "$HEAD_IP" != *$'\n'* ]]
export RAY_ADDRESS="${HEAD_IP}:$((20000 + SLURM_JOB_ID % 20000))"
srun --mpi=none --nodes="$CONFIG_NNODES" --ntasks="$CONFIG_NNODES" --ntasks-per-node=1 \
    --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 --unbuffered \
    "$OPD_RAY" symmetric-run --address "$RAY_ADDRESS" --min-nodes "$CONFIG_NNODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$CONFIG_GPUS_PER_NODE" --disable-usage-stats -- \
    "$OPD_PYTHON" scripts/run_baseline.py configs/qwen3_4b_adopd_v2.yaml \
    --auto-prepare-teacher-cache "$@"
