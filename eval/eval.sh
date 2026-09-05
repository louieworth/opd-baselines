#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" train.py configs/qwen3_4b_grpo.yaml --eval-only "$@"
