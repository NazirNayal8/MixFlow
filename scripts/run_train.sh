#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: bash scripts/run_train.sh <experiment> [hydra overrides...]

Examples:
  bash scripts/run_train.sh cifar10
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh cifar10
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_train.sh ffhq_64x64 trainer.num_nodes=1
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  show_usage
  exit 0
fi

if [ "$#" -lt 1 ]; then
  show_usage
  exit 1
fi

experiment="$1"
shift

python -m src.main \
  "+experiment=${experiment}" \
  hydra.job.name=train \
  "$@"
