#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: bash scripts/run_inference.sh <experiment> <checkpoint> <output_dir> [hydra overrides...]

Examples:
  bash scripts/run_inference.sh cifar10 /path/to/model.ckpt outputs/inference/cifar10
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_inference.sh cifar10 /path/to/model.ckpt outputs/inference/cifar10
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_inference.sh ffhq_64x64 /path/to/model.ckpt outputs/inference/ffhq test.num_samples=10000
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  show_usage
  exit 0
fi

if [ "$#" -lt 3 ]; then
  show_usage
  exit 1
fi

experiment="$1"
checkpoint="$2"
output_dir="$3"
shift 3

mkdir -p "${output_dir}"

python -m src.main \
  "+experiment=${experiment}" \
  checkpointing.load="${checkpoint}" \
  mode=test \
  test.output_path="${output_dir}" \
  hydra.job.name=test \
  "$@"
