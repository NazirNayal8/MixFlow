#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'EOF'
Usage: bash scripts/run_curvature.sh <experiment> <checkpoint> <output_dir> [hydra overrides...]

Examples:
  bash scripts/run_curvature.sh cifar10 /path/to/model.ckpt outputs/curvature/cifar10
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_curvature.sh cifar10 /path/to/model.ckpt outputs/curvature/cifar10
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_curvature.sh ffhq_64x64 /path/to/model.ckpt outputs/curvature/ffhq test.num_samples=10000
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

case "$experiment" in
  cifar10)
    data_shape='[3,32,32]'
    ;;
  ffhq_64x64|afhqv2_64x64)
    data_shape='[3,64,64]'
    ;;
  *)
    echo "Unsupported experiment '${experiment}'. Expected one of: cifar10, ffhq_64x64, afhqv2_64x64"
    exit 1
    ;;
esac

mkdir -p "${output_dir}"

python -m src.main \
  "+experiment=${experiment}" \
  checkpointing.load="${checkpoint}" \
  mode=test \
  test.curvature=true \
  test.output_path="${output_dir}" \
  test.num_samples=10000 \
  test.num_inference_timesteps=128 \
  test.ode_solver=euler \
  test.curvature_cfg.data_shape="${data_shape}" \
  hydra.job.name=curvature \
  "$@"
