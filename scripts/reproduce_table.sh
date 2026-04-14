#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

show_usage() {
  cat <<'EOF'
Usage: bash scripts/reproduce_table.sh <experiment> [hydra overrides...]

Supported experiments:
  cifar10
  ffhq_64x64
  afhqv2_64x64

Environment variables:
  CHECKPOINT   Optional explicit checkpoint path. If unset, the script will look
               for the default checkpoint under ./checkpoints/.
  OUTPUT_ROOT  Optional output directory. Defaults to
               outputs/main_results/<experiment>.

Examples:
  bash scripts/reproduce_table.sh cifar10
  CHECKPOINT=/path/to/model.ckpt bash scripts/reproduce_table.sh ffhq_64x64
  OUTPUT_ROOT=outputs/paper_tables bash scripts/reproduce_table.sh afhqv2_64x64
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
extra_overrides=("$@")

resolve_checkpoint() {
  local experiment_name="$1"
  local candidates=()

  if [ -n "${CHECKPOINT:-}" ]; then
    if [ ! -f "${CHECKPOINT}" ]; then
      echo "Checkpoint not found: ${CHECKPOINT}" >&2
      return 1
    fi
    printf '%s\n' "${CHECKPOINT}"
    return 0
  fi

  case "${experiment_name}" in
    cifar10)
      candidates=(
        "${repo_root}/checkpoints/mixflow_cifar10.ckpt"
        "${repo_root}/checkpoints/mixflow_cifar10"
      )
      ;;
    ffhq_64x64)
      candidates=(
        "${repo_root}/checkpoints/mixflow_ffhqv2_64x64.ckpt"
        "${repo_root}/checkpoints/mixflow_ffhqv2_64x64"
      )
      ;;
    afhqv2_64x64)
      candidates=(
        "${repo_root}/checkpoints/mixflow_afhqv2_64x64.ckpt"
        "${repo_root}/checkpoints/mixflow_afhqv2_64x64"
      )
      ;;
    *)
      echo "Unsupported experiment '${experiment_name}'." >&2
      return 1
      ;;
  esac

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "No default checkpoint found for '${experiment_name}'." >&2
  echo "Set CHECKPOINT=/path/to/model.ckpt or place the file under ${repo_root}/checkpoints/." >&2
  return 1
}

json_field() {
  local metrics_path="$1"
  local field_name="$2"
  python - "$metrics_path" "$field_name" <<'PY'
import json
import sys

metrics_path, field_name = sys.argv[1], sys.argv[2]
with open(metrics_path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get(field_name)
if value is None:
    print("NA")
elif isinstance(value, float):
    print(f"{value:.4f}")
else:
    print(value)
PY
}

write_summary_header() {
  case "${experiment}" in
    cifar10)
      cat > "${summary_path}" <<'EOF'
# CIFAR-10 Table Reproduction

| Run | Solver | Timesteps | Mean NFE | FID | IS |
| --- | --- | ---: | ---: | ---: | ---: |
EOF
      ;;
    ffhq_64x64|afhqv2_64x64)
      cat > "${summary_path}" <<'EOF'
# Sampling Budget Sweep

| Steps | Solver | Mean NFE | FID |
| ---: | --- | ---: | ---: |
EOF
      ;;
  esac
}

append_cifar_row() {
  local label="$1"
  local solver="$2"
  local timesteps="$3"
  local metrics_path="$4"

  local mean_nfe fid is_score
  mean_nfe="$(json_field "${metrics_path}" mean_nfe)"
  fid="$(json_field "${metrics_path}" FID)"
  is_score="$(json_field "${metrics_path}" IS)"

  printf '| %s | %s | %s | %s | %s | %s |\n' \
    "${label}" "${solver}" "${timesteps}" "${mean_nfe}" "${fid}" "${is_score}" \
    >> "${summary_path}"
}

append_budget_row() {
  local steps="$1"
  local solver="$2"
  local metrics_path="$3"

  local mean_nfe fid
  mean_nfe="$(json_field "${metrics_path}" mean_nfe)"
  fid="$(json_field "${metrics_path}" FID)"

  printf '| %s | %s | %s | %s |\n' \
    "${steps}" "${solver}" "${mean_nfe}" "${fid}" \
    >> "${summary_path}"
}

run_eval() {
  local run_name="$1"
  shift

  local run_dir="${output_root}/${run_name}"
  mkdir -p "${run_dir}"

  echo
  echo "==> ${experiment}: ${run_name}"

  bash "${repo_root}/scripts/run_inference.sh" \
    "${experiment}" \
    "${checkpoint}" \
    "${run_dir}" \
    "output_dir=${run_dir}" \
    "wandb.mode=disabled" \
    "$@" \
    "${extra_overrides[@]}"

  local metrics_path="${run_dir}/metrics.json"
  if [ ! -f "${metrics_path}" ]; then
    echo "Expected metrics file was not produced: ${metrics_path}" >&2
    return 1
  fi
}

checkpoint="$(resolve_checkpoint "${experiment}")"
output_root="${OUTPUT_ROOT:-${repo_root}/outputs/main_results/${experiment}}"
mkdir -p "${output_root}"
summary_path="${output_root}/summary.md"

write_summary_header

case "${experiment}" in
  cifar10)
    run_eval \
      "rk45" \
      "test.num_samples=50000" \
      "test.num_inference_timesteps=1001" \
      "test.ode_solver=rk45" \
      "test.use_scipy=true" \
      "test.compute_is=true"
    append_cifar_row "RK45" "rk45" "1001" "${output_root}/rk45/metrics.json"

    run_eval \
      "heun_5nfe" \
      "test.num_samples=50000" \
      "test.num_inference_timesteps=3" \
      "test.ode_solver=manual_heun" \
      "test.compute_is=true"
    append_cifar_row "Heun (5 NFE)" "manual_heun" "3" "${output_root}/heun_5nfe/metrics.json"

    run_eval \
      "heun_9nfe" \
      "test.num_samples=50000" \
      "test.num_inference_timesteps=5" \
      "test.ode_solver=manual_heun" \
      "test.compute_is=true"
    append_cifar_row "Heun (9 NFE)" "manual_heun" "5" "${output_root}/heun_9nfe/metrics.json"
    ;;
  ffhq_64x64|afhqv2_64x64)
    for steps in 4 5 10 20 32 64 128; do
      run_eval \
        "euler_${steps}_steps" \
        "test.num_samples=10000" \
        "test.num_inference_timesteps=${steps}" \
        "test.ode_solver=euler"
      append_budget_row "${steps}" "euler" "${output_root}/euler_${steps}_steps/metrics.json"
    done
    ;;
  *)
    echo "Unsupported experiment '${experiment}'." >&2
    exit 1
    ;;
esac

echo
echo "Finished reproducing ${experiment} table runs."
echo "Per-run outputs: ${output_root}"
echo "Summary table: ${summary_path}"
