#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec bash "${repo_root}/scripts/reproduce_table.sh" --help
fi

exec bash "${repo_root}/scripts/reproduce_table.sh" afhqv2_64x64 "$@"
