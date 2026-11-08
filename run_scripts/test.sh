#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:?Usage: bash run_scripts/test.sh EXPERIMENT CHECKPOINT [Hydra overrides...]}"
checkpoint="${2:?Usage: bash run_scripts/test.sh EXPERIMENT CHECKPOINT [Hydra overrides...]}"
shift 2
bash "$project_dir/run_scripts/run.sh" "$experiment" mode=test "ckpt_path=$checkpoint" save_checkpoints=false "$@"
