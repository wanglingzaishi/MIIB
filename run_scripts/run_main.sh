#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment="${1:?Usage: bash run_scripts/run_main.sh EXPERIMENT [Hydra overrides...]}"
shift
for seed in 42 43 44 45 46; do
    bash "$project_dir/run_scripts/run.sh" "$experiment" "seed=$seed" "$@"
done
