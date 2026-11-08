#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
experiment="${1:-mosi}"
if [ "$#" -gt 0 ]; then shift; fi
export TOKENIZERS_PARALLELISM=false
exec "${MIIB_PYTHON:-python}" run.py --config-name "$experiment" "$@"
