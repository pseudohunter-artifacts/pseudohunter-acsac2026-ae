#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
fi

"$PYTHON" artifact/tools/verify_claims.py claim1 \
  --result artifact/paper_results/path_ablation/lopo_results_routing_path_dropout_full.json
