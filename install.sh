#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_CMD="${PYTHON:-python}"

read -r -a PYTHON_ARGS <<< "$PYTHON_CMD"

"${PYTHON_ARGS[@]}" -m venv .venv

if [ -x ".venv/bin/python" ]; then
  VENV_PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  echo "Could not find Python inside .venv" >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e "artifact/.[dev,metrics]"

echo "PseudoHunter artifact install complete."
echo "Run: bash claims/claim1_lopo_main/run.sh"
