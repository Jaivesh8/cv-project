#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -v PYTHON ]]; then
    python_cmd="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_cmd=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
    python_cmd=.venv/Scripts/python.exe
else
    python_cmd=python
fi
export PYTHONUTF8=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
if [[ "$#" == 0 && -f results/main.csv && -f results/predictions.jsonl ]]; then
    set -- --replay
fi
exec "$python_cmd" -m audit "$@"
