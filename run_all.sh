#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -n "${PYTHON:-}" ]]; then
    python_cmd="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
    python_cmd=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
    python_cmd=.venv/Scripts/python.exe
else
    python_cmd=python
fi
export PYTHONUTF8=1
export PYTHONFAULTHANDLER=1
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
if [[ "$#" == 0 && -f results/main.csv && -f results/predictions.jsonl ]]; then
    set -- --replay
fi
for arg in "$@"; do
    if [[ "$arg" == "--replay" || "$arg" == "--help" || "$arg" == "-h" ]]; then
        exec "$python_cmd" -m audit "$@"
    fi
done
checkpoints="$("$python_cmd" -c 'import json; print(*json.load(open("protocol.json"))["models"], sep="\n")')"
while IFS= read -r checkpoint; do
    "$python_cmd" -m audit "$@" --model "${checkpoint%$'\r'}"
done <<< "$checkpoints"
exec "$python_cmd" -m audit "$@" --replay
