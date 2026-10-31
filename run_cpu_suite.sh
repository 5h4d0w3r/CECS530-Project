#!/bin/sh
# Run from any directory after installing the project dependencies.
set -eu
cd "$(dirname "$0")"
attention_python="${1:-.venv/bin/python}"
"$attention_python" -m unittest -v > test_results.txt 2>&1
"$attention_python" -m pip freeze > environment.lock.txt
"$attention_python" capture_environment.py
"$attention_python" benchmark_extended.py --output extended_results.json > extended_run.log 2>&1
"$attention_python" benchmark_extended.py --lengths 2048 4096 --scopes kernel --phases forward \
  --warmup 1 --repeats 5 --output memory_results.json > memory_run.log 2>&1
for tile in 16 64 128; do
  "$attention_python" benchmark_extended.py --lengths 512 --scopes kernel \
    --phases forward forward_backward --methods online --block-size "$tile" \
    --output "tile_${tile}_results.json" > "tile_${tile}_run.log" 2>&1
done
"$attention_python" summarize_results.py
