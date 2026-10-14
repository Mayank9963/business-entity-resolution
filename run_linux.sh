#!/usr/bin/env bash
# One-command run on Linux / macOS. Usage: ./run_linux.sh [--jobs 3] [--from train]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
python3 -m pip install -r requirements.txt
cd src
python3 -u run_all.py "$@" 2>&1 | tee ../run_log.txt
echo "Finished. Submission file: $(pwd)/../output/matching_results.tsv"
