#!/usr/bin/env bash
# CZU published-baseline reproduction: statistical-moments + CRC on identical LOSO splits.
# Run under nohup; sequential (single fast job). Log -> czu_crc_baseline.log in repo root.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== [1] CZU CRC baseline ($(date)) ====="
python scripts/external/czu/crc_baseline.py
echo "===== CZU CRC BASELINE DONE ($(date)) ====="
