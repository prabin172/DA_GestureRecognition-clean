#!/usr/bin/env bash
# Stage 9b (T5 multi-seed extension): seeds 43/44 for CZU-DUAL cold-start, added
# 2026-07-20 after human decision to promote T5 from single-seed scoping check to
# a load-bearing 3-seed result (matches R6c/R4c treatment). Seed 42 output
# (trained_models/CZU-DUAL-subjectScaling/) untouched.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 9b: CZU-DUAL COLD-START MULTISEED START $(date) ====="
for SEED in 43 44; do
  for N in 0 1 2 3; do
    echo "----- seed=${SEED} N=${N} ($(date)) -----"
    python scripts/external/czu/dualbranch.py --mode dual --priors "${METHODS}" \
      --n-train-subjects "${N}" --seed "${SEED}" \
      --out-root "trained_models/CZU-DUAL-subjectScaling-seed${SEED}/N${N}"
  done
done
echo "===== STAGE 9b COLD-START MULTISEED DONE $(date) ====="
