#!/usr/bin/env bash
# Stage 5b (OOV multi-seed extension): seeds 43/44, added 2026-07-20 after human
# decision to make every experiment 3-seed. Seed 42 output
# (trained_models/LOSO-LeaveClassOutFewShot/) untouched. ~15h/seed based on the
# seed-42 timing (full_rerun.log, Stage 5: 2026-07-14 08:36 -> 23:47).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 5b: OOV MULTISEED START $(date) ====="
for SEED in 43 44; do
  echo "----- seed=${SEED} ($(date)) -----"
  python scripts/main_experiment/loso_leave_class_out_fewshot.py \
    --methods "scratch,supLP120,supMAE,mae,supcon" \
    --base-seed "${SEED}" \
    --out-dir "trained_models/LOSO-LeaveClassOutFewShot-seed${SEED}"
done
echo "===== STAGE 5b OOV MULTISEED DONE $(date) ====="
