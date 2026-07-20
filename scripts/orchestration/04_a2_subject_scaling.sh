#!/usr/bin/env bash
# Stage 4: A2 subject-count scaling. N in {0..4} x 3 seeds x 5 methods x k{0,1,3}.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 4: A2 SUBJECT-SCALING START $(date) ====="
for SEED in 42 43 44; do
  if [ "$SEED" = "42" ]; then ROOT="trained_models/A2-subjectScaling"
  else ROOT="trained_models/A2-subjectScaling-seed${SEED}"; fi
  for N in 0 1 2 3 4; do
    echo "----- seed ${SEED} N=${N} ($(date)) -----"
    python scripts/main_experiment/loso_fulltrain_calibration.py \
      --out-dir "${ROOT}/N${N}" --methods "${METHODS}" --k-values "0,1,3" \
      --n-train-subjects "${N}" --base-seed "${SEED}"
  done
done

echo "----- pooled 3-seed analysis + paired-t stats -----"
python scripts/main_experiment/analyze_a2_multiseed.py

echo "===== STAGE 4 A2 DONE $(date) ====="
