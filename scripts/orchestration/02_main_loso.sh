#!/usr/bin/env bash
# Stage 2: main Xsens LOSO -- THE core experiment. 3 seeds, all 5 methods together (not
# bolted on incrementally), k in {0,1,3}. seed 42 unsuffixed (matches every downstream
# analysis script's hardcoded expectation), 43/44 suffixed.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"

METHODS="scratch,supLP120,supMAE,mae,supcon"
KVALS="0,1,3"

echo "===== STAGE 2: MAIN XSENS LOSO START $(date) ====="
for SEED in 42 43 44; do
  if [ "$SEED" = "42" ]; then OUT="trained_models/LOSO-fullTrainCalibrate-v2"
  else OUT="trained_models/LOSO-fullTrainCalibrate-v2-seed${SEED}"; fi
  echo "----- seed ${SEED} -> ${OUT} ($(date)) -----"
  python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "${OUT}" --methods "${METHODS}" --k-values "${KVALS}" --base-seed "${SEED}"
done
echo "===== STAGE 2 MAIN LOSO DONE $(date) ====="
