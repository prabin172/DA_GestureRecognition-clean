#!/usr/bin/env bash
# Stage 7: UTD-MHAD external validity (R6d) -- second independent same-modality public
# dataset. 3 seeds x 5 methods (UTD suffixes ALL seeds incl. 42 -- matches crc_baseline.py's
# hardcoded split path, do not "clean up" this one inconsistency).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 7: UTD EXTERNAL START $(date) ====="
for SEED in 42 43 44; do
  echo "----- seed ${SEED} skeleton LOSO ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/utd_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/UTD-skeleton-LOSO-seed${SEED}" --methods "${METHODS}" --k-values "0,1,3" --base-seed "${SEED}"
done

echo "----- CRC published-baseline (deterministic, seed-42 splits) -----"
python scripts/external/utd/crc_baseline.py

echo "===== STAGE 7 UTD EXTERNAL DONE $(date) ====="
