#!/usr/bin/env bash
# Stage 6: CZU-MHAD external validity -- skeleton (R6), IMU-quat (R6b), dual-branch (R6c),
# DIAL dose-response rung. 3 seeds x 5 methods throughout, then CRC published-baselines
# (deterministic given seed-42 splits, run once each) + 3-seed pooling/stats.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 6: CZU EXTERNAL START $(date) ====="
for SEED in 42 43 44; do
  if [ "$SEED" = "42" ]; then SK_OUT="trained_models/CZU-skeleton-LOSO"; IMU_OUT="trained_models/CZU-IMU-LOSO"; DUAL_OUT="trained_models/CZU-IMU-DUAL"; DIAL_OUT="trained_models/CZU-IMU-DIAL/mag20"
  else SK_OUT="trained_models/CZU-skeleton-LOSO-seed${SEED}"; IMU_OUT="trained_models/CZU-IMU-LOSO-seed${SEED}"; DUAL_OUT="trained_models/CZU-IMU-DUAL-seed${SEED}"; DIAL_OUT="trained_models/CZU-IMU-DIAL/mag20-seed${SEED}"; fi

  echo "----- seed ${SEED} [1/4] skeleton R6 ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/czu_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "${SK_OUT}" --methods "${METHODS}" --k-values "0,1,3" --base-seed "${SEED}"

  echo "----- seed ${SEED} [2/4] IMU-quat R6b ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/czu_imu_quats" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "${IMU_OUT}" --methods "${METHODS}" --k-values "0,1,3" --base-seed "${SEED}"

  echo "----- seed ${SEED} [3/4] dual-branch R6c: raw/quat/dual ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode raw --seed "${SEED}" --out-root "${DUAL_OUT}"
  python scripts/external/czu/dualbranch.py --mode quat --priors "${METHODS}" --seed "${SEED}" --out-root "${DUAL_OUT}"
  python scripts/external/czu/dualbranch.py --mode dual --priors "${METHODS}" --seed "${SEED}" --out-root "${DUAL_OUT}"

  echo "----- seed ${SEED} [4/4] DIAL dose-response (mag20 rung) ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode dual --priors "${METHODS}" \
    --raw-dir "Data_Processed/czu_imu_mag20" --raw-dim 20 --seed "${SEED}" --out-root "${DIAL_OUT}"
done

echo "----- CRC published-baselines (deterministic, seed-42 splits only) -----"
python scripts/external/czu/crc_baseline.py
python scripts/external/czu/imu_crc_baseline.py

echo "----- 3-seed pooling + sign-test/paired-t stats -----"
python scripts/external/czu/multiseed_analyze.py

echo "===== STAGE 6 CZU EXTERNAL DONE $(date) ====="
