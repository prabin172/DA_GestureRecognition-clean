#!/usr/bin/env bash
# CZU multi-seed (item 2): seeds 43 + 44 for R6 skeleton, R6b quat, R6c dual. Pooled with the
# existing seed-42 runs -> 3 seeds / 45 folds, retiring the "n.s. paired-t at n=5" caveats.
# All flags identical to the original seed-42 launches (czu_loso.sh, czu_imu_loso.sh,
# czu_dual_run.sh) so results pool cleanly. ALL outputs to NEW *-seed<S> dirs; existing
# CZU-skeleton-LOSO / CZU-IMU-LOSO / CZU-IMU-DUAL are left UNTOUCHED. Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

for S in 43 44; do
  echo "===== SEED ${S} [1] skeleton R6 ($(date)) ====="
  LOSO_IMU_DIR="Data_Processed/czu_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/CZU-skeleton-LOSO-seed${S}" \
    --methods "scratch,supLP120,supMAE,mae" --k-values "0,1,3" --base-seed ${S}

  echo "===== SEED ${S} [2] quat R6b ($(date)) ====="
  LOSO_IMU_DIR="Data_Processed/czu_imu_quats" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/CZU-IMU-LOSO-seed${S}" \
    --methods "scratch,supLP120,supMAE,mae" --k-values "0,1,3" --base-seed ${S}

  echo "===== SEED ${S} [3] dual R6c ($(date)) ====="
  python scripts/external/czu/dualbranch.py --mode dual --priors "scratch,supLP120,supMAE,mae" \
    --seed ${S} --out-root "trained_models/CZU-IMU-DUAL-seed${S}"
done
echo "===== CZU MULTISEED ALL DONE ($(date)) ====="
