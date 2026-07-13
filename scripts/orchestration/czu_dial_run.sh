#!/usr/bin/env bash
# Target-strength DIAL (item 1): does the NTU prior's benefit decay monotonically as the target
# branch gets richer? Rungs on the target axis:
#   0 ch  = R6b quat-only            (existing: CZU-IMU-LOSO)
#   20 ch = quat + mag20 dual        (THIS run)
#   60 ch = R6c full-raw dual        (existing: CZU-IMU-DUAL/dual_*)
# If prior_benefit(0) > prior_benefit(20) > prior_benefit(60)~=0, "contingent on target poverty"
# becomes a dose-response curve, not a 2-point contrast. New outputs only; existing runs untouched.
# Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== [1] export mag20 target representation ($(date)) ====="
python scripts/data_pipeline/czu_imu_mag_export.py

echo "===== [2] dial rung: quat + mag20 dual, all priors ($(date)) ====="
python scripts/external/czu/dualbranch.py --mode dual --priors "scratch,supLP120,supMAE,mae" \
  --raw-dir "Data_Processed/czu_imu_mag20" --raw-dim 20 \
  --out-root "trained_models/CZU-IMU-DIAL/mag20"

echo "===== CZU DIAL ALL DONE ($(date)) ====="
