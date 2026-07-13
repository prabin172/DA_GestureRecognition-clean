#!/usr/bin/env bash
# UTD-MHAD skeleton LOSO (second same-modality target). Tests the small-gap "supervised wins"
# prediction on an independent public dataset: expect supLP120 >= scratch, replicating CZU-R6.
# 8-subject LOSO, k=0/1/3, methods scratch/supLP120/supMAE/mae, 3 seeds (42,43,44) from the start.
# Data: Data_Processed/utd_skeleton_lrq (scripts/data_pipeline/utd_parser.py). All NEW dirs; nothing existing touched.
# Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

for S in 42 43 44; do
  echo "===== UTD SEED ${S} skeleton LOSO ($(date)) ====="
  LOSO_IMU_DIR="Data_Processed/utd_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/UTD-skeleton-LOSO-seed${S}" \
    --methods "scratch,supLP120,supMAE,mae" --k-values "0,1,3" --base-seed ${S}
done
echo "===== UTD-SKELETON ALL DONE ($(date)) ====="
