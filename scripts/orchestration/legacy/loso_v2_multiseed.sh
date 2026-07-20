#!/usr/bin/env bash
# Multi-seed LOSO-v2 sweep for the stats requirement (paper_idea.md §7.2 / A3 policy).
# Seed 42 already exists in trained_models/LOSO-fullTrainCalibrate-v2/ (all k).
# This adds seeds 43,44 at k in {0,1,3} for the 4 core methods -> 3 seeds total for paired stats.
# Sequential (not parallel) to avoid GPU contention. Run under nohup; do not monitor.
set -euo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"

METHODS="scratch,supLP120,supMAE,mae"
KVALS="0,1,3"

for SEED in 43 44; do
  OUT="trained_models/LOSO-fullTrainCalibrate-v2-seed${SEED}"
  echo "===== SEED ${SEED} -> ${OUT} ($(date)) ====="
  python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "${OUT}" \
    --methods "${METHODS}" \
    --k-values "${KVALS}" \
    --base-seed "${SEED}"
done
echo "===== MULTISEED SWEEP DONE ($(date)) ====="
