#!/usr/bin/env bash
# Tier C external-validity run: NTU-pretrained encoders -> LOSO k-shot on CZU-MHAD skeleton LRQ.
# Independent public dataset (5 subj, 22 actions, Kinect-v2 skeleton). Does the prior-benefit
# ordering (supMAE/supLP120 vs scratch) reproduce on a totally different target?
# Waits for the alpha sweep to finish first (no GPU contention). Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== [1] wait for alpha sweep to finish (avoid GPU contention) ($(date)) ====="
while ! grep -q "ALPHA SWEEP DONE" alpha_sweep.log 2>/dev/null; do
  sleep 180
done
echo "alpha sweep done, starting CZU LOSO ($(date))"

echo "===== [2] CZU skeleton LOSO ($(date)) ====="
LOSO_IMU_DIR="Data_Processed/czu_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
  --out-dir trained_models/CZU-skeleton-LOSO \
  --methods "scratch,supLP120,supMAE,mae" \
  --k-values "0,1,3" --base-seed 42
echo "===== CZU LOSO DONE ($(date)) ====="
