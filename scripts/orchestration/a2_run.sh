#!/usr/bin/env bash
# Phase 2 (A2) — subject-count scaling. Seed 42 only.
# For each N in {0,1,2,3,4}: NTU-pretrained init -> fine-tune on N nested non-held-out
# subjects -> k-shot calibrate+eval on the held-out subject. 5 held-out x 4 methods x k{0,1,3}.
# Usage: nohup bash scripts/orchestration/a2_run.sh > a2_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"   # v2 data, matches Phase-1 checkpoints

echo "===== A2 START $(date) ====="
for N in 0 1 2 3 4; do
  echo "----- N=${N} fine-tuning subjects -----"
  python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/A2-subjectScaling/N${N}" \
    --methods "scratch,supLP120,supMAE,mae" \
    --k-values "0,1,3" \
    --n-train-subjects "${N}" \
    --base-seed 42
done

echo "----- A2 analysis -----"
python scripts/main_experiment/analyze_a2.py

echo "===== A2 DONE $(date) ====="
echo "Artifacts: trained_models/A2-subjectScaling/N*/summary.csv"
echo "           trained_models/A2-subjectScaling/analysis/{a2_results.csv,a2_benefit_vs_N.png,a2_acc_vs_N.png}"
