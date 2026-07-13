#!/usr/bin/env bash
# tasks.md T1 — multi-seed completion runs (seeds 43/44) for the three single-seed pieces
# flagged in the 2026-07-09 publishability review: A2 subject-scaling (T1.1), R6c raw-only
# mode (T1.2), DIAL 20-ch rung (T1.3). Chained serially under one nohup (single shared GPU,
# already running an unrelated IsaacLab job — avoid contending with itself).
# All outputs to NEW dirs; existing seed-42 runs (A2-subjectScaling/, CZU-IMU-DUAL/dual_*,
# CZU-IMU-DIAL/mag20/) are left untouched. Flags mirror the original seed-42 launches
# (a2_run.sh, czu_dual_run.sh --mode raw, czu_dial_run.sh) so results pool.
# Run under nohup; do not monitor -- check for "T1 ALL DONE" in this log.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"   # v2 data, matches Phase-1 checkpoints (A2 only)

echo "===== T1 START $(date) ====="

for S in 43 44; do
  echo "===== T1.1 SEED ${S} A2 subject-scaling ($(date)) ====="
  for N in 0 1 2 3 4; do
    echo "----- seed ${S} N=${N} -----"
    python scripts/main_experiment/loso_fulltrain_calibration.py \
      --out-dir "trained_models/A2-subjectScaling-seed${S}/N${N}" \
      --methods "scratch,supLP120,supMAE,mae" \
      --k-values "0,1,3" \
      --n-train-subjects "${N}" \
      --base-seed ${S}
  done
  python scripts/main_experiment/analyze_a2.py \
    --root "trained_models/A2-subjectScaling-seed${S}" \
    --out-dir "trained_models/A2-subjectScaling-seed${S}/analysis"
done
echo "===== T1.1 A2 SEEDS 43/44 DONE ($(date)) ====="

for S in 43 44; do
  echo "===== T1.2 SEED ${S} R6c raw-only mode ($(date)) ====="
  python scripts/external/czu/dualbranch.py --mode raw --seed ${S} \
    --out-root "trained_models/CZU-IMU-DUAL-seed${S}"
done
echo "===== T1.2 RAW-ONLY SEEDS 43/44 DONE ($(date)) ====="

for S in 43 44; do
  echo "===== T1.3 SEED ${S} DIAL 20-ch rung ($(date)) ====="
  python scripts/external/czu/dualbranch.py --mode dual --priors "scratch,supLP120,supMAE,mae" \
    --raw-dir "Data_Processed/czu_imu_mag20" --raw-dim 20 \
    --seed ${S} --out-root "trained_models/CZU-IMU-DIAL/mag20-seed${S}"
done
echo "===== T1.3 DIAL SEEDS 43/44 DONE ($(date)) ====="

echo "===== T1 ALL DONE $(date) ====="
