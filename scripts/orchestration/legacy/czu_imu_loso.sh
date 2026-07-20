#!/usr/bin/env bash
# TRUE cross-modal external replication: NTU-pretrained encoders -> LOSO k-shot on CZU-MHAD
# INERTIAL (wearable IMU, 10 sensors, 6-axis, no mag -> Madgwick-derived 17-seg local quats).
# Independent public dataset (5 subj, 22 actions). Does the NTU prior transfer across modality,
# and does mae/supMAE beat supLP120 when the modality actually changes (vs the skeleton->skeleton
# CZU run where supLP120 dominated)? Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== [1] CZU INERTIAL deep LOSO ($(date)) ====="
LOSO_IMU_DIR="Data_Processed/czu_imu_quats" python scripts/main_experiment/loso_fulltrain_calibration.py \
  --out-dir trained_models/CZU-IMU-LOSO \
  --methods "scratch,supLP120,supMAE,mae" \
  --k-values "0,1,3" --base-seed 42
echo "===== deep LOSO DONE ($(date)) ====="

echo "===== [2] CZU inertial CRC baseline (same splits) ($(date)) ====="
python scripts/external/czu/imu_crc_baseline.py
echo "===== CZU-IMU ALL DONE ($(date)) ====="
