#!/usr/bin/env bash
# tasks.md T2.2 stages 1-2 -- SupCon parity. Waits for the T1 multi-seed chain to finish first
# (single shared GPU, serial schedule per human decision 2026-07-09) then runs:
#   stage 1: NTU linear probe for supcon (R1 within-domain row; in-script 10ep pretrain + 20ep
#            head, mirrors the existing scratch/supervised/mae/supmae rows in
#            trained_models/NTU-to-NTU-objective-sanity/summary.csv -- new supcon method branch
#            added to scripts/main_experiment/ntu_to_ntu_objective_sanity.py, same --seed 0 so train/adapt/test
#            split is byte-identical to the existing 4-method run).
#   stage 2: LOSO-v2, seeds 42/43/44, k in {0,1,3}, supcon only, isolated new out-dirs. Splits
#            are deterministic per (subject,k,base-seed) independent of the --methods list (see
#            scripts/main_experiment/loso_fulltrain_calibration.py:989-1009), so these pool/pair validly against
#            the existing 4-method LOSO-fullTrainCalibrate-v2{,-seed43,-seed44} runs.
# Run under nohup; do not monitor -- check for "T2 STAGE1+2 DONE" in this log.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== T2 waiting for T1 chain (t1_multiseed.log: T1 ALL DONE) ($(date)) ====="
while ! grep -q "T1 ALL DONE" t1_multiseed.log 2>/dev/null; do
  sleep 180
done
echo "T1 done, starting T2 stage 1-2 ($(date))"

echo "===== T2 STAGE 1: NTU linear probe, supcon ($(date)) ====="
python scripts/main_experiment/ntu_to_ntu_objective_sanity.py \
  --methods supcon \
  --out-dir trained_models/NTU-to-NTU-objective-sanity-supcon \
  --seed 0
echo "===== T2 STAGE 1 DONE ($(date)) ====="

echo "===== T2 STAGE 2: LOSO-v2, supcon, seeds 42/43/44 ($(date)) ====="
export LOSO_IMU_DIR="Data_Processed/imu_quats_v2"
for SEED in 42 43 44; do
  OUT="trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed${SEED}"
  echo "----- SEED ${SEED} -> ${OUT} ($(date)) -----"
  python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "${OUT}" \
    --methods "supcon" \
    --k-values "0,1,3" \
    --base-seed "${SEED}"
done
echo "===== T2 STAGE 2 DONE ($(date)) ====="

echo "===== T2 STAGE1+2 DONE $(date) ====="
