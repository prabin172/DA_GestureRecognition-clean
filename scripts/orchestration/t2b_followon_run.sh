#!/usr/bin/env bash
# tasks.md follow-on chain -- T2.2 stages 3/5/6 + T5 cold-start sweep. Gated on
# t2_supcon_run.sh (t2_supcon.log: "T2 STAGE1+2 DONE") so it runs unattended once the
# NTU probe + LOSO-v2 supcon checkpoints land. Does NOT touch t2_supcon_run.sh (a separate
# script, to avoid editing a file bash is actively reading). New dirs / new --out-dir flags only;
# the locked R5 numbers in trained_models/Phase3-controller/robust/ are never touched (stage 5
# below writes to .../robust-supcon/ instead). Run under nohup; do not monitor -- check for
# "T2B FOLLOWON ALL DONE" in this log.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== T2B waiting for T2 stage1+2 (t2_supcon.log: T2 STAGE1+2 DONE) ($(date)) ====="
while ! grep -q "T2 STAGE1+2 DONE" t2_supcon.log 2>/dev/null; do
  sleep 180
done
echo "T2 stage1+2 done, starting T2B follow-on ($(date))"

# ---------- T2.2 stage 3: posterior dumps + McNemar/ECE, supcon extended ----------
echo "===== T2B STAGE 3: posterior dumps (scratch..supcon, all 6 seed-dirs) ($(date)) ====="
python scripts/main_experiment/dump_posteriors.py \
  --seed-dirs "trained_models/LOSO-fullTrainCalibrate-v2,trained_models/LOSO-fullTrainCalibrate-v2-seed43,trained_models/LOSO-fullTrainCalibrate-v2-seed44,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed42,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed43,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed44" \
  --methods "scratch,supLP120,supMAE,mae,supcon"
echo "===== T2B STAGE 3a: dump DONE ($(date)) ====="

echo "===== T2B STAGE 3b: McNemar/ECE analysis, supcon extended ($(date)) ====="
python scripts/main_experiment/analyze_calibration.py \
  --seed-dirs "trained_models/LOSO-fullTrainCalibrate-v2,trained_models/LOSO-fullTrainCalibrate-v2-seed43,trained_models/LOSO-fullTrainCalibrate-v2-seed44,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed42,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed43,trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed44" \
  --out-dir "trained_models/Phase1-analysis-supcon"
echo "===== T2B STAGE 3 DONE ($(date)) ====="

# ---------- T2.2 stage 5: controller robust re-run, supcon extended ----------
echo "===== T2B STAGE 5: controller_robust re-run with supcon ($(date)) ====="
python scripts/controller/controller_robust.py --out-dir "trained_models/Phase3-controller/robust-supcon"
echo "===== T2B STAGE 5 DONE ($(date)) ====="

# ---------- T2.2 stage 6: external datasets, supcon only, 3 seeds each ----------
echo "===== T2B STAGE 6: external datasets, supcon, seeds 42/43/44 ($(date)) ====="
for S in 42 43 44; do
  echo "----- STAGE 6a SEED ${S} CZU skeleton (R6) ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/czu_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/CZU-skeleton-LOSO-supcon-seed${S}" \
    --methods "supcon" --k-values "0,1,3" --base-seed ${S}

  echo "----- STAGE 6b SEED ${S} CZU IMU quat (R6b) ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/czu_imu_quats" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/CZU-IMU-LOSO-supcon-seed${S}" \
    --methods "supcon" --k-values "0,1,3" --base-seed ${S}

  echo "----- STAGE 6c SEED ${S} CZU dual (R6c) ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode dual --priors "supcon" \
    --seed ${S} --out-root "trained_models/CZU-IMU-DUAL-supcon-seed${S}"

  echo "----- STAGE 6d SEED ${S} UTD skeleton (R6d) ($(date)) -----"
  LOSO_IMU_DIR="Data_Processed/utd_skeleton_lrq" python scripts/main_experiment/loso_fulltrain_calibration.py \
    --out-dir "trained_models/UTD-skeleton-LOSO-supcon-seed${S}" \
    --methods "supcon" --k-values "0,1,3" --base-seed ${S}
done
echo "===== T2B STAGE 6 DONE ($(date)) ====="

# ---------- T5: CZU dual cold-start subject-scaling sweep ----------
echo "===== T5: CZU dual N-scaling sweep, N=0..3, seed 42 ($(date)) ====="
for N in 0 1 2 3; do
  echo "----- T5 N=${N} ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode dual --priors "scratch,supLP120,supMAE" \
    --n-train-subjects ${N} --seed 42 \
    --out-root "trained_models/CZU-DUAL-subjectScaling/N${N}"
done
echo "===== T5 DONE ($(date)) ====="

echo "===== T2B FOLLOWON ALL DONE $(date) ====="
