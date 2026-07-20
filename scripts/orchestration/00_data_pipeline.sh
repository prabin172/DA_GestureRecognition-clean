#!/usr/bin/env bash
# Stage 0: build every Data_Processed/ subdir the rest of the rerun needs, from raw data.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 0: DATA PIPELINE START $(date) ====="
echo "----- [1/6] Xsens v2 (position-derived quats) -----"
python scripts/data_pipeline/IMU_batch_processor_v2.py

echo "----- [2/6] CZU skeleton -----"
python scripts/data_pipeline/czu_parser.py

echo "----- [3/6] CZU IMU quats -----"
python scripts/data_pipeline/czu_imu_parser.py

echo "----- [4/6] CZU IMU raw (dual-branch target) -----"
python scripts/data_pipeline/czu_imu_raw_export.py

echo "----- [5/6] CZU IMU mag20 (DIAL dose-response rung) -----"
python scripts/data_pipeline/czu_imu_mag_export.py

echo "----- [6/6] UTD skeleton -----"
python scripts/data_pipeline/utd_parser.py

echo "===== STAGE 0 DATA PIPELINE DONE $(date) ====="
