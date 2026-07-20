#!/usr/bin/env bash
# Dual-branch cross-modal study. (1) raw-only diagnostic: does a learned encoder on the full
# raw signal reach CRC? (2) quat-only: sanity re-check vs R6b. (3) dual: does the NTU prior add
# value on top of a strong raw target model? Run under nohup; do not monitor.
set -uo pipefail
cd /home/ptimilsina/projects/DA_GestureRecognition
source .venv/bin/activate

echo "===== [1] RAW-only diagnostic ($(date)) ====="
python scripts/external/czu/dualbranch.py --mode raw
echo "===== [2] QUAT-only sanity (vs R6b) ($(date)) ====="
python scripts/external/czu/dualbranch.py --mode quat --priors "scratch,supLP120,supMAE,mae"
echo "===== [3] DUAL (raw + prior) ($(date)) ====="
python scripts/external/czu/dualbranch.py --mode dual --priors "scratch,supLP120,supMAE,mae"
echo "===== CZU-DUAL ALL DONE ($(date)) ====="
