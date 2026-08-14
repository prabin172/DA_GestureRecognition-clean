#!/usr/bin/env bash
# Stage 10 (ad hoc, 2026-07-21): CZU dual-raw retrain for the controller cross-setting extension.
# dualbranch.py's `--mode dual` never persisted checkpoints or per-clip posteriors (only
# summary.csv accuracy) -- the controller needs per-clip posteriors, so this retrains mode=dual
# from scratch (3 seeds x 5 priors) with --dump-posteriors-dir added. Writes to NEW out-roots
# (CZU-IMU-DUAL-controller[-seed43/44]) so the existing locked CZU-IMU-DUAL* summary.csv (cited
# in paper_results.md R6c) is never touched.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 10 (ad hoc): CZU DUAL-RAW CONTROLLER RETRAIN START $(date) ====="
for SEED in 42 43 44; do
  if [ "$SEED" = "42" ]; then OUT="trained_models/CZU-IMU-DUAL-controller"
  else OUT="trained_models/CZU-IMU-DUAL-controller-seed${SEED}"; fi
  echo "----- seed ${SEED} mode=dual ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode dual --priors "${METHODS}" --seed "${SEED}" \
    --out-root "${OUT}" --dump-posteriors-dir "${OUT}/posteriors"
done
echo "===== STAGE 10 CZU DUAL-RAW CONTROLLER RETRAIN DONE $(date) ====="
