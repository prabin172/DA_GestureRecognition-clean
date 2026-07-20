#!/usr/bin/env bash
# Stage 9 (T5): does the A2 cold-start deployment lever survive on a strong (raw-signal)
# target? Single seed (42) -- this is a scoping check, not a headline claim (see
# czu-dual-cold-start.md). N in {0..3}, all 5 methods (original scope was
# scratch/supLP120/supMAE only; extended here for full 5-method parity).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

METHODS="scratch,supLP120,supMAE,mae,supcon"

echo "===== STAGE 9: CZU-DUAL COLD-START START $(date) ====="
for N in 0 1 2 3; do
  echo "----- N=${N} ($(date)) -----"
  python scripts/external/czu/dualbranch.py --mode dual --priors "${METHODS}" \
    --n-train-subjects "${N}" --seed 42 --out-root "trained_models/CZU-DUAL-subjectScaling/N${N}"
done
echo "===== STAGE 9 COLD-START DONE $(date) ====="
