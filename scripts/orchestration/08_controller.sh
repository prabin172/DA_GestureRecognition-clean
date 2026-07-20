#!/usr/bin/env bash
# Stage 8: downstream controller -- prototype (its numbers ARE cited, R5's opening headline)
# + the load-bearing robustness protocol (3 locks). Both read Stage 2/3's posteriors, no
# retraining. Reads all 5 methods natively (METHODS lists fixed during the migration).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 8: CONTROLLER START $(date) ====="
echo "----- [1/2] prototype (single fixed config, illustrative headline number) -----"
python scripts/controller/controller_sim.py

echo "----- [2/2] robustness protocol (120 vocabs x 2 outcome models x tuning-free op point) -----"
python scripts/controller/controller_robust.py

echo "===== STAGE 8 CONTROLLER DONE $(date) ====="
echo "Locked output: trained_models/Phase3-controller/robust/ -- do not overwrite in any later run."
