#!/usr/bin/env bash
# Stage 3: everything derived from Stage 2's posteriors -- McNemar, ECE, CKA, MMD, raw-domain-gap.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 3: MAIN ANALYSIS START $(date) ====="
echo "----- [1/5] dump per-clip posteriors (3 seeds x 5 methods x k{0,1,3}) -----"
python scripts/main_experiment/dump_posteriors.py

echo "----- [2/5] ECE + reliability + McNemar -----"
python scripts/main_experiment/analyze_calibration.py

echo "----- [3/5] layer-wise CKA (NTU vs Xsens-v2, + per-target T3 gap-axis check) -----"
python scripts/main_experiment/cka_analysis.py
python scripts/main_experiment/cka_analysis.py --multi-target

echo "----- [4/5] encoder-space MMD -----"
python scripts/main_experiment/mmd_domain_gap.py --include-supcon

echo "----- [5/5] raw, encoder-free domain gap (MMD^2/Frechet -- confirms the gap ordering CKA misses) -----"
python scripts/main_experiment/raw_domain_gap.py

echo "===== STAGE 3 MAIN ANALYSIS DONE $(date) ====="
