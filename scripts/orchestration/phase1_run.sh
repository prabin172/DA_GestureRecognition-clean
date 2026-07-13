#!/usr/bin/env bash
# Phase 1 — one unmonitored pass. No retraining.
#   1. dump per-clip posteriors over existing multi-seed v2 checkpoints (3 seeds)
#   2. ECE + reliability + McNemar from those posteriors
#   3. layer-wise CKA (NTU vs Xsens-v2) per encoder
# Usage: nohup bash scripts/orchestration/phase1_run.sh > phase1_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

echo "===== PHASE 1 START $(date) ====="

echo "----- [1/3] dump posteriors (3 seeds x 5 subj x 4 methods x k{0,1,3}) -----"
python scripts/main_experiment/dump_posteriors.py

echo "----- [2/3] ECE + reliability + McNemar -----"
python scripts/main_experiment/analyze_calibration.py

echo "----- [3/3] layer-wise CKA (NTU vs Xsens-v2) -----"
python scripts/main_experiment/cka_analysis.py

echo "===== PHASE 1 DONE $(date) ====="
echo "Artifacts:"
echo "  trained_models/LOSO-fullTrainCalibrate-v2*/posteriors/*.csv"
echo "  trained_models/Phase1-analysis/{ece_results,mcnemar_results,cka_results}.csv"
echo "  trained_models/Phase1-analysis/{reliability_k*,cka_heatmap,cka_vs_benefit}.png"
