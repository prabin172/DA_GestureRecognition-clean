#!/usr/bin/env bash
# Stage 1: NTU pretraining, 4 objectives (scratch needs no pretrain -- random init,
# baked into the downstream harness). Sequential -- single GPU.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 1: NTU PRETRAINING START $(date) ====="
echo "----- [1/4] supLP120 (pure supervised) -----"
python scripts/pretrain/pretrain_supLP120.py

echo "----- [2/4] mae (geodesic reconstruction) -----"
python scripts/pretrain/pretrain_mae.py

echo "----- [3/4] supMAE (hybrid reconstruction+supervision) -----"
python scripts/pretrain/pretrain_supMAE.py

echo "----- [4/4] supcon (supervised contrastive) -----"
python scripts/pretrain/pretrain_supcon.py

echo "===== STAGE 1 PRETRAINING DONE $(date) ====="
echo "Checkpoints: trained_models/{SUPERVISED,MAE,SUPMAE,ContrastiveNTU}/"
