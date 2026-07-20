#!/usr/bin/env bash
# Stage 5: OOV / leave-class-out few-shot onboarding (borderline load-bearing -- has
# numbers in RESEARCH_LOG.md but no dedicated paper_results.md table; kept in scope
# per human decision during the DA_GestureRecognition-clean migration).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source .venv/bin/activate

echo "===== STAGE 5: OOV LEAVE-CLASS-OUT START $(date) ====="
python scripts/main_experiment/loso_leave_class_out_fewshot.py --methods "scratch,supLP120,supMAE,mae,supcon"
echo "===== STAGE 5 OOV DONE $(date) ====="
