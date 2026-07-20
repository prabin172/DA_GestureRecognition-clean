#!/usr/bin/env bash
# Master orchestration: the entire reproducible rerun, start to finish, one nohup process.
# Stages are sequential (single GPU) and each is idempotent-ish (re-running a finished stage
# just retrains over existing output dirs -- no stage deletes another's output). If this dies
# partway, inspect the log for the last "===== STAGE N" line and resume by commenting out
# completed stages below, or by re-running the single stage script directly.
#
# Usage: nohup bash scripts/orchestration/RUN_FULL_RERUN.sh > full_rerun.log 2>&1 &
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="scripts/orchestration"

echo "===== FULL RERUN START $(date) ====="

# Stages 0-7 already completed 2026-07-13/15 (see full_rerun.log) -- do not re-run, see SESSION_HANDOFF.md.
# bash "$HERE/00_data_pipeline.sh"
# bash "$HERE/01_pretrain.sh"
# bash "$HERE/02_main_loso.sh"
# bash "$HERE/03_main_analysis.sh"
# bash "$HERE/04_a2_subject_scaling.sh"
# bash "$HERE/05_oov.sh"
# bash "$HERE/06_czu_external.sh"
# bash "$HERE/07_utd_external.sh"
bash "$HERE/08_controller.sh"
bash "$HERE/09_czu_cold_start.sh"

echo "===== FULL RERUN ALL DONE $(date) ====="
