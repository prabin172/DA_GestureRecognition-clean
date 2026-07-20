# Legacy orchestration scripts

These are the original repo's orchestration scripts, carried over for reference only.
**Do not run them in this repo** — they were written for an incremental history that doesn't
apply here:
- Hardcoded `cd /home/ptimilsina/projects/DA_GestureRecognition` (the OLD repo's path) in most
  of them, or `cd "$(dirname "$0")"` (cd's into `scripts/orchestration/` itself, not repo root)
  in `a2_run.sh`/`phase1_run.sh` — both would fail immediately here.
- Most default to 4 methods (`scratch,supLP120,supMAE,mae`) with supcon bolted on later via
  separate `-supcon-seed*` output dirs (`t2_supcon_run.sh`, `t2b_followon_run.sh`) — this repo's
  rerun trains all 5 methods together from the start (see `../0*_*.sh`), so that incremental
  patching structure doesn't apply.
- `czu_loso.sh` waits on an `alpha_sweep.log` marker from a dropped, dead experiment (the
  α-sweep) — that wait-loop would hang forever here.

Kept because they document the exact flags/methodology used for the original numbers cited in
`paper_results.md` — useful as a cross-check, not as something to execute. The actual rerun
path is `scripts/orchestration/00_data_pipeline.sh` through `09_czu_cold_start.sh`, chained by
`RUN_FULL_RERUN.sh`.
