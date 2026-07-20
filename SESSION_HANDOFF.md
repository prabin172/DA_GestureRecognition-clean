# Session Handoff — DA_GestureRecognition-clean

_Created 2026-07-13. This is the clean reproducibility rerun of `RTHMLab/DA_GestureRecognition`
(branch `swing-mode-xsens`), not a fork of its git history — fresh repo, verified data only._

## State — full rerun COMPLETE (2026-07-15) + paper_results.md reconciled against it (2026-07-20)

This repo exists to produce final, verified-reproducible numbers for the paper, replacing the
original repo's ~50-script, 234GB, partially-undocumented working state. What's here:

- **`scripts/`** — 35 load-bearing scripts (of ~50 in the original), reorganized by pipeline
  stage (was flat `temp_*.py` in repo root), every root-path resolution numerically verified.
  Two real bugs found and fixed during migration (not before): near-duplicate pretrain scripts
  writing unused checkpoints, a hardcoded machine-specific absolute path. See
  `scripts/orchestration/README.md` for the 10-stage pipeline this repo runs.
- **`src/`** — pruned shared library (models/, data/); dead DANN/DSTformer-early-experiment/
  label_encoder code dropped.
- **Data** — `Data_Processed/ntu_quats/` (976MB) and `DataCollection/` (6.8GB) copied and
  SHA256-verified byte-identical against `migration/*.sha256`. `external_data/` (8.3GB)
  copied, size-verified. **`Data_Processed/ntu_quats` is irreplaceable** — the raw
  `NTU-SkeletalData/` source no longer exists anywhere on the original machine.
- **`requirements.txt`** — frozen (90 packages, Python 3.12.3) — did not exist anywhere before
  this migration, a real reproducibility gap now closed.
- **`wiki/`, `paper/`, `tasks.md`, `RESEARCH_LOG.md`** — copied from the original repo, wiki
  script-name references updated to match this repo's layout. `paper/paper_results.md` is
  still the *original* repo's numbers (single/few-seed, incremental supcon patches) — this
  repo's rerun will produce the numbers that should eventually replace them there.
- **`wiki/concepts/controller.md`** — corrected a wrong claim (prototype's numbers ARE cited
  in R5, not "not cited as evidence" as the page previously said).
- **A2 reproducibility gap closed** — `scripts/main_experiment/analyze_a2_multiseed.py` is a
  new, committed, verified script (matches the original's uncommitted "ad hoc" pooling output
  byte-for-byte) — see `RESEARCH_LOG.md` §B8 in the original repo.

## What's running right now

`scripts/orchestration/RUN_FULL_RERUN.sh` launched via nohup at **2026-07-13 10:41 CDT**
(PID 2161288, parent chain → `systemd --user` — confirmed detached, survives logout) — **5
methods (scratch, mae, supMAE, supLP120, supcon) trained together from the start** (not bolted
on incrementally like the original repo's history), 3 seeds throughout, all external datasets
(CZU-MHAD, UTD-MHAD), the full controller pipeline. Log: `full_rerun.log` in repo root. Stage 0
is commented out in `RUN_FULL_RERUN.sh` (already completed in attempt 1, do not re-run).

**Attempt 1** (2026-07-12 23:53 CDT, PID 2024853) completed Stage 0 then died 2s into Stage 1 —
`src/data/action_names.json` was missing from the migration (non-`.py` data file, missed by the
script inventory). Copied it in (byte-identical to original). Log kept as
`full_rerun.log.stage0_and_failed_attempt1`.

**Attempt 2** (2026-07-13 09:10 CDT, PID 2136371) got past that, ran `pretrain_supLP120.py`'s
`all120` mode to completion (epoch 50/50, checkpoint saved), then died moving to its `relevant23`
mode — `src/data/ntu_relevant_action_ids.json` also missing. **My first audit of missing
`src/data/`/`src/models/` files was wrong**: I'd confirmed `imu_to_ntu_action_map.json`,
`ntu_relevant_action_ids.json`, `imu_loader.py`, `dstformer_quat_encoder.py` were all unused by
grepping the DANN-era scripts, but `ntu_relevant_action_ids.json` turned out to be used inside
`pretrain_supLP120.py` itself (`RELEVANT_JSON_PATH`, line 35) — a script that *does* survive in
this repo. Lesson: grep the actual surviving pipeline scripts for the literal filename, not just
the old dead-code scripts. Copied `ntu_relevant_action_ids.json` in (byte-identical to original).
Re-swept all of `scripts/`+`src/` for every literal `.json`/`.npy`/`.npz`/`.pkl`/`.csv`/`.txt`
path reference afterward — everything else matched is a script writing its own output (csv
results tables), not a missing input. `imu_to_ntu_action_map.json` and `imu_loader.py` are still
confirmed zero-reference and correctly left out. Log kept as
`full_rerun.log.attempt2_died_relevant23_missing`.

**Attempt 3** — relaunched after the fix, confirmed healthy. **Stage 1 (NTU pretraining)
completed cleanly at 11:48 CDT** — all 4 objectives (supLP120 incl. the previously-broken
`relevant23`, mae, supMAE, supcon), no errors. **Stage 2 (main Xsens LOSO) completed cleanly at
13:58:18 CDT** — all 5 methods (scratch, mae, supMAE, supLP120, supcon) x 3 seeds (42/43/44) x
k{0,1,3}, no tracebacks. This is well past where attempts 1 and 2 died, so both migration gaps
(`action_names.json`, `ntu_relevant_action_ids.json`) are confirmed fixed. **Stage 3 (main
analysis) crashed immediately** on its first substep (`dump_posteriors.py`):
`ModuleNotFoundError: No module named 'scripts'`. Root cause — 3 of the 5 Stage 3 scripts had a
broken copy of the `sys.path.insert(PROJECT_ROOT)` pattern that Stage 1/2 scripts use correctly
(pattern: define `PROJECT_ROOT`, insert into `sys.path`, *then* import `scripts.*`/`src.*` —
these had the insert placed after the import, or missing `import sys` entirely, or missing the
insert altogether):
- `scripts/main_experiment/dump_posteriors.py` — no `import sys`; insert was after the
  `scripts.main_experiment...` import that needed it. Fixed.
- `scripts/main_experiment/cka_analysis.py` — no `import sys`; insert was after the `src.models...`
  import. Fixed.
- `scripts/main_experiment/raw_domain_gap.py` — no `sys.path.insert` at all. Fixed.

**Attempt 4** — relaunched (Stages 0-2 commented out in `RUN_FULL_RERUN.sh`, they don't need
re-running). Stage 3 got further (posteriors, ECE/calibration, CKA both single- and
multi-target all completed) then died at substep 4/5, `mmd_domain_gap.py`:
`FileNotFoundError: Data_Processed/imu_quats/index.csv`. Root cause — this script still had the
**stale v1 data path** (`Data_Processed/imu_quats`) hardcoded as its `--xsens-root`/`--xsens-index`
default; every sibling Stage-3 script had already migrated to v2 (`imu_quats_v2`) but this one
didn't. Fixed the defaults to `imu_quats_v2`. Also found `03_main_analysis.sh` was calling this
script *without* `--include-supcon` — since this rerun trains supcon as a first-class method
from the start (unlike when that flag was added, when supcon checkpoints weren't guaranteed to
exist), added `--include-supcon` to the Stage 3 invocation so the MMD table isn't silently
missing supcon while CKA/posteriors include it. Both fixes smoke-tested clean
(`--smoke`/`--smoke --include-supcon`) before relaunching.

**Attempt 5** — relaunched **2026-07-13 17:09:48 CDT, PID 2613100**. **Stage 3 completed
cleanly at 17:13:13 CDT** — all 5 substeps (posteriors dump, ECE/McNemar, CKA x2, MMD w/
supcon, raw domain gap), confirming both Attempt-3 and Attempt-4 fixes hold. **Stage 4 (A2
subject-scaling) completed cleanly at 2026-07-14 00:17:37 CDT** (full 3-seed x N{0..4} sweep
+ pooled analysis, output in `trained_models/A2-subjectScaling-pooled`). **Stage 5 (OOV) died
instantly** — `loso_leave_class_out_fewshot.py:58` had the stale v1 path
(`Data_Processed/imu_quats` hardcoded as `IMU_DIR`; same bug class as Attempt 4's
`mmd_domain_gap.py`, and ironically line 41 of the same file finds project root by checking
for `imu_quats_v2`). Fixed to `imu_quats_v2`. Swept all Stage 5-9 entry points for the same
class — clean; `loso_fulltrain_calibration.py`'s v1 fallback (line 70) is never exercised
(every orchestration call exports `LOSO_IMU_DIR` explicitly). Log kept as
`full_rerun.log.attempt5_died_stage5_oov_v1path`.

**Attempt 6** — relaunched after the path fix (Stages 0-4 now commented out), died instantly
on the *next* latent Stage-5 bug: `ValueError: Unknown method supcon` —
`loso_leave_class_out_fewshot.py`'s `METHOD_CONFIGS` predates supcon-as-first-class (same
"supcon bolted on later" class as Attempt 4's missing `--include-supcon`), while `05_oov.sh`
passes `--methods "scratch,supLP120,supMAE,mae,supcon"`. Added a `supcon` entry mirroring
`loso_fulltrain_calibration.py`'s (ckpt `trained_models/ContrastiveNTU/supcon_epoch_50.pth`,
verified present). `--dry-run` with all 5 methods then resolved clean (550 base + 2200
calibration runs, all subject/label counts OK). Log kept as
`full_rerun.log.attempt6_died_stage5_supcon_missing`.

**Attempt 7** — relaunched **2026-07-14, PID 4147191** (`bash
scripts/orchestration/RUN_FULL_RERUN.sh`). Stages 5, 6, 7 (OOV, CZU external, UTD external) all
completed cleanly (Stage 5 done 2026-07-14 23:46:42, Stage 6 done 2026-07-15 08:32:30, Stage 7
done 2026-07-15 10:28:21). Stage 8 (controller) started 10:28:21 and crashed almost immediately:
`KeyError: 'supcon'` in `scripts/controller/controller_sim.py` — same "supcon bolted on later"
bug class as attempts 4/6, this time in the plotting `colors` dict (only had `scratch`, `mae`,
`supMAE`, `supLP120`; `METHODS` includes `supcon`). Fixed by adding `"supcon": "tab:red"`.
Checked the sibling Stage 9 script `controller_robust.py` for the same bug — already had
`"supcon": "tab:purple"`, no fix needed there. Process had died (not just the terminal
disconnecting); log left as historical record, not renamed.

**Attempt 8 (final)** — relaunched **2026-07-15 13:21:03 CDT** (Stages 0-7 commented out in
`RUN_FULL_RERUN.sh`, only Stage 8 + Stage 9 active). **Stage 8 (controller) completed cleanly at
13:24:04 CDT** — confirms the `supcon` colors-dict fix held, no errors. **Stage 9 (CZU-DUAL
cold-start) completed cleanly at 15:08:10 CDT**. `===== FULL RERUN ALL DONE Wed Jul 15 03:08:10
PM CDT 2026 =====`. Zero tracebacks/errors anywhere in `full_rerun.log.attempt8`. Verified output
exists: `trained_models/Phase3-controller/robust/*` and
`trained_models/CZU-DUAL-subjectScaling/N3/dual_supcon/summary.csv` both present. **All 10
stages (0-9) of the reproducibility rerun are now done.** Log: `full_rerun.log.attempt8`.

**Linger enabled** (`loginctl enable-linger ptimilsina`, was `Linger=no`) — this was NOT set
before this session despite earlier handoff notes claiming "survives logout": parent-chain
being `systemd --user` only keeps a process alive across *closing a terminal*, not a *full
logout*, unless linger is on. Without this fix the whole run would have died when the human
logged out. Now confirmed `Linger=yes` — safe to leave unattended through logout. Still does
not survive a machine reboot.

**Check progress:** `tail -f full_rerun.log` or `grep "=====" full_rerun.log` for stage
boundaries. Each stage script also echoes its own start/done timestamps. Dead-end logs from
attempts 1-4 kept as `full_rerun.log.stage0_and_failed_attempt1`,
`full_rerun.log.attempt2_died_relevant23_missing`, `full_rerun.log.attempt3_died_stage3_moduleimport`,
`full_rerun.log.attempt4_died_stage3_mmd_v1path`, `full_rerun.log.attempt5_died_stage5_oov_v1path`,
`full_rerun.log.attempt6_died_stage5_supcon_missing`.

**If it's still running:** don't touch `trained_models/` while it runs. Check which stage is
active before doing anything else.

**If it died again:** find the last completed `===== STAGE N ... DONE` line, then either
re-launch `RUN_FULL_RERUN.sh` after commenting out completed stages (0-4 are now commented out —
comment out further stages as they complete), or run the next stage's script directly
(`bash scripts/orchestration/0N_*.sh`). Two bug classes have bitten twice now, check both before
assuming it's a new problem:
1. **Missing migration files** — grep the *literal filename* across all of `scripts/`+`src/`
   (surviving pipeline scripts can reference an old data file even if DANN-era scripts don't,
   as `ntu_relevant_action_ids.json` did). Only treat a file as safe-to-skip once grep for its
   exact name turns up zero hits.
2. **Broken `sys.path`/stale-path copy-paste** — several Stage 3+ scripts are near-duplicates of
   each other (and of the Stage 1/2 pattern) with small drifted bugs (import order, stale v1
   paths, missing flags). Before re-running, diff the failing script's import block and default
   paths against a sibling script that already works, don't just retry.

**Known unknowns going in** (verified plumbing, not full correctness): stage 0's first script
smoke-tested clean against real data; the main LOSO script's `--dry-run` resolved correct
checkpoint paths for all 5 methods.

## Next (continue here)

1. **DONE 2026-07-20: paper_results.md reconciled against this repo's fresh rerun.** Full
   number-by-number comparison (2 subagents + direct checks), root-cause investigation of every
   discrepancy (no bugs found — training non-determinism, `cudnn.deterministic=False` everywhere,
   first-ever pinned-environment run), and a full rewrite of `paper/paper_results.md` R1–R6e +
   regenerated `paper/*.tex` + propagated to 10 wiki pages. Five claims genuinely broke and were
   honestly softened/rewritten (not silently kept); R5's controller section required the deepest
   rework (Lock 1 says mae worst, Locks 2/3 now say supLP120 worst under harsh penalties — a real,
   explained divergence, not a bug). Full account: `wiki/log.md` 2026-07-20 entries (3 of them).
   **Two loose ends, both running in background as of this writing** (`multiseed_extension.log`,
   ~1.5 days ETA): OOV leave-class-out and CZU-dual cold-start (T5) were promoted from single-seed
   to 3-seed. Once done: pool the new seeds, update [[czu-dual-cold-start]] and
   `paper_results.md`'s R6c cold-start paragraph (currently still single-seed numbers with a
   pending-confirmation note) and [[oov-leave-class-out]]'s table.
2. **Old repo deleted locally 2026-07-16** — `~/projects/DA_GestureRecognition` no longer exists;
   this is the sole active repo. GitHub remote (`RTHMLab/DA_GestureRecognition`) deliberately left
   alone, pending human check-in with the team — do not delete it without explicit instruction.
3. **notes.md** (repo root) — a suspicious "SYSTEM DIRECTIVE"-style file found in the old repo
   before deletion, copied here for human review. Verified against repo state 2026-07-20: only its
   §1 (terminology scrub) was ever executed; §2/§4 (new multi-gap controller experiments + R5
   rewrite) were not, and were not acted on this session (flagged as possibly-injected content,
   never verified as coming from the human or collaborators) — awaiting explicit human direction.
4. **T6 (data release), T7 (live study)** — see `tasks.md` and `paper/live_study_protocol.md`;
   not started, decisions live with the human.

## Owed / do-nots

- **Do not** overwrite `trained_models/Phase3-controller/robust/` — stage 8 is done, this is now
  the locked output future analysis cites.
- **Do not** run anything in `scripts/orchestration/legacy/` — see that folder's README.
- **Do not** re-run stage 0 unless a parser changed — it's slow and nothing downstream needs it
  regenerated unless the processing logic itself changes (same rule as the original repo).

## Entry points
`scripts/orchestration/README.md` (the 10-stage pipeline) · `migration/MANIFEST.md` (data
provenance) · `wiki/index.md` · `tasks.md` · original repo: `~/projects/DA_GestureRecognition/`
(`SESSION_HANDOFF.md` there points back here)
