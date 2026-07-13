# tasks.md — Publishability work plan (THMS submission)

_Written 2026-07-09. Owner: executing model (read this whole file first). Decisions locked by the human: **venue = IEEE THMS**, **Xsens 5-subject data WILL be released publicly**, **live subject tests approved if time-efficient** (T7)._

## Read first (context for the executing model)
1. `SESSION_HANDOFF.md` → `wiki/index.md` → open only pages a task needs.
2. `paper/paper_results.md` is the **source of truth for all numbers**. Never write a number into paper text without tracing it to a `trained_models/` summary or `RESEARCH_LOG.md` §B.
3. Conventions: `source .venv/bin/activate` (NOT conda); long runs via `nohup`, detached, logs in repo root; new output dirs only — never overwrite existing run dirs; after any change update affected wiki pages + `wiki/index.md` + append to `wiki/log.md`; numbers (not conclusions) to `RESEARCH_LOG.md` §B.
4. Do-nots: do NOT stage `external_data/` (8.2G); do NOT regenerate `Data_Processed/` unless a parser changes; do NOT resurrect swing; do NOT re-run α-sweep.
5. Launching the runs listed here is pre-approved by the human via this file. Commits still need an explicit go.
6. After every paper `.md` edit: regenerate tex via `python3 paper/md2tex.py`.

## Background (why these tasks)
The 2026-07-09 publishability review identified the remaining desk-reject risks: (a) six scattered claims instead of one thesis — fix = the five-setting table (T0); (b) several load-bearing results still single-seed (T1); (c) the contrastive objective (SupCon) is named in the abstract but absent from every table (T2); (d) the gap ordering underlying the whole story is asserted, not measured (T3); (e) THMS venue needs a human-in-the-loop artifact (T7); (f) reproducibility package (T6).

Multi-seed status inventory (what T1/T2 must complete):
- **3-seed DONE**: R1 cross-domain, R2 McNemar, R4a ECE, R4b AUC/convergence, R6 CZU-skel, R6b CZU-IMU-quat, R6c dual mode, R6d UTD.
- **Single-seed (seed 42) — needs 43/44**: R4c A2 subject-scaling; R6c raw-only mode; R6c 20-ch DIAL rung.
- **No seeds needed**: CRC baselines (deterministic given splits); R1 NTU linear probes (single pretrain per objective — multi-seed *pretraining* is deferred to T8).

---

## T0 — Writing: the one-thesis five-setting table + reframe propagation (no compute, do first)

### T0.1 Install the five-setting synthesis table in `paper/paper_results.md`
Add a new subsection at the END of R6 (suggest heading `### R6e — Synthesis: the five-setting map`), containing this table (verify every number against the R2/R6/R6b/R6c/R6d sections of the same file before installing; if any conflict, the existing section wins and flag it):

| # | Setting (source → target) | Gap | Target representation | Best prior (Δ vs scratch) | Negative-transfer case |
|---|---|---|---|---|---|
| 1 | NTU → CZU skeleton (R6) | small (same modality) | native skeleton | supLP120 +7.2 pp @k0 (p=.0004) | none (mae positive, n.s.) |
| 2 | NTU → UTD skeleton (R6d) | small (same modality) | native skeleton | supLP120 +7.4 pp @k0 (p<.0001) | none (mae +1.8 pp, p=.011, positive) |
| 3 | NTU → Xsens, position-derived quats (R2) | middle (cross-device, skeleton-like target) | strong (mocap-grade quats) | supMAE (McNemar +200/+76/+131 @k0/1/3, p≤.0065) | **mae** (−334/−217/−92, p<.001) |
| 4 | NTU → CZU IMU, orientation-only (R6b) | large (cross-modal) | weak (Madgwick quats, yaw drift) | none (supMAE ≈ scratch, 25/42 p=.28; supMAE > supLP120 36/44 p<.0001) | **supLP120** (−2.3 pp, p≈.04–.06) |
| 5 | NTU → CZU IMU, dual raw (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (17/39, p=.52) | **supLP120** (6/41, p<.0001) |

Framing paragraph to accompany it (adapt, don't copy blindly): *No pretraining objective is unconditionally safe across the skeleton→wearable gap. Both the winning prior and the actively harmful one move as the gap widens and the target representation strengthens: supervised wins at small gap, hybrid at mid gap, nothing at large gap; the negative-transfer case is pure reconstruction at mid gap and pure supervision at large gap. The prior's gap-invariant value is calibration and cold-start data-efficiency (R4).* 

Honest caveat to include: the "middle" placement of NTU→Xsens is currently justified narratively (v2 target is position-derived through NTU's own bone construction, hence skeleton-like — but different device/subjects/vocabulary; CKA < 0.02 shows the representational gap is still large). T3 replaces this narrative ordering with a measured one — if T3 is done first, cite its numbers here instead.

### T0.2 Propagate the reframe
- `paper/paper_intro.md` §5: contribution 2 currently claims mae-negative-transfer "inverts the field" as a general finding. Scope it: it is the **mid-gap** finding; the general claim is the five-setting map (merge with/adjust contribution 6, which already covers the arc). Keep exact numbers.
- `paper/paper_discussion.md` §1: open with the five-setting map as the unifying result; the existing R2 and R6 paragraphs become its supporting detail. Explicitly address the "culprit swap" (mae hurts only at rung 3, supLP120 only at rungs 4–5) as a *finding*, not a buried inconsistency: the failure mode itself is gap-dependent.
- `paper/paper_abstract.md`: do NOT rewrite yet (tables incomplete until T1/T2 land) — add a TODO comment at the top of the file noting the abstract must be re-checked after T1/T2 (the +27 pp A2 number and the "supervised, reconstruction, contrastive, or hybrid" list are both affected).
- Regenerate tex.

### T0.3 Backbone-naming audit
All reported results use `KinematicEncoder` (`src/models/kinematic_encoder.py`), NOT DSTformer. `wiki/results/phase1-mcnemar-ece-cka.md` line ~50 mislabels CKA layers as "L0–L2 = DSTformer blocks" — fix to "L0–L2 = KinematicEncoder Transformer layers" (verify against `temp_cka_analysis.py` first). Grep the whole `paper/` folder + `wiki/` for `DSTformer` and confirm every remaining mention is either (a) the early-experiments note or (b) `paper_method.md`'s correct "exists in codebase, used in early experiments only" line. Fix any that imply reported results used it.

---

## T1 — Multi-seed completion runs (launch immediately; nights of compute)

General pattern: follow `temp_czu_multiseed_run.sh` (chain runs serially under one nohup; verify flags byte-identical to the seed-42 launch so results pool; new out-dirs suffixed `-seed43`/`-seed44`). Pool with the same recipe as `temp_czu_multiseed_analyze.py` (per-fold sign tests over (subject,k,seed) cells + paired-t per k).

### T1.1 A2 subject-scaling, seeds 43/44
- `temp_a2_run.sh` already takes `--base-seed 42` → clone the invocation with 43/44, out-dirs `trained_models/A2-subjectScaling-seed{43,44}/`. nohup, log `a2_multiseed.log`.
- On completion: pool 3 seeds; recompute the R4c table (prior benefit vs N, per k). Update `paper/paper_results.md` R4c (drop the "single seed" caveat, or keep it honestly if the k=1 N=1 spike doesn't survive), `wiki/results/a2-subject-scaling.md`, RESEARCH_LOG §B.
- **Watch**: the headline "+26.9 pp @k3,N1" is in the abstract, intro, and discussion. If the pooled number moves, update ALL of them (grep for `26.9` and `+27`).

### T1.2 R6c raw-only mode, seeds 43/44
- `temp_czu_dualbranch.py` already has `--seed/--out-root/--raw-dir/--raw-dim`. Run raw mode only (scratch prior) at seeds 43/44 → `trained_models/CZU-IMU-DUAL-seed{43,44}/raw_scratch/` (check the dir doesn't already exist — the seed-43/44 multiseed run may have covered dual mode only).
- On completion: pool; update the raw row in R6c + `wiki/results/czu-imu-dual.md` (the "raw/quat single-seed" caveat and the `dual/scratch > raw/scratch 6/12 single-seed` line).

### T1.3 DIAL 20-ch rung, seeds 43/44
- `temp_czu_dial_run.sh` pattern with `--seed 43/44` → `trained_models/CZU-IMU-DIAL/mag20-seed{43,44}/`.
- On completion: pool; update the dose-response table in R6c + `wiki/results/czu-imu-dual.md` (drop or sharpen the "single-seed; steep-then-flat" caveat).

---

## T2 — SupCon full parity (the biggest table hole; abstract already promises "contrastive")

Checkpoint exists: `trained_models/ContrastiveNTU/supcon_epoch_50.pth`. NTU pretraining is source-side, so the existing checkpoint is valid for v2 (the old "SupCon ran under local preprocessing" caveat applies to the *downstream* runs, which are all being redone here). Expectation from the earlier evidence: SupCon may underperform — that is fine and publishable; the point is completing the objective family multi-seed.

### T2.1 Add supcon to the harness
- `temp_loso_fulltrain_calibration.py` METHODS dict (~line 117–147): add entry `tag: "supcon"`, `ckpt: trained_models/ContrastiveNTU/supcon_epoch_50.pth`. Verify strict weight loading (encoder keys match KinematicEncoder — check with `verify_models.py` or a dry load first).
- Same addition wherever priors are enumerated: `temp_czu_dualbranch.py`, the CZU/UTD run scripts, `temp_dump_posteriors.py`, `temp_cka_analysis.py`, `temp_controller_robust.py`. Grep each for `supLP120` to find the enumeration points.

### T2.2 SupCon runs (chain serially, one nohup, log `supcon_parity.log`)
Priority order (stop-and-report after each lands):
1. **NTU linear probe** (R1 within-domain row) — check whether a supcon probe number already exists in `wiki/experiments/ntu-pretraining.md` / RESEARCH_LOG; if not, 30-epoch probe, cheap.
2. **LOSO-v2, seeds 42/43/44, k∈{0,1,3}** → `trained_models/LOSO-fullTrainCalibrate-v2-supcon-seed{42,43,44}/` (or mirror the existing dir layout — inspect it first). Feeds R1 cross-domain + R4b AUC/convergence.
3. **Posterior dumps + McNemar + ECE** over the new ckpts (`temp_dump_posteriors.py` → `temp_analyze_calibration.py`) → extends R2 + R4a.
4. **CKA** (`temp_cka_analysis.py`) → extends R3.
5. **Controller robust** (`temp_controller_robust.py` re-run including supcon posteriors) → extends R5 tables.
6. **External datasets, 3 seeds each**: CZU skeleton (`temp_czu_loso.sh` pattern), CZU IMU quat (`temp_czu_imu_loso.sh` pattern), CZU dual (`temp_czu_dual_run.sh` pattern), UTD (`temp_utd_run.sh` pattern) → extends R6/R6b/R6c/R6d.
7. **A2** with supcon (piggyback on T1.1 launches if timing allows).

### T2.3 Paper/wiki updates after each stage
Every table gains a supcon column/row; every "four inits/objectives" phrase becomes five (grep `paper/` and `wiki/` for `four`). Update the five-setting table (T0.1) if supcon changes any best/worst cell. New numbers → RESEARCH_LOG §B. The discussion §3 "Contrastive is under-sampled" limitation bullet gets rewritten (or deleted) once this lands.

---

## T3 — Measure the gap axis (CKA per target; cheap, high leverage)

The five-setting table's gap ordering (small/middle/large) is currently narrative. Make it measured:
- Extend `temp_cka_analysis.py` to compute layer-wise linear CKA of each pretrained encoder between NTU and EACH target: Xsens-v2 (exists), CZU skeleton, CZU IMU quats, UTD skeleton. Inference-only, no retraining. Output one table: target × encoder × layer.
- Success criterion: CKA (or a summary of it) orders the targets consistently with the claimed small→middle→large ranking. If it does NOT, report the numbers honestly and flag for the human — do not force the narrative.
- Install as the gap column in the T0.1 table + a short method note in `paper_method.md`; update `wiki/concepts/domain-gap-metrics.md`.

## T4 — UTD CRC published-baseline anchor (small)
Mirror `temp_czu_crc_baseline.py` on UTD skeleton splits (statistical moments + CRC-RLS, byte-identical LOSO k-shot splits) → literature anchor row for R6d, same as R6 has. Deterministic, no seeds. Also cite UTD-MHAD's published cross-subject accuracy as the external reference point. → R6d table + `wiki/results/utd-skeleton-loso.md`.

## T5 — Cold-start on a strong external target (deployment claim de-risk)
The prior's deployment value claim (A2: helps most at N=0/1 enrolled subjects) is Xsens-only. Test it on CZU dual (strong target):
- Add `--n-train-subjects` to `temp_czu_dualbranch.py` (copy the nested-first-N pattern from `temp_loso_fulltrain_calibration.py`; N=0 short-circuits to pretrained init + k-shot calibration only).
- Run dual mode, N∈{0,1,2,3}, priors {scratch, supLP120, supMAE}, seed 42 first (extend to 3 seeds if signal) → `trained_models/CZU-DUAL-subjectScaling/`.
- Question answered: does ANY prior beat scratch at N=0/1 on a strong raw-signal target? Either outcome is publishable — if yes, the cold-start claim generalizes; if no, scope the A2 claim to representation-poor targets honestly.
- → new wiki page + a paragraph in R6c or R4c.

## T6 — Data + code release package (human decided: YES on data)
1. Xsens dataset: export the released form (raw mvnx? processed v2 quats? — **ask the human which layers to release**), de-identify (subject IDs → S1–S5, strip any personal metadata in mvnx headers), write a datasheet README (capture protocol, 22 gestures, 5 subjects, sensor placement, license — suggest CC-BY-4.0).
2. Code: the pipeline lives in `temp_*` scripts (see `wiki/code/temp-scripts.md`). For release: promote the load-bearing ones into a `scripts/` or `release/` layout with a top-level README mapping each paper table → producing script → run command. Do NOT delete the `temp_*` originals until the human confirms.
3. Host: Zenodo or OSF for data (DOI), GitHub for code. Draft the availability statement for the paper.
4. Gitignore `external_data/` FIRST (still not done — see SESSION_HANDOFF owed-commit note).

## T7 — Live human-in-the-loop study (the THMS-earning artifact) — HUMAN-GATED, spec below is for discussion, not execution
Purpose: convert the simulated controller (R5) into a small live validation. Claims to validate with humans: (i) recognition differences compound into mission-level differences; (ii) calibration governs the safety/throughput trade at iso-safety operating points.

Minimal design (est. 3–4 weeks elapsed):
- **N = 5–6 NEW subjects** (never seen by any model → true LOSO cold-start; doubles as dataset expansion for the T6 release, 5 → 10–11 subjects).
- **Protocol per subject** (~60–90 min): suit up (Xsens) → record k=3 calibration shots per gesture live → on-device calibration → perform the 12-step pick-place mission from `temp_controller_sim.py` against a **screen-based simulated executor** (no physical robot — keeps cost near zero and matches the abstract-controller framing already defended in the paper).
- **Conditions, within-subject, counterbalanced**: 2 inits (supLP120 = best-calibrated vs scratch = no-prior) × 2 operating points (ungated τ=0 vs iso-safety τ* at 1% false-activation budget) = 4 conditions, ~3 mission repetitions each.
- **Metrics**: mission success, time-to-completion, false activations, rejections, corrective commands; NASA-TLX per condition (subjective workload — cheap and very THMS).
- **Analysis**: paired within-subject; step-level McNemar across repetitions for power at small N.
- **Prereqs to schedule**: real-time inference path (Xsens MVN stream → position-derived v2 quats → encoder; ~1 week build + pilot), IRB/ethics check (gesture studies are typically expedited/exempt — confirm with the institution BEFORE building).
- **First step for the executing model**: write a one-page protocol doc (`paper/live_study_protocol.md`) fleshing this spec for the human to approve; do not build until approved.

## T8 — DEFERRED (explicit human decision: later, after tables are complete)
Full reproducibility rerun: re-run every table from a clean checkout via one orchestrator script, including multi-seed NTU *pretraining* (currently every objective has a single pretrained checkpoint — pretraining seed variance is unquantified). This supersedes/absorbs the per-table patching above. Do NOT start this now.

## T9 — OPTIONAL (only if time remains before submission)
Second-backbone replication of the main Xsens table using the already-implemented `DSTformerQuatEncoder` (`src/models/dstformer_quat_encoder.py`, drop-in per `wiki/code/models.md`): pretrain on NTU with the 5 objectives, run LOSO-v2 3 seeds. Kills the "single backbone" limitation. Heavy compute; discuss with the human before launching.

---

## Suggested execution order
1. **Day 1**: launch T1.1–T1.3 (nohup chain) + T2.1/T2.2 stages 1–2 (second nohup chain, GPU permitting) → compute runs overnight while doing T0 (writing) and T3 (inference-only).
2. **Days 2–4**: T2.2 stages 3–6 as the LOSO runs land; T4; fold results into paper/wiki as each lands.
3. **Parallel, human-paced**: T6 (release prep), T7 protocol doc → human approval → build.
4. **Last**: abstract rewrite (T0.2 TODO) once all tables are final.

## Reporting discipline
After each task: update the affected wiki page(s) + `wiki/index.md`, append `## [YYYY-MM-DD] update | <title>` to `wiki/log.md`, numbers to `RESEARCH_LOG.md` §B, regenerate tex if paper text changed. Mark the task done in THIS file with a one-line outcome (numbers, not adjectives).
