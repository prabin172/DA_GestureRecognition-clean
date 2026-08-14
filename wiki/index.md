---
type: index
status: active
updated: 2026-07-21
---

# Wiki Index

Start at [[overview]] (thesis + current state + where stuck). **This is now the sole active repo** —
the original `DA_GestureRecognition` (sibling repo) was deleted locally 2026-07-16 after this repo's
full 10-stage reproducibility rerun completed and superseded it; its GitHub remote
(`RTHMLab/DA_GestureRecognition`) still exists, deletion deferred pending team check-in.

**Latest state (2026-07-20): full paper_results.md reconciliation against this repo's independently
retrained checkpoints.** Every number in `paper/paper_results.md` (R1–R6e) was recomputed from this
repo's fresh rerun and compared against the original repo's numbers. Pattern found: `scratch`/CRC
numbers (no pretrained-checkpoint dependency) reproduce exactly; everything downstream of a retrained
NTU-pretraining checkpoint carries a 1–5pp noise floor from non-deterministic GPU training
(`cudnn.deterministic=False` throughout, by original design — this is the first time the pipeline has
ever run in a pinned, reproducible environment). Most claims held directionally; five did not survive
as originally stated and were rewritten/softened rather than kept: R2's supMAE-k=0 McNemar flip
(+200→−22, both n.s.), R3's supcon-CKA "sharpest instance" claim (now lower than supLP120 — CKA
de-prioritized in favor of raw MMD²/Frechet as primary domain-distance evidence, per human decision),
R6b's two headline significance claims (weakened to trends), and **R5's controller Lock 2/3 no longer
agree with Lock 1 on which method compounds worst** (supLP120, not mae, is now worst under harsh
critical-cost penalties — a real, mechanistically-explained effect tied to supLP120's known
confident-false-critical-activation mode, not a bug). Conversely, R4c (A2) and R5 Lock 1 reproduced
almost exactly, and R6c's supcon-hurts-on-strong-targets finding got *stronger*, not weaker. Full
account: `paper/paper_results.md`'s 2026-07-20 update note + each section's inline notes; wiki pages
below updated to match. **Two experiments were promoted from single-seed to 3-seed and launched**
(OOV leave-class-out, CZU-dual cold-start T5) — both in progress as of this writing, ~1.5 days ETA.

Prior state (2026-07-10, THMS publishability push): SupCon (5th objective) analyzed end-to-end across
R1–R6e; T1 multi-seed pooling (A2, R6c); T5 cold-start scoping check (single-seed then). See git
history / `wiki/log.md` for the full trail — superseded by the 2026-07-20 reconciliation above where
the two conflict. [[multiseed-loso-v2]] · [[czu-skeleton-loso]] · [[czu-imu-crossmodal]] ·
[[czu-imu-dual]] · [[utd-skeleton-loso]] · [[position-reconstruction-v2]] · [[open-questions]].

## Concepts
- [[lrq]] — Local Relative Quaternions, 17-segment model, padding; the shared representation
- [[loso-protocol]] — LOSO, k-shot calibration, OOV variant, the n=5 stats problem
- [[pretraining-objectives]] — MAE / supervised / SupMAE / SupCon / DANN / scratch + verdicts
- [[swing-mode]] — axial-twist removal: hypothesis, outcome, sub7/sub8 asymmetry
- [[domain-gap-metrics]] — MMD (flawed), CKA (per-encoder + **per-target: does NOT confirm the small/middle/large gap ordering, in either the original or 2026-07-20 rerun**), necessary-not-sufficient nuance, **+ raw method-independent gap: MMD²/Frechet on hand-crafted features DOES confirm the ordering exactly, unchanged by the checkpoint retrain (no pretrained encoder involved) — now the paper's primary domain-distance evidence, CKA de-prioritized per 2026-07-20 human decision**
- [[controller]] — full design doc for the C6 controller — task/FSM/cost-model mechanics, the 3 robustness locks explained, what the abstraction represents, + the planned (unapproved) live-study extension. **2026-07-16: terminology revised** — grasp/release/mission language replaced with System Input/Safety-Critical State/Sequential Control Task (no numbers changed)

## Data
- [[ntu-dataset]] — source: 43 490 clips, 120 classes, parser/loader, variants
- [[xsens-dataset]] — target: ~2 736 clips, 22 gestures, 5 subjects, currently swing mode

## Code
- [[models]] — KinematicEncoder/Decoder, DSTformer, LabelEncoder, checkpoint conventions
- [[data-pipeline]] — batch processors, parser, loaders
- [[pretrain-scripts]] — `src/scripts/pretrain/` catalog
- [[downstream-scripts]] — `src/scripts/downstream/` + `analysis/` (earlier generation)
- [[temp-scripts]] — root `temp_*` scripts; **the current main pipeline lives here**

## Experiments
- [[ntu-pretraining]] — the encoder zoo (which checkpoints exist, what's stale)
- [[loso-fulltrain-calibrate]] — **main experiment**, local + swing tables
- [[xsens-to-xsens-loso]] — Job 2: target-only frozen LOSO + its two critical caveats
- [[mmd-domain-gap]] — MMD tables local vs swing
- [[oov-leave-class-out]] — novel-class onboarding
- [[sanity-checks]] — NTU→NTU control, quat QA
- [[dann-experiments]] — all DANN variants (stale under swing)
- [[cleaned-source-pretraining]] — slerp035 cleaning (dead end)
- [[early-experiments]] — superseded first-generation runs

## Results & paper
- [[phase1-mcnemar-ece-cka]] — **Phase 1: per-clip McNemar (mae neg-transfer p<.001 every k, seed/checkpoint-stable; supMAE/supLP120/supcon only significant at k=3 now), ECE (supLP120 best-calibrated, supcon 2nd — most checkpoint-stable result in the paper), CKA (2026-07-20: supcon's "numerically highest" claim did not survive — now below supLP120; CKA de-prioritized, raw MMD²/Frechet is primary evidence)**
- [[a2-subject-scaling]] — **Phase 2: prior benefit vs #FT-subjects, 3-seed pooled — peaks N=0/1 (supLP120 +28.3pp @k3,N1, reproduces the original +27.5pp within noise), washes to ≈1–2pp by N=4; mae ≤0 from N≥2** (replaces dead gap-knob). **The most checkpoint-stable result in the paper besides ECE. Caveat: lever is target-representation-contingent, see [[czu-dual-cold-start]]**
- [[czu-dual-cold-start]] — **T5: does A2's cold-start lever survive on a strong target? No** — neither supLP120 nor supMAE beats scratch at any N on CZU-dual; both significantly worse at N=0,k=1. **3-seed extension launched 2026-07-20, in progress**
- [[phase3-controller]] — **Phase 3: controller on real posteriors, fully-randomized robustness protocol (120 random System Input assignments × 1000 Monte Carlo trials, shared by all 3 locks — no recall-based or otherwise fixed vocab anywhere). All three locks agree: mae compounds worst under every stress test. supLP120's secondary confident-misfire mode still shows up as a narrow nuance but never overtakes mae. supcon is the standout at k=1.**
- [[phase3-controller-crosssetting]] — **2026-07-21, now in the paper as R5b: Lock1+Lock2 only, reproduced across all 5 transfer settings with compatible posteriors (NTU→Xsens, CZU skeleton, CZU IMU orientation-only, UTD skeleton, CZU dual-raw — the last required an actual retrain, dualbranch.py never saved checkpoints before). Finding: the two locks agree with each other exactly where the setting's underlying recognition-level effect is large/significant (NTU→Xsens's mae, CZU dual-raw's supcon); they disagree exactly where the recognition-level effect is itself small/marginal. The controller amplifies real signal cleanly and amplifies noise into lock-disagreement — a stress-test harness, not an independent ranking. Also caught & fixed a stale paper_method.md §8 paragraph still describing the retired reliability-ranked assignment.**
- [[multiseed-loso-v2]] — **3-seed LOSO-v2 stats: accuracy + AUC + convergence; the load-bearing stats** (McNemar now paid in [[phase1-mcnemar-ece-cka]])
- [[czu-skeleton-loso]] — **CZU-MHAD skeleton→skeleton external validity (R6, pooled 3-seed + supcon, 2026-07-20 reconciled)** — supLP120 +5.0pp @k0 p=.0012 (was +7.2pp, still clearly significant); **supcon +5.2/+4.7/+3.4pp all k, p<.005, beats supLP120 itself p=.044**
- [[czu-imu-crossmodal]] — **CZU-MHAD skeleton→IMU TRUE cross-modal (R6b, 3-seed pooled + supcon, 2026-07-20: significance weakened to trends)** — direction unchanged (supLP120 trends worst, supMAE trends best) but neither clears p<.05 anymore; the significant version of this contrast now lives in [[czu-imu-dual]]
- [[czu-imu-dual]] — **CZU-MHAD dual-branch (R6c, dual pooled + supcon + cold-start, 2026-07-20: got sharper, not weaker)** (raw scratch ≈ CRC → representation was the bottleneck; on a strong target `dual/scratch` is now the best performer at every k — supLP120 worst p=.0015, **supcon worse still, p<.0001, nearly double the original effect size**; **cold-start 3-seed extension launched 2026-07-20, in progress**)
- [[utd-skeleton-loso]] — **UTD-MHAD skeleton→skeleton (R6d, + supcon, 2026-07-20 reconciled)** (2nd independent public dataset; supLP120 win holds at k=0/k=1 (+5.0pp p=.0014), k=3 now a trend; supLP120 vs supMAE lost significance (was p=.0003, now p=.20); **supcon is now the clear standout, beating supLP120 (was a wash) at +8.0pp @k0 p<.0001**)
- [[position-reconstruction-v2]] — **v2 position-derived Xsens: the real domain-gap fix** (supersedes swing)
- [[swing-mode-findings]] — synthesis of the 2026-07-01 swing runs
- [[publishability-review]] — the 8 problems + fix list (2026-07-01 audit)
- [[paper-framing]] — locked framing, 3 pillars, stats policy, controller pillar (mirrors RESEARCH_LOG §A)
- [[literature-landscape]] — Undermind deep-research report (2026-07-03): the gap is ours, four competitor clusters, THMS venue precedent

## Questions
- [[open-questions]] — TODO, blockers, standing puzzles

## Reproducibility rerun — COMPLETE (2026-07-13 to 2026-07-15, reconciled 2026-07-20)
This repo (`DA_GestureRecognition-clean`) is a from-scratch reproduction of the original
`DA_GestureRecognition` repo's numbers — organized `scripts/` by pipeline stage (was flat `temp_*` in
the original repo's root), verified data migration (`migration/MANIFEST.md`), frozen
`requirements.txt`. All 10 stages completed 2026-07-15 (`SESSION_HANDOFF.md`); `paper/paper_results.md`
was reconciled against the fresh numbers 2026-07-20 (see the note at the top of this page). The
original repo is now deleted locally — this is the sole active repo.

## Non-wiki key files
**`paper/` folder (repo root)** — the drafted manuscript, split by section (**`paper_results.md` = source of truth for numbers**): `paper_idea.md` (THMS blueprint + 2026-07-05 reshape note), `paper_intro.md`, `paper_method.md`, `paper_results.md`, `paper_discussion.md`, `paper_conclusion.md`, `paper_abstract.md`, `paper_experiments_plan.md` (tiered experiment roadmap), `live_study_protocol.md` (T7 — **NOT approved**, discussion draft only, do not build from it without human sign-off). `RESEARCH_LOG.md` (Planning↔Implementation channel — follow its rules), `SESSION_HANDOFF.md` (thin pointer), `CODEBASE_DESCRIPTION.md` / `projectAnalysis.md` (raw sources, superseded for navigation).
