---
type: index
status: active
updated: 2026-07-10
---

# Wiki Index

Start at [[overview]] (thesis + current state + where stuck). Latest state (2026-07-10, THMS publishability push, `tasks.md` T1/T2/T5 analysis + write-up): **SupCon (5th objective) analyzed end-to-end** — R1 (within/cross-domain accuracy, wash vs scratch like the others), R2/R4a (McNemar + ECE, 2nd-best-calibrated after supLP120), R5 (controller, mae still worst of 5; **found + documented a script quirk**: `reliability_ordered_vocab()` pools recall across all loaded methods, so adding supcon silently shifted the fixed vocab used by Locks 2/3 — the locked table is untouched, supcon's own numbers reported separately, ordering still replicates), and all four external datasets (R6/R6b/R6c/R6d) — **supcon tracks supLP120 almost exactly at every setting: wins big at small gap (both CZU/UTD, p≤.006), loses at large gap (both quat/dual)**, reframing the R6e axis from "supervised vs reconstruction" to "label-supervision-only (any recipe) vs has-a-reconstruction-component". **T1 multi-seed pooling landed**: A2 (now 3-seed, the k=1/N=1 spike confirmed real not a seed artifact, +18.4pp p<.0001), R6c raw-only + DIAL dial (both pooled, same conclusions tighter). **T5 (new): the A2 cold-start deployment lever is itself target-representation-contingent** — repeating the subject-scaling sweep on CZU's strong target finds no N where any prior beats scratch, both significantly *worse* at N=0,k=1 (see [[czu-dual-cold-start]]). Prior state (2026-07-09): **R6e five-setting synthesis table** (`paper/paper_results.md`) unifies R2/R6/R6b/R6c/R6d into one gap×target-strength map. **T3 (CKA per target): does NOT confirm the small/middle/large gap ordering** — reported as an honest null (see [[domain-gap-metrics]]). Prior state: [[multiseed-loso-v2]] (**3-seed stats — mae = negative transfer**) · [[czu-skeleton-loso]] (independent public target, now pooled+supcon) · [[czu-imu-crossmodal]] (**cross-modal, 3-seed pooled + supcon**) · [[czu-imu-dual]] (**R6c pooled + supcon + cold-start**) · [[utd-skeleton-loso]] (**R6d, pooled + supcon**) · [[position-reconstruction-v2]] (v2 domain-gap fix) · [[open-questions]].

## Concepts
- [[lrq]] — Local Relative Quaternions, 17-segment model, padding; the shared representation
- [[loso-protocol]] — LOSO, k-shot calibration, OOV variant, the n=5 stats problem
- [[pretraining-objectives]] — MAE / supervised / SupMAE / SupCon / DANN / scratch + verdicts
- [[swing-mode]] — axial-twist removal: hypothesis, outcome, sub7/sub8 asymmetry
- [[domain-gap-metrics]] — MMD (flawed), CKA (done, per-encoder + **per-target T3: does NOT confirm the small/middle/large gap ordering**), necessary-not-sufficient nuance, **+ raw method-independent gap (2026-07-12): MMD²/Frechet on hand-crafted features DOES confirm the ordering, opposite of CKA — now in `paper_results.md` R3/R6e**
- [[controller]] — **new (2026-07-12): full design doc for the C6 controller** — mission/FSM/cost-model mechanics, the 3 robustness locks explained, what the abstraction represents, + the planned (unapproved) live-study extension

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
- [[phase1-mcnemar-ece-cka]] — **Phase 1: per-clip McNemar (mae neg-transfer p<.001 every k), ECE (supLP120 best-calibrated, supcon 2nd), CKA (supLP120/supcon align best, supcon numerically highest yet accuracy is a wash — sharpest alignment≠accuracy dissociation)** — no retrain
- [[a2-subject-scaling]] — **Phase 2: prior benefit vs #FT-subjects, now 3-seed pooled — peaks N=0/1 (supLP120 +27.5pp @k3,N1, p<.0001), washes to ≈1–1.4pp by N=4; mae ≤0** (replaces dead gap-knob). **Caveat: lever is target-representation-contingent, see [[czu-dual-cold-start]]**
- [[czu-dual-cold-start]] — **T5 (new): does A2's cold-start lever survive on a strong target? No** — neither supLP120 nor supMAE beats scratch at any N on CZU-dual; both significantly worse at N=0,k=1
- [[phase3-controller]] — **Phase 3 (prototype+robust): controller on real posteriors — compounding (mae 0.75 vs supMAE 0.97 task-success @k1), calibration→safety, locked** + **supcon extension (2026-07-10): Lock 1 exact-reproduces, Locks 2/3 exposed a vocab-selection script quirk (documented, ordering still replicates)**
- [[multiseed-loso-v2]] — **3-seed LOSO-v2 stats: accuracy + AUC + convergence; the load-bearing stats** (McNemar now paid in [[phase1-mcnemar-ece-cka]])
- [[czu-skeleton-loso]] — **CZU-MHAD skeleton→skeleton external validity (R6, now pooled 3-seed + supcon)** — supLP120 +7.2pp @k0 p=.0004; **supcon +7.0/+4.8/+4.4pp all k, p<.001, beats supLP120 itself p=.0002**
- [[czu-imu-crossmodal]] — **CZU-MHAD skeleton→IMU TRUE cross-modal (R6b, 3-seed pooled + supcon)** (prior ranking inverts: supLP120 worst / below scratch; supMAE > supLP120 36/44 p<.0001 but supMAE≈scratch — the single-seed +3 pp was a seed-42 artifact; **supcon tracks supLP120's negative/neutral pattern**, ruling out "softmax-specific" as the mechanism)
- [[czu-imu-dual]] — **CZU-MHAD dual-branch (R6c, dual pooled + supcon + cold-start)** (raw scratch ≈ CRC → representation was the bottleneck; on a strong target the prior adds nothing, supLP120 worst p<.0001, **supcon also worse p=.014**; +target-richness dose-response dial pooled — three-column gap-contingent arc; **+cold-start extension: no N helps either, see [[czu-dual-cold-start]]**)
- [[utd-skeleton-loso]] — **UTD-MHAD skeleton→skeleton (R6d, + supcon)** (2nd independent public dataset; small-gap supLP120 win replicates +7.4 pp @k0 p<.0001, mae positive not negative; **T4 CRC anchor: supLP120−CRC +14.85pp @k0**; **supcon +8.4pp @k0 p<.0001, statistically tied with supLP120**)
- [[position-reconstruction-v2]] — **v2 position-derived Xsens: the real domain-gap fix** (supersedes swing)
- [[swing-mode-findings]] — synthesis of the 2026-07-01 swing runs
- [[publishability-review]] — the 8 problems + fix list (2026-07-01 audit)
- [[paper-framing]] — locked framing, 3 pillars, stats policy, controller pillar (mirrors RESEARCH_LOG §A)
- [[literature-landscape]] — Undermind deep-research report (2026-07-03): the gap is ours, four competitor clusters, THMS venue precedent

## Questions
- [[open-questions]] — TODO, blockers, standing puzzles

## Non-wiki key files
**`paper/` folder (repo root)** — the drafted manuscript, split by section (**`paper_results.md` = source of truth for numbers**): `paper_idea.md` (THMS blueprint + 2026-07-05 reshape note), `paper_intro.md`, `paper_method.md`, `paper_results.md`, `paper_discussion.md`, `paper_conclusion.md`, `paper_abstract.md`, `paper_experiments_plan.md` (tiered experiment roadmap), `live_study_protocol.md` (T7 — **NOT approved**, discussion draft only, do not build from it without human sign-off). `RESEARCH_LOG.md` (Planning↔Implementation channel — follow its rules), `SESSION_HANDOFF.md` (thin pointer), `CODEBASE_DESCRIPTION.md` / `projectAnalysis.md` (raw sources, superseded for navigation).
