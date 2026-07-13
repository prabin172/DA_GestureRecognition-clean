---
type: index
status: active
updated: 2026-07-10
---

# Project Overview — the thesis page

## What this project is
Gesture recognition via domain adaptation. Large vision-captured skeleton dataset ([[ntu-dataset|NTU RGB+D]], ~43k clips, 120 classes) pre-trains a Transformer encoder; knowledge transfers to a small inertial mocap dataset ([[xsens-dataset|Xsens IMU]], ~2 700 clips, 22 gestures, 5 subjects). Both domains share one representation: [[lrq|Local Relative Quaternions]] — 17 segments × 4 quat components = 68-d per frame at 30 Hz.

**Core research question:** which pre-training objective on NTU transfers best to the target IMU domain under few-shot ([[loso-protocol|k-shot LOSO]]) conditions?

## The answer so far (locked framing, RESEARCH_LOG.md §A1)
The contribution is a **characterization, not a method win**: *source-objective effects are conditional on source–target representation compatibility.*

Three pillars ([[paper-framing]]):
1. **Objectives matter within a domain** — NTU→NTU spread is huge (supervised 59.6% vs MAE 22.6% vs scratch 14.8%). See [[sanity-checks]].
2. **The effect compresses across the domain gap** — NTU→Xsens k=1 spread is only ~7 pp. See [[loso-fulltrain-calibrate]].
3. **Representation mismatch is the cause** — shown via CKA (necessary-not-sufficient: supLP120 aligns 3–5× better and monotonically with depth, yet isn't the best-transferring objective). See [[domain-gap-metrics]], [[phase1-mcnemar-ece-cka]].

Target venue: IEEE THMS. A controller pillar (gesture → abstract event-driven controller, uncertainty → task reliability) is confirmed GO — see [[paper-framing]].

## Current state (2026-07-10, branch `swing-mode-xsens`)
**v2 preprocessing is the chosen/locked representation** (rebuilds Xsens from mvnx positions through NTU's own shortest-arc construction, [[position-reconstruction-v2]]; swing was a falsified hypothesis, dead — see [[swing-mode-findings]]). All headline numbers are **3-seed pooled**: mae is a seed-stable significant negative-transfer case at every k (McNemar p<.001); supLP120 is best-calibrated and dominates at low subject count; the raw accuracy spread compresses to a few points across the gap but the prior pays off in calibration/data-efficiency ([[multiseed-loso-v2]], [[phase1-mcnemar-ece-cka]], [[a2-subject-scaling]]).

The thesis has grown a fourth leg beyond the original three pillars: **external validity across two independent public datasets and five source→target settings** (CZU-MHAD skeleton, UTD-MHAD skeleton, NTU→Xsens, CZU-MHAD IMU orientation-only, CZU-MHAD IMU dual-raw). The **five-setting synthesis map** (`paper/paper_results.md` R6e) is now the paper's unifying result: both the best-performing prior *and* the actively-harmful one change identity as the gap widens and the target representation strengthens — supervised wins at small/native-skeleton gaps, hybrid wins and pure reconstruction hurts at the middle gap (NTU→Xsens), nothing helps and pure supervision hurts at the two large cross-modal gaps. See [[czu-skeleton-loso]], [[utd-skeleton-loso]], [[czu-imu-crossmodal]], [[czu-imu-dual]].

**A fifth pretraining objective (SupCon) now runs through every table** — it tracks supLP120 almost exactly at every setting: a wash vs scratch on the core NTU→Xsens path, a big win on both small-gap external datasets, a loss/wash on both large-gap cross-modal ones. This sharpens the R6e axis: the split isn't "supervised (softmax) vs reconstruction," it's **"has a reconstruction component (supMAE, mae) vs doesn't (supLP120, supcon)"** — see [[pretraining-objectives]]. The A2 cold-start deployment claim (prior helps most with 0–1 enrolled subjects) is now scoped: it does **not** hold on a strong (raw-signal) target — [[czu-dual-cold-start]] finds no subject count at which any prior beats scratch there.

A controller pillar (C6, [[phase3-controller]]) closes the human-machine-systems loop: a robustness-locked simulation shows these compressed accuracy differences compound into large mission-success/safety differences, invariant to vocabulary/cost-model/threshold choices; the SupCon extension also surfaced and documented a vocabulary-selection script quirk without touching the already-locked numbers.

The full manuscript is drafted in `paper/` (`paper_results.md` = source of truth). The **2026-07-09 publishability review → `tasks.md` work plan** is largely executed: five-setting synthesis (T0), multi-seed completion (T1), SupCon full parity (T2), CKA-per-target gap measurement (T3, an honest null — doesn't confirm the claimed small/middle/large ordering), UTD CRC anchor (T4), and CZU cold-start scoping (T5) are all done. Still open: a public data release, paused on a PII de-identification pass ([[open-questions]] T6), and a live human-in-the-loop study with a written but **unapproved** protocol (T7). T8 (full reproducible rerun) and T9 (second backbone) are explicitly deferred. See `SESSION_HANDOFF.md` for exact next steps.

## Where the project is stuck / at risk
Nothing is a hard blocker. Two things are open and human-gated, not technical risk:
1. **T6 (public data release)** is paused: the raw Xsens `.mvnx` files were found to embed subjects' real names in an XML metadata field (not visible from filenames) — a de-identification pass is planned but not yet executed. See `wiki/questions/open-questions.md`.
2. **T7 (live human-in-the-loop study)** — the THMS-earning artifact beyond the simulated controller — has a written, unapproved protocol (`paper/live_study_protocol.md`) awaiting a human go/no-go before any real-time-inference build work starts.

Everything else structural (McNemar, CKA, multi-seed pooling, the gap-ordering measurement) has landed. Full standing-question list: [[open-questions]].

## Key files outside the wiki
- `RESEARCH_LOG.md` — shared channel with the Planning assistant (Section A = Planning, Section B = this side). Follow its header rules; write numbers, not conclusions.
- `SESSION_HANDOFF.md` — thin pointer: current task + wiki entry points.
- `CODEBASE_DESCRIPTION.md`, `projectAnalysis.md` — raw sources the wiki was built from (superseded for navigation).
