---
type: result
status: active
updated: 2026-07-10
---

# Paper framing (locked — RESEARCH_LOG.md Section A)

Ground truth decided by the human + Planning assistant. Do not re-derive; the human may revise. Full authoritative text: `RESEARCH_LOG.md` §A.

## Locked claims (A1)
- **Contribution = characterization, not a method:** source-objective effects are conditional on source–target representation compatibility. Objectives matter within-domain; the effect compresses across the skeleton→wearable gap; representation mismatch dominates.
- **No "first skeleton-to-IMU transfer" claim** (prior work: Moya Rueda/Fink, Awasthi, SKELAR, UniMTS, PSKD, Zolfaghari, Xia). Novelty = controlled objective comparison under deployment-realistic protocols + explicit mismatch analysis + the OOV result.
- **"SupMAE" is an existing vision method** — cite on first use or rename ("Sup+MAE"/"hybrid").
- **Venue: IEEE THMS** (journal; depth beats big numbers; small N survivable if honest).

## Three pillars (A2)
1. Objectives matter within-domain — [[sanity-checks]].
2. Effect compresses across the gap — [[loso-fulltrain-calibrate]]; reinforced by the relatedness ablation ([[early-experiments]]).
3. Mismatch is the cause — must be *shown* (CKA/UMAP), not asserted — [[domain-gap-metrics]].

## Stats & honesty policy (A3)
N=5 openly stated as a limitation; per-subject appendix. Paired subject-blocked stats (Wilcoxon / paired bootstrap), per-subject deltas, effect sizes + CIs. Never best-test-epoch as primary. No tuning on test signal; report selection rules; λ work = robustness analysis. (Practical n=5 machinery: [[loso-protocol]].)

## Controller pillar (A0 — GO)
Abstract event-driven controller (no physics sim): what earns the THMS venue. Gesture subset → primitives (next/previous/approach/grasp/release/confirm/cancel); posterior stream from real held-out clips through the actual recognizer; FSM + safety layer (temporal smoothing, confidence threshold with reject, dwell filter, error recovery, safe-home). Metrics: task success, time-to-completion, false-activation rate, corrective commands, rejection rate, latency — across 4 inits × k; one confidence-threshold sweep → safety/throughput curve. Design requirements: sequential asymmetric task (errors compound), held-out non-target gestures as distractor stream, tune on validation not test, safety-critical commands on reliably-recognized gestures ([[oov-leave-class-out]]). **Sequencing: build only after recognizer numbers are final.** Each step human-OK'd first. **Built and locked** — see [[phase3-controller]].

## Proposals awaiting human OK (A6)
Distinctiveness-predicts-onboarding (still open, not in the current `tasks.md` T0–T9 plan — flag for Planning); CKA transfer-gap quantification (done, [[phase1-mcnemar-ece-cka]], [[domain-gap-metrics]]); frozen/partial/full Pareto (not run); contrastive baseline (SupCon).

**SupCon status update (2026-07-10, T2 full-parity pass):** the original "SupCon underperforms scratch, empirically scoped out" call (RESEARCH_LOG §A item 4) was based on a single-seed Xsens-only run. Running SupCon through every analysis (R1–R6e) instead finds it **tracks supLP120 almost exactly**: a wash vs scratch on NTU→Xsens (like supLP120), but wins big on both small-gap external datasets (CZU/UTD, p≤.006) and loses/washes on both large-gap cross-modal ones — the same pattern as the pure-supervised objective, not a distinct "contrastive is bad" story. See [[pretraining-objectives]], [[czu-skeleton-loso]], [[utd-skeleton-loso]]. This is a factual update to what "scoped out" means empirically, not a re-litigation of the Section A framing call itself — flagged here for Planning to revisit if it changes how the paper should describe SupCon.
