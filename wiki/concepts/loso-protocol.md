---
type: concept
status: active
updated: 2026-07-10
---

# LOSO protocol & k-shot calibration

## LOSO (Leave-One-Subject-Out)
The 5 Xsens subjects (sub7–sub11) each take a turn as the fully held-out test subject; the other 4 provide training data. Tests generalization to an unseen person.

## k-shot calibration (the main evaluation paradigm)
Used by [[loso-fulltrain-calibrate]] and [[xsens-to-xsens-loso]]:
1. **Base phase**: initialize encoder (NTU-pretrained / target-pretrained / scratch), train on the 4 non-held-out subjects.
2. **Calibration phase**: take k examples per class *from the held-out subject*, train **head-only** (encoder frozen), evaluate on the rest of that subject's clips. k ∈ {0, 1, 3, 5, 10}; k=0 means no calibration.

Note: k=0 with a frozen never-target-trained encoder + untrained head is meaningless (~4–5%) — ignore those cells in [[xsens-to-xsens-loso]].

## OOV / leave-class-out variant
Hold out one *class* instead: train 21-way, then expand the head to 22 rows (known rows copied, OOV row random-init), calibrate with k shots of the OOV class only. See [[oov-leave-class-out]].

## Statistics problem at N=5 — solved
Stats policy (RESEARCH_LOG.md §A3): paired subject-blocked tests, effect sizes + CIs, never best-test-epoch, no tuning on test. Two-sided Wilcoxon at n=5 bottoms out at p=0.0625 — no single-seed, single-k subject-level comparison can reach p<0.05 alone. The workable machinery ([[publishability-review]] item 4) is now built and run:
- **Multi-seed pooling** (3 seeds × 5 subjects = n=15 per cell) turns subject-level comparisons into paired-t-testable samples — [[multiseed-loso-v2]], [[a2-subject-scaling]], and every external dataset ([[czu-skeleton-loso]], [[czu-imu-crossmodal]], [[czu-imu-dual]], [[utd-skeleton-loso]]).
- **Clip-level McNemar per subject** (~400–500 paired clips each, pooled across seeds to n≈7–8k pairs per k) — [[phase1-mcnemar-ece-cka]]. This is the per-clip significance layer the subject-level means alone couldn't supply.
- Per-subject deltas still reported descriptively alongside the pooled stats.

## Ceiling effect
Everything is 94–98% at k≥5; the discriminating regime is k∈{0,1,3}.
