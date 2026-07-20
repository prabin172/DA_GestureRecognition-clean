---
type: experiment
status: done
updated: 2026-07-20
---

# OOV / leave-class-out few-shot

**Question:** how well can a novel gesture class be onboarded with k shots? Fully run: 22 classes × 5 subjects × 5 methods × k∈{1,3,5,10}. Script: `scripts/main_experiment/loso_leave_class_out_fewshot.py`. Dir: `trained_models/LOSO-LeaveClassOutFewShot/` (seed 42; `-seed{43,44}` 3-seed extension launched 2026-07-20, in progress, `scripts/orchestration/05b_oov_multiseed.sh`, ~15h/seed).

**Protocol (confirmed executed):** frozen base encoder trained on 21 classes → expand head to 22 rows, known rows copied, OOV row random-init → calibrate with k shots of the OOV class only ([[loso-protocol]]).

**2026-07-20:** numbers below are now from `DA_GestureRecognition-clean`'s completed Stage 5 rerun — **5 methods incl. supcon**, seed 42 (550 base + 2200 calibration runs, completed 2026-07-14). Not a paper_results.md table (RESEARCH_LOG-only, "borderline load-bearing"), so it wasn't part of the R2/R3/R6 discrepancy audit — treat as directional pending the 3-seed pooling.

Mean OOV recall across 5 subjects × 22 classes:

| method | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| supcon | **73.9%** | 88.2% | 91.4% | 95.0 |
| supLP120 | **73.9%** | **85.6%** | **89.4%** | 95.3% |
| supMAE | 67.3% | 83.6% | 84.6% | **95.5%** |
| scratch | 67.4% | 79.9% | 81.9% | 93.9% |
| mae | 65.5% | 78.9% | 79.5% | 92.7% |

- **supLP120/supcon now lead, not supMAE** — a different ranking than the original 4-method run
  (which had supMAE leading by 6–8pp at k≤5). This mirrors the small-gap pattern the paper's R6/R6d
  external datasets found (label-aware objectives winning when the target isn't too far from NTU) —
  plausible here since OOV onboarding stays in-domain (Xsens), unlike the cross-modal settings where
  supMAE wins. Not yet statistically tested (single seed); treat as a lead worth pooling, not a
  confirmed finding.
- Per-action reliability: crossarms/squat/wave reliable, throw/jump/hop not → feeds the controller's safety-command assignment ([[paper-framing]]).
- **Still missing:** per-action statistics + the distinctiveness analysis (correlate inter-class kinematic separability in encoder space with OOV recall — RESEARCH_LOG A4.6/A6, highest-value cheap addition). See [[open-questions]].
