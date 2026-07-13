---
type: experiment
status: done
updated: 2026-07-03
---

# OOV / leave-class-out few-shot

**Question:** how well can a novel gesture class be onboarded with k shots? Fully run: 22 classes × 5 subjects × 4 methods × k∈{1,3,5,10}. Script: `temp_loso_leave_class_out_fewshot.py`. Dir: `trained_models/LOSO-LeaveClassOutFewShot/` (+ a `_PARTIAL_20260616_153911` leftover).

**Protocol (confirmed executed):** frozen base encoder trained on 21 classes → expand head to 22 rows, known rows copied, OOV row random-init → calibrate with k shots of the OOV class only ([[loso-protocol]]).

Mean OOV recall across 5 subjects × 22 classes:

| method | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| supMAE | **71.2%** | **77.5%** | **81.0%** | **93.4%** |
| MAE | 65.1% | 73.3% | 75.4% | 91.6% |
| scratch | 65.1% | 73.4% | 75.8% | 90.9% |
| supLP120 | 63.4% | 75.9% | 79.7% | 91.4% |

- supMAE leads by 6–8 pp at k≤5 — richer representations help most when onboarding a truly novel class.
- Per-action reliability: crossarms/squat/wave reliable, throw/jump/hop not → feeds the controller's safety-command assignment ([[paper-framing]]).
- **Still missing:** per-action statistics + the distinctiveness analysis (correlate inter-class kinematic separability in encoder space with OOV recall — RESEARCH_LOG A4.6/A6, highest-value cheap addition). See [[open-questions]].
