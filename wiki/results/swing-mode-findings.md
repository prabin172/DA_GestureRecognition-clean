---
type: result
status: active
updated: 2026-07-03
---

# Swing-mode findings (branch `swing-mode-xsens`, runs of 2026-07-01)

Synthesis of Job 1 ([[loso-fulltrain-calibrate]] swing table), Job 2 ([[xsens-to-xsens-loso]]), and swing [[mmd-domain-gap]]. All three complete on disk; **not yet written into RESEARCH_LOG.md Section B and not committed** — see [[open-questions]].

## The three-way comparison (mean final_acc)
| | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| Job1 supMAE (NTU-pretrained, fine-tuned) | 86.9 | 94.8 | 95.0 | **97.5** |
| Job2 supmae (target-only, frozen, no NTU) | 84.5 | 93.5 | **95.7** | 96.4 |
| scratch | **88.2** | 92.2 | 94.6 | 96.4 |

## What the data honestly supports
1. **NTU pretraining is redundant given 4 labeled target subjects.** Target-only supervised+MAE matches NTU-pretrained supMAE within noise (per-subject paired deltas: wins only k=5, 4/5; loses k=10 0/5). Not "edges it" — "matches."
2. **This holds despite a 3× wider feature-space gap** (swing MMD) — the redundancy is invariant to the gap change, which *strengthens* the characterization framing ([[paper-framing]]).
3. **True self-supervision fails frozen:** target-MAE collapses to 58–73%. The "skip NTU, self-supervise" story is dead; Job 2's "supmae" uses labels ([[publishability-review]] item 1).
4. **Swing raised absolute accuracy via mounting-variance normalization** — sub7 +54.6 pp, sub8 −19.9 pp ([[swing-mode]]). Not "helped all methods/subjects."
5. Scratch wins k=1 outright (4/5 subjects); only sign-consistent supMAE win is k=3.

## Before these are paper claims
Complete the 2×2 (init × frozen/finetuned), McNemar-per-subject + multi-seed stats, explain sub8, symmetric swing MMD or CKA — full list in [[publishability-review]] and [[open-questions]].
