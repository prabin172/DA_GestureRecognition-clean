---
type: result
status: active
updated: 2026-07-03
---

# Publishability review (2026-07-01)

Critical audit of the swing-mode results (source: `projectAnalysis.md`). TLDR: runs are fine; the interpretation in the old handoff overstated, one condition is mislabeled, and the A3-required stats machinery doesn't exist.

## The eight problems
1. **Job 2 "supmae" is not self-supervision** — `temp_xsens_to_xsens_loso_calibration.py:121-124` adds CE on training-subject labels. The true no-labels condition (`mae`) collapses frozen. → [[xsens-to-xsens-loso]]
2. **Headline comparison cherry-picks k=5** — the only k Job 2 wins (paired: k=1 −2.4 pp 1/5, k=3 −1.3 1/5, k=5 +0.8 4/5, k=10 −1.1 0/5). Say "matches within noise."
3. **Scratch wins k=1 outright** (4/5 subjects); handoff means hid it.
4. **"Swing helped all" false per-subject** — sub8 −19.9 pp, unexplained. → [[swing-mode]]
5. **MMD flawed 3 ways** — scale confound (fixed sigmas), asymmetric (Xsens-only twist removal), no CIs; scratch row uninterpretable. → [[domain-gap-metrics]]
6. **Job1 vs Job2 confounds init source × frozen/fine-tuned** — 2×2 incomplete.
7. **A3 stats policy violated everywhere** — single seed, no paired stats; Wilcoxon floor at n=5 is p=0.0625.
8. **Ceiling at k≥5** (94–98%) — only k∈{0,1,3} discriminates.

## The fix list (→ actionable items in [[open-questions]])
1. Reframe headline to the A1 characterization: *objective choice matters within-domain; with 4 labeled target subjects, cross-domain skeleton pretraining is redundant, invariant to a 3× gap change.*
2. Relabel Job 2 supmae → "target supervised+MAE"; keep + report the mae collapse as evidence.
3. Complete the 2×2 (two more runs, same scripts).
4. Stats that work at n=5: multi-seed (3–5) at k∈{0,1,3}; clip-level McNemar per subject (~400–500 paired clips) → "significant in x/5 subjects"; subject-level deltas as descriptive.
5. MMD → CKA (+ median-heuristic MMD secondary, bootstrap CIs, swing-projected NTU rerun).
6. Explain sub8; per-subject appendix; DANN-swing only if swing is chosen.
7. Housekeeping: rename/cite SupMAE, commit swing work, Section B gets numbers not conclusions.

**Biggest single risk:** items 1+2 — the current spine rests on a mislabeled condition and one cherry-picked k. Reframed as redundancy-of-source-pretraining, the same data supports a clean claim.
