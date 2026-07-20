---
type: experiment
status: done
updated: 2026-07-03
---

# Xsens→Xsens LOSO (Job 2 — target-only, no NTU)

**Question:** does target-only pretraining match NTU pretraining? (The RESEARCH_LOG A5 "spine-deciding" question.)

**Protocol:** pretrain encoder on the 4 non-held-out Xsens subjects with a given objective, **freeze it**, run the identical head-only k-shot calibration as [[loso-fulltrain-calibrate]] (calibration code imported from `scripts/main_experiment/loso_fulltrain_calibration.py` as `L` so it cannot drift). `supervised` column ≡ Job 1 `scratch` by construction. Script: `temp_xsens_to_xsens_loso_calibration.py`. Dir: `trained_models/XsensToXsens-LOSO-swing/` (50 rows, swing mode, 2026-07-01).

| objective | k=0 | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| mae | 5.2 | 58.3 | 61.4 | 61.0 | 72.9 |
| supmae | 3.9 | 84.5 | 93.5 | **95.7** | 96.4 |

- k=0 meaningless (frozen encoder + untrained head, ~4–5%).
- **Frozen target-MAE collapses** (58–73%) — consistent with MAE's weak NTU→NTU features ([[sanity-checks]]).

## ⚠️ Two critical caveats ([[publishability-review]])
1. **"supmae" here uses labels.** `temp_xsens_to_xsens_loso_calibration.py:121-124` sets `sup=True` → CE on the 4 training subjects' labels. It is *supervised + MAE aux loss*, NOT self-supervision. The true no-labels condition is `mae`, which collapses — the "no labels needed" story is dead. Relabel as "target supervised+MAE".
2. **vs Job 1 supMAE, it only wins k=5** (paired Δ: k=1 −2.4 pp 1/5, k=3 −1.3 1/5, k=5 +0.8 4/5, k=10 −1.1 0/5). Correct claim: "matches within noise."
3. **Confound:** Job 1 = NTU init + full fine-tune; Job 2 = target pretrain + frozen. The 2×2 needs completing (NTU-pretrained-frozen; target-supmae-finetuned) — [[open-questions]].

Honest conclusion: *with 4 labeled target subjects, NTU pretraining adds nothing* — fits the A1 characterization framing. See [[swing-mode-findings]].
