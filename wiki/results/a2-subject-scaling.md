---
type: result
status: active
updated: 2026-07-10
---

# A2 — Subject-count scaling (how long does the prior stay useful?)

Replaces the dead gap-knob pillar (C4). Question: **how many enrolled fine-tuning subjects before the pretrained prior stops mattering?** Sweeps N = number of non-held-out subjects used to fine-tune the base encoder before k-shot LOSO calibration. N=0 = pretrained init + k-shot calibration only (no FT subjects); N=4 = the full LOSO-v2 setup. **3 seeds (42/43/44)**, 5 LOSO folds per N per seed (n=15/cell), methods {scratch, mae, supLP120, supMAE}, k∈{0,1,3}.

Plumbing: `--n-train-subjects N` on `temp_loso_fulltrain_calibration.py` (nested first-N of sorted non-held-out subjects; N=0 short-circuits to pretrained init). Runner `temp_a2_run.sh` (seed 42) + `temp_t1_multiseed_run.sh` (seeds 43/44, T1.1). Outputs `trained_models/A2-subjectScaling{,-seed43,-seed44}/N{0..4}/summary.csv`, pooled `trained_models/A2-subjectScaling-pooled/a2_pooled_results.csv`. See [[temp-scripts]].

## Prior benefit vs scratch (pp), by N × k — pooled 3 seeds, n=15/cell
| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|-------|---|-----|-----|-----|-----|-----|
| supLP120 | 0 | −0.36 | +3.34 | +1.45 | +2.26 | +1.27 |
| supLP120 | 1 | +6.96 | **+18.43** | +3.64 | −0.37 | −1.34 |
| supLP120 | 3 | +18.47 | **+27.46** | +7.15 | +2.41 | +1.42 |
| supMAE | 0 | −1.10 | −0.37 | −0.28 | −0.01 | +2.52 |
| supMAE | 1 | +3.35 | +6.42 | +3.48 | +2.02 | +0.97 |
| supMAE | 3 | +9.18 | +9.40 | +2.13 | +2.15 | +1.85 |
| mae | 0 | −1.40 | −0.95 | −4.67 | −5.24 | −4.24 |
| mae | 1 | +2.42 | +0.44 | −1.80 | −2.44 | −2.84 |
| mae | 3 | +5.29 | +2.74 | −2.04 | −0.76 | −1.34 |

(scratch is the 0 baseline by construction.) Paired-t (n=15): supLP120 − scratch significant at every (N,k) shown above for N=0/1, k=1/3 (all p<.0001); supMAE − scratch also significant (p=.004 to p<.0001); mae − scratch turns significantly negative by N=4, k=1 (p=.0018).

## Findings
- **Prediction confirmed, now statistically.** Prior benefit peaks at N=0/1 and washes to ≈1–1.4pp by N=4. supLP120 @ k=3: **+27.5pp at N=1** (p<.0001), decaying +7.2 → +2.4 → +1.4 as subjects accumulate.
- **The k=1, N=1 spike is real, not a seed artifact.** Per-seed breakdown: +15.5 / +21.2 / +18.6 pp (seeds 42/43/44) — all large and positive, pooled +18.4 (p<.0001). This resolves the caveat the single-seed version carried; treat both k=1 and k=3 trends as robust.
- **supLP120 dominates the low-N regime** — consistent with its best CKA alignment and best calibration in [[phase1-mcnemar-ece-cka]]. The objective that transfers best is exactly the one whose value decays fastest as target data arrives.
- **mae reinforces C2**: negative for N≥2 at k=1 (significant by N=4, p=.0018), negative at k=0 for all N. Reconstruction prior actively hurts once any real subjects are available.
- supMAE is the steady middle — modest positive benefit that degrades gracefully, positive for N≥1 at k=1/3.

## Caveats
- N=0 k=0 is near-chance for all methods (pretrained init, no calibration, cross-domain) — benefit numbers there are on tiny absolute accuracies; read k=1/3.
- Pooling analysis via `temp_t1_multiseed_run.sh` output, ad hoc pooling script (not yet folded into `temp_analyze_a2.py`, which remains single-seed-only — TODO if this page needs re-generating).
- **This cold-start lever is target-representation-contingent** — see [[czu-dual-cold-start]]: repeating this exact sweep on CZU's strong (raw-signal) target finds no N where the prior beats scratch, and both priors are significantly *worse* than scratch at N=0, k=1 (the cell where Xsens shows the prior's biggest win). Read this page's headline claim as scoped to weak/moderate targets, not universal.

Related: [[phase1-mcnemar-ece-cka]] · [[multiseed-loso-v2]] · [[paper-framing]] · [[pretraining-objectives]]
