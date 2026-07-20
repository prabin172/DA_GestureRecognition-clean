---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: numbers below refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints** (`trained_models/A2-subjectScaling-pooled/`). Every flagship number holds within ~1pp of the original — the most checkpoint-stable result in the paper alongside ECE ranking. See `paper/paper_results.md` R4c.

# A2 — Subject-count scaling (how long does the prior stay useful?)

Replaces the dead gap-knob pillar (C4). Question: **how many enrolled fine-tuning subjects before the pretrained prior stops mattering?** Sweeps N = number of non-held-out subjects used to fine-tune the base encoder before k-shot LOSO calibration. N=0 = pretrained init + k-shot calibration only (no FT subjects); N=4 = the full LOSO-v2 setup. **3 seeds (42/43/44)**, 5 LOSO folds per N per seed (n=15/cell), methods {scratch, mae, supLP120, supMAE}, k∈{0,1,3}.

Plumbing: `--n-train-subjects N` on `scripts/main_experiment/loso_fulltrain_calibration.py` (nested first-N of sorted non-held-out subjects; N=0 short-circuits to pretrained init). Runner `scripts/orchestration/a2_run.sh` (seed 42) + `scripts/orchestration/t1_multiseed_run.sh` (seeds 43/44, T1.1). Outputs `trained_models/A2-subjectScaling{,-seed43,-seed44}/N{0..4}/summary.csv`, pooled `trained_models/A2-subjectScaling-pooled/a2_pooled_results.csv`. See [[temp-scripts]].

## Prior benefit vs scratch (pp), by N × k — pooled 3 seeds, n=15/cell
| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|-------|---|-----|-----|-----|-----|-----|
| supLP120 | 0 | −0.41 | +4.79 | +1.93 | +1.51 | +0.52 |
| supLP120 | 1 | +7.35 | **+18.20** | +5.02 | +1.14 | −0.81 |
| supLP120 | 3 | +18.22 | **+28.25** | +7.57 | +2.66 | +1.86 |
| supMAE | 0 | −0.96 | −0.85 | −1.98 | −1.94 | −0.35 |
| supMAE | 1 | +3.15 | +5.20 | +2.25 | −0.34 | +0.26 |
| supMAE | 3 | +8.58 | +8.92 | +1.95 | +0.84 | +2.11 |
| mae | 0 | −0.01 | −0.78 | −3.53 | −4.42 | −3.39 |
| mae | 1 | +1.35 | −0.42 | −1.74 | −2.94 | −3.15 |
| mae | 3 | +4.39 | +3.52 | −1.84 | −0.89 | −1.45 |

(scratch is the 0 baseline by construction.) Paired-t (n=15): supLP120 − scratch significant at N=0/N=1 both k (p≤.008); supMAE − scratch also significant at N=0/N=1 both k (p≤.008); mae − scratch turns significantly negative at N=2 k=0 (p=.007), N=3 k=0/k=1 (p≤.02), N=4 k=1 (p=.003).

## Findings
- **Prediction confirmed, and now confirmed across an independent full retrain.** Prior benefit peaks at N=0/1 and washes to ≈1–2pp by N=4. supLP120 @ k=3: **+28.3pp at N=1** (matching the original +27.5pp within pooling noise), decaying to +2.7 → +1.9 as subjects accumulate.
- **supLP120 dominates the low-N regime** — consistent with its best calibration in [[phase1-mcnemar-ece-cka]] (its CKA-alignment lead did not survive the retrain as cleanly, see that page's 2026-07-20 note; calibration is the more stable mechanism claim). The objective that transfers best is exactly the one whose value decays fastest as target data arrives.
- **mae reinforces C2**: negative from N≥2 onward, at both k=0 and k=1. Reconstruction prior actively hurts once any real subjects are available — this is the one A2 claim that both replicated across seeds originally and now also across an independent checkpoint retrain.
- supMAE is the steady middle — modest positive benefit that degrades gracefully, positive for N≤1 at k=1/3, fading to near-zero by N=2+.

## Caveats
- N=0 k=0 is near-chance for all methods (pretrained init, no calibration, cross-domain) — benefit numbers there are on tiny absolute accuracies; read k=1/3.
- ~~Pooling analysis via ad hoc, uncommitted script~~ **Fixed 2026-07-13.** `scripts/main_experiment/analyze_a2_multiseed.py` reproduces this page's pooled table byte-for-byte from the raw per-seed `summary.csv` files (verified: diffed against the existing `a2_pooled_results.csv`, identical) and adds a paired-t companion (`a2_pooled_stats.csv`) — every p-value cited above (e.g. supLP120 N=1,k=3 p=2.07e-06; mae N=4,k=1 p=.0018) now traces to a committed, re-runnable script. `scripts/main_experiment/analyze_a2.py` remains single-seed-only by design (used for quick single-seed checks); it is not the source of this page's numbers.
- **This cold-start lever is target-representation-contingent** — see [[czu-dual-cold-start]]: repeating this exact sweep on CZU's strong (raw-signal) target finds no N where the prior beats scratch, and both priors are significantly *worse* than scratch at N=0, k=1 (the cell where Xsens shows the prior's biggest win). Read this page's headline claim as scoped to weak/moderate targets, not universal.

Related: [[phase1-mcnemar-ece-cka]] · [[multiseed-loso-v2]] · [[paper-framing]] · [[pretraining-objectives]]
