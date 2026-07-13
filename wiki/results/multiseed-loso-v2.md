---
type: result
status: active
updated: 2026-07-04c
---

# Multi-seed LOSO-v2 — accuracy, AUC & convergence stats

Closes the load-bearing stats debt for [[position-reconstruction-v2]]. Pools 3 seeds
(42/43/44) of the v2 LOSO run so the prior-vs-no-prior claim rests on more than one seed.

- Runs: `trained_models/LOSO-fullTrainCalibrate-v2/` (seed 42), `-v2-seed43/`, `-v2-seed44/`.
- Metric sources: `summary.csv` (final_acc) + `Logs/*_calibration_epochs.csv` (per-epoch eval_acc).
- Methods: scratch, mae, supMAE, supLP120. Pairing unit for deltas = (subject, seed).
- **n=15** (5 subj × 3 seeds) at k∈{0,1,3}. **k=5/k=10 are seed-42 only (n=5)** — seeds 43/44 ran only k∈{1,3}.
- Metric defs: AUC = normalized area (mean eval_acc %) under the calibration curve over first N epochs;
  convergence = first epoch where eval_acc ≥ 0.9 × final eval_acc (lower = faster).

## Headline: the single-seed +4.1 pp @ k=1 was a seed-42 artifact

Seed-by-seed supMAE − scratch (mean over 5 subj):

| k | seed42 | seed43 | seed44 | pooled |
|---|--------|--------|--------|--------|
| 0 | +0.63 | +4.05 | +2.86 | +2.52 |
| 1 | **+4.13** | **−1.77** | +0.56 | **+0.97** |
| 3 | +1.03 | +2.25 | +2.28 | +1.85 |

The k=1 headline collapses under pooling (seed 43 is negative). Robust prior benefit lives at k=0/k=3, not k=1.

## 1. Final accuracy — per-(method,k) mean ± sd (n=15)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 0 | 57.15±6.84 | 52.90±5.71 | **59.66±6.68** | 58.41±7.43 |
| 1 | 82.31±6.55 | 79.47±5.45 | **83.28±6.41** | 80.97±4.91 |
| 3 | 89.39±4.90 | 88.05±5.33 | **91.24±4.43** | 90.81±4.19 |

Paired Δ vs scratch (95% CI, paired-t p):

| | k=0 | k=1 | k=3 |
|---|-----|-----|-----|
| supMAE−scratch | +2.52 [−0.46,+5.49] p=.091 | +0.97 [−1.78,+3.72] p=.461 | +1.85 [−0.03,+3.74] p=.054 |
| supLP120−scratch | +1.27 p=.494 | −1.34 p=.375 | +1.42 p=.264 |
| mae−scratch | −4.24 [−6.78,−1.71] p=.003 | −2.84 [−4.43,−1.25] p=.002 | −1.34 p=.201 |

## 2. AUC-30 (mean eval_acc % over first 30 calib epochs)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 1 | 75.99 | 71.63 | 76.72 | 76.86 |
| 3 | 80.88 | 77.05 | 82.92 | **84.63** |
| 5 (n=5) | 83.00 | 78.13 | 83.77 | 87.20 |
| 10 (n=5) | 89.64 | 86.34 | 90.19 | 92.26 |

Paired Δ vs scratch:

| | k=1 | k=3 |
|---|-----|-----|
| supMAE−scratch | +0.72 p=.412 | +2.04 [+0.26,+3.81] **p=.028** |
| supLP120−scratch | +0.87 p=.527 | +3.74 [+1.03,+6.46] **p=.010** |
| mae−scratch | −4.37 [−5.95,−2.79] p<.001 | −3.83 [−5.20,−2.46] p<.001 |

(AUC-20 mirrors AUC-30: supMAE k=3 Δ+1.90 p=.071, supLP120 k=3 Δ+4.34 p=.008, mae negative everywhere.)

## 3. Convergence — epochs to ≥90% of final eval_acc (lower = faster)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 1 | 10.07 | 13.00 | 9.53 | **6.07** |
| 3 | 11.93 | 14.87 | 12.33 | **8.53** |

Paired Δ vs scratch:

| | k=1 | k=3 |
|---|-----|-----|
| supMAE−scratch | −0.53 p=.701 | +0.40 p=.686 |
| supLP120−scratch | −4.00 [−6.97,−1.03] **p=.012** | −3.40 [−5.07,−1.73] **p=.001** |
| mae−scratch | +2.93 [+0.51,+5.36] p=.021 | +2.93 [+0.26,+5.61] p=.034 |

## Per-subject supMAE−scratch @ k=1 (mean over 3 seeds)

sub7 +7.57, sub10 +0.70, sub9 −0.65, sub11 −0.46, sub8 −2.30. sub7 carries most of the k=1 mean.

## Not computed

**Clip-level McNemar** — the runs saved checkpoints + aggregate summaries only, no per-clip
prediction dumps. McNemar needs per-sample correct/incorrect vectors; requires an inference pass
over saved `base_ckpt` + `calibration_ckpt` (the Tier-B `--dump-posteriors` plumbing). Deferred.

Analysis scripts: `scratchpad_multiseed_stats.py`, `scratchpad_seedbreak.py`, `scratchpad_auc_stats.py` (repo root).

See also [[loso-fulltrain-calibrate]] · [[czu-skeleton-loso]] · [[position-reconstruction-v2]].
