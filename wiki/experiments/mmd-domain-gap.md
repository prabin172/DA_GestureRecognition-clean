---
type: experiment
status: done
updated: 2026-07-03
---

# MMD domain-gap analysis

Squared MMD between NTU and Xsens features in each encoder's space ([[domain-gap-metrics]] — method + its three flaws). Script: `scripts/main_experiment/mmd_domain_gap.py`. Apples-to-apples across modes: same encoders, same NTU pool, n=500; only Xsens data changed.

## Local — `trained_models/MMD_DomainGap/mmd_table.csv` (2026-06-29)
| encoder | MMD² |
|---|---|
| sup | 0.0053 |
| supcon | 0.0058 |
| supMAE | 0.0109 |
| MAE | 0.0204 |
| scratch | 0.0514 |

## Local vs swing vs symmetric-swing vs v2 (2026-07-01 / 2026-07-04)
Dirs: `MMD_DomainGap-swing/`, `-symmetricSwing/`, `-v2/`. n=500, local NTU.
| encoder | local | swing | symmetric swing | **v2 (positions)** |
|---|---|---|---|---|
| sup | 0.0053 | 0.0054 | 0.0054 | **0.0047** |
| supmae | 0.0109 | 0.0322 | 0.0322 | **0.0092** |
| mae | 0.0204 | 0.0634 | 0.0692 | **0.0182** |
| scratch | 0.0515 | 0.1791 | 0.1643 | **0.0415** |

- **Symmetric swing** (swing-project NTU too, `mmd_domain_gap_symmetric.py (dead, not in this repo)`) ≈ asymmetric swing → the gap increase was NOT the asymmetry; twist-stripping damaged Xsens. NTU-local is already ~twist-free (projecting it was ~a no-op).
- **v2** (position-derived Xsens, `scripts/main_experiment/mmd_domain_gap.py --xsens-root Data_Processed/imu_quats_v2`) is **below local for every encoder** → the real fix. See [[position-reconstruction-v2]].

## Findings
- **Swing INCREASED the NTU↔Xsens gap ~3×** — opposite of the branch hypothesis ([[swing-mode]]). Partly built-in: twist removed from Xsens only (asymmetric test).
- Supervised NTU pretraining gives the smallest gap; SupCon matches it yet transfers worst ([[loso-fulltrain-calibrate]]) → **low MMD necessary-not-sufficient**.
- Don't interpret the scratch row (random-init encoder ≠ domain gap).
- Planned replacement: CKA + median-heuristic MMD + bootstrap CIs + swing-projected NTU — [[open-questions]].
