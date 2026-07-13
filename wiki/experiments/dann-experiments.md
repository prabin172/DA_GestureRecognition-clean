---
type: experiment
status: stale
updated: 2026-07-03
---

# DANN experiments

Domain-Adversarial Neural Networks: gradient-reversal domain discriminator on the pooled encoder output, aligning NTU (source) and Xsens (target) distributions. Scripts in [[pretrain-scripts]].

| variant | dir | design |
|---|---|---|
| sourceSupDANN all120 | `SOURCE_SUPERVISED_DANN_ALL120_PRETRAIN_ADAPT_LOSO/` | NTU 120 labels + adversary vs unlabeled IMU (4 subj, 1 held out), then adapt |
| sourceSupDANN relevant23 | `SOURCE_SUPERVISED_DANN_RELEVANT23_PRETRAIN_ADAPT_LOSO/` | same, 23 relevant NTU classes |
| targetSupDANN | `TARGET_SUPERVISED_DANN_LOSO/`, `_RELEVANT23_LOSO/` | labeled IMU used during DANN pretraining |
| two-heads | `SOURCE_TARGET_SUPERVISED_DANN_TWOHEADS_ALL120_LOSO/` | source + target heads, shared encoder, adversary |
| diagnostics | `KSHOT_DANN_DIAGNOSTICS_LOSO/` | k × λ ablation |
| base | `SUPERVISED_DANN/` | supervised NTU + adversary |

## Standing in the results
Local-mode [[loso-fulltrain-calibrate]]: targetSupDANN ≈ supMAE at k=1 (79.8 vs 79.9); sourceSupDANN mid-pack; relatedness (all120 vs relevant23) doesn't rescue supervised transfer — supports the "mismatch, not label relatedness" pillar.

## Status: stale under swing
DANN encoders are per-fold and were trained on **local-mode** data — excluded from swing Job 1. Re-adapt only if swing becomes the chosen preprocessing ([[open-questions]]).
