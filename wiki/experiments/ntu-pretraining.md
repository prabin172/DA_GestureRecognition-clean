---
type: experiment
status: done
updated: 2026-07-03
---

# NTU pretraining runs (the encoder zoo)

The pretrained encoders every downstream experiment initializes from. Objectives: [[pretraining-objectives]]; scripts: [[pretrain-scripts]].

| checkpoint dir | objective |
|---|---|
| `trained_models/MAE/` | MAE (geodesic) |
| `trained_models/SUPMAE/` (`supmae_best.pth`, `supmae_latest.pth`) | SupMAE |
| `trained_models/SUPERVISED/` | supervised NTU (supLP120 / relevant-23) |
| `trained_models/ContrastiveNTU/supcon_epoch_50.pth` | SupCon (Khosla 2020) |
| `trained_models/SUPERVISED_DANN/` + DANN dirs | DANN variants — [[dann-experiments]] |
| `trained_models/pretrain_cleaned_slerp035*/` | pretraining on cleaned NTU — [[cleaned-source-pretraining]] |
| `trained_models/ContrastiveNTU-smoke/`, `*_smoke*` dirs | smoke runs, ignore |

⚠️ All of these were trained on **local-mode** representations. Under [[swing-mode]] the DANN encoders are stale (per-fold, target-dependent) and were excluded from swing Job 1; MAE/SupMAE/supervised NTU encoders are target-independent and were reused as inits.
