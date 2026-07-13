---
type: code
status: active
updated: 2026-07-03
---

# Model architectures (`src/models/`)

## KinematicEncoder — `src/models/kinematic_encoder.py`
The **primary encoder** in almost all experiments. `(B, T, 68)` → Linear 68→512 → learnable pos-emb `(1, 120, 512)` → 3-layer TransformerEncoder (norm_first, 8 heads, 4× FFN). `pool=False` → `(B, T, 512)` (MAE); `pool=True` → `(B, 512)` L2-normalized (classification). Accepts padding mask `(B, T)`, 1=real.

## KinematicDecoder — `src/models/kinematic_decoder.py`
MAE-only. 1-layer TransformerEncoder + linear back to 68-d: `(B, T, 512)` → `(B, T, 68)`.

## DSTformerQuatEncoder — `src/models/dstformer_quat_encoder.py`
Newer MotionBERT-style encoder; drop-in replacement used in some LOSO experiments ([[early-experiments]]). Per-joint embed 4→256, segment+temporal pos-emb, 5 blocks of dual-branch spatial↔temporal attention with learned soft gate, DropPath; LayerNorm → project to 512 → mean over joints → `(B, T, 512)`. Explicitly models joint interactions; more compute than KinematicEncoder.

## LabelEncoder — `src/models/label_encoder.py`
MIL/experimental. Word2Vec label embedding `(B, 300)` → `(B, 512)` L2-normalized, for motion↔semantic alignment.

## DANN heads
Gradient-reversal domain discriminator on pooled encoder output — defined inside the DANN pretrain scripts, not `src/models/`. See [[dann-experiments]].

## Checkpoint conventions
`trained_models/{EXPERIMENT}/*_best.pth` / `*_latest.pth`; keys `encoder_state_dict`, `decoder_state_dict`, `head_state_dict` + metadata (`epoch`, `avg_total`, `ntu_root`, `num_classes`). `verify_models.py` checks loadability.
