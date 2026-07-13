---
type: concept
status: active
updated: 2026-07-10
---

# Pre-training objectives

The independent variable of the whole project. All source pretraining is on [[ntu-dataset|NTU]] unless noted. Scripts in `src/scripts/pretrain/` — see [[pretrain-scripts]]; checkpoints cataloged in [[ntu-pretraining]].

| objective | what it is | verdict so far |
|---|---|---|
| **MAE** | mask ~70% of frames, reconstruct (MSE or geodesic quat loss `1 - \|q·q̂\|`) | weak alone: 22.6% NTU→NTU; frozen target-MAE collapses (58–73%) in [[xsens-to-xsens-loso]]; the only objective with a reconstruction term, and the only one that survives the large cross-modal gap — but the seed-stable negative-transfer culprit on the Xsens (middle-gap) path (McNemar p<.001 every k, [[phase1-mcnemar-ece-cka]]) |
| **Supervised / supLP120** | cross-entropy on 120 NTU labels (or 23-class relevant subset) | best within-domain (59.6%); mixed/negative transfer at low k on Xsens (middle gap); best-calibrated + best CKA-aligned; **wins big on same-modality external targets** (CZU/UTD skeleton, +7–8 pp p<.001) but **worst prior on both cross-modal IMU settings** (below scratch, p<.0001) — hero→villain as the gap widens ([[czu-imu-dual]] R6e) |
| **SupMAE** | multi-task MAE + CE, shared encoder, CE warm-up + EMA loss balancing | best cross-domain at low k on Xsens (leads k=1/3); the only objective that rescues cross-modal negative transfer (supMAE > supLP120 p<.0001 on CZU-IMU-quat) — because it's the only one with a reconstruction term; NOTE: existing vision method — must cite or rename (§A1) |
| **SupCon** | supervised contrastive (Khosla 2020) on NTU | R1: NTU→Xsens accuracy is a wash vs scratch (like supLP120), best/near-best CKA alignment of all five encoders yet a wash on accuracy — the sharpest alignment≠accuracy dissociation in the project; 2nd-best-calibrated after supLP120. **Externally, tracks supLP120 almost exactly**: wins big at both small-gap datasets (CZU +7.0/+4.8/+4.4pp p<.001, beating supLP120 itself p=.0002; UTD +8.4/+4.0/+2.9pp p≤.006, tied with supLP120), loses/washes at both large-gap cross-modal settings — same pattern as supLP120, not distinct from it. (Earlier verdict — "underperforms scratch, scoped out" — was the single-seed, Xsens-only read; superseded by the T2 full-parity pass, 2026-07-10. See [[phase1-mcnemar-ece-cka]], [[czu-skeleton-loso]], [[utd-skeleton-loso]], [[czu-imu-dual]].) |
| **DANN variants** | gradient-reversal domain adversary aligning NTU↔Xsens | mid-pack; stale under swing (not re-run). See [[dann-experiments]] |
| **scratch** | no pretraining — random init | the baseline every prior is judged against; competitive at low subject-count, and the only thing that doesn't hurt on strong cross-modal targets ([[czu-imu-dual]] R6c, [[czu-dual-cold-start]]) |

## The through-line
Within-domain the spread is ~45 pp; cross-domain at k=1 on Xsens it is ~4–7 pp. Objective choice matters where representations are compatible; the domain gap compresses everything ([[paper-framing]] Pillar 2). Low [[domain-gap-metrics|MMD]]/CKA alignment is necessary-not-sufficient: supcon has the highest CKA of all five encoders yet a wash for accuracy; mae has higher MMD than supervised but similar k=1 accuracy on Xsens. Extending to two label-aware objectives (supLP120, supcon) and four external settings reframes the real axis: it isn't "supervised (softmax) vs reconstruction" — it's **"has a reconstruction component (supMAE, mae) vs doesn't (supLP120, supcon)."** The two label-only objectives track each other closely everywhere; supMAE is the one objective that doesn't collapse at the large cross-modal gap. See [[czu-imu-dual]] R6e for the full five-setting map.
