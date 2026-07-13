---
type: result
status: active
updated: 2026-07-10
---

# CZU-MHAD inertial LOSO — TRUE cross-modal external validity (skeleton→IMU)

The independent-public **cross-modal** replication: NTU skeleton prior → CZU wearable-**IMU**
target. Complements the same-modality [[czu-skeleton-loso]] (skeleton→skeleton). This is the
external analogue of the main NTU→Xsens path — a second, independent instance of the
skeleton→inertial gap on a public dataset.

- Run: `trained_models/CZU-IMU-LOSO{,-seed43,-seed44}/` (**3 seeds** 42/43/44, 5 subj LOSO,
  k=0/1/3 → 45 folds pooled). Data `Data_Processed/czu_imu_quats/` (1165 clips, 22 actions).
  Parser `temp_czu_imu_parser.py`.
- Launched via `temp_czu_imu_loso.sh` (seed 42) + `temp_czu_multiseed_run.sh` (seeds 43/44);
  CRC via `temp_czu_imu_crc_baseline.py`. Pooled stats: `temp_czu_multiseed_analyze.py`.

## How CZU inertial is made encoder-compatible (the representation bottleneck)

CZU `sensor_mat/` = 10 MPU9250 sensors, **6-axis accel+gyro only (no magnetometer in the
released data)**, async ~340–555 Hz, one spurious ~40 s timestamp gap per clip. The
NTU-pretrained encoder eats `(T,17,4)` **local-relative orientation quaternions**, so the parser:
1. Madgwick AHRS per sensor (accel+gyro → orientation; **yaw unobservable → drifts**).
2. SLERP-resample to 30 Hz; per-sensor initial-pose normalization (removes mounting + world-yaw offset).
3. Map 10 sensors → 17 segments (7 uninstrumented → identity local); global→parent-relative
   local quats (same hierarchy as `src/scripts/IMU_batch_processor.py`).

**Sensor index→body order was recovered empirically** from per-action gyro energy (not in the
.mat): `0=Abdomen,1=Chest,2=L Elbow,3=L Wrist,4=R Elbow,5=R Wrist,6=L Knee,7=L Ankle,8=R Knee,
9=R Ankle`. Validated per action (bend→abdomen; R-arm→R-arm segs; L-kick→L-leg segs).

**Key bottleneck:** matching the skeleton encoder forces an **orientation-only** representation —
the accelerometer magnitude signal (which the CRC baseline exploits) is discarded, and yaw drifts.
Absolute deep accuracy is therefore far below CRC. But scratch and all priors share this
representation, so the **prior-vs-scratch comparison stays valid**.

## Final accuracy — per-(method,k) mean, pooled 3 seeds × 5 subj (n=45)

| k | scratch | mae | **supMAE** | supLP120 | CRC (raw accel+gyro, 1 seed) |
|---|---|---|---|---|---|
| 0 | 55.72 | 54.88 | **57.58** | 53.60 | 86.84 |
| 1 | 54.25 | 52.97 | **54.16** | 51.92 | 90.84 |
| 3 | 58.31 | 57.36 | **59.56** | 55.99 | 94.42 |

Single-seed (seed-42) numbers were 59.3 / 56.3 / 61.1 for supMAE — the seed-42 supMAE run ran
high; pooling brings supMAE down to ≈scratch (see below).

## The headline — pooling reframes it: supervised prior best→worst, reconstruction is the antidote

Multi-seed pooling (seeds 42/43/44) sharpened *which* claim is real:

- Same-modality (CZU skeleton [[czu-skeleton-loso]], UTD skeleton [[utd-skeleton-loso]]):
  **supLP120 dominates** (+7.2 / +7.4 pp zero-shot, pooled p<.001) — pure-supervised prior wins.
- Cross-modal (CZU inertial, here): **supLP120 is the *worst* prior — below scratch at every k**;
  supMAE recovers to ≈scratch but does **not** beat it.

Per-fold sign tests, pooled over (subject, k, seed) cells (ties dropped):

| contrast | single-seed | **pooled (3 seeds)** | reading |
|---|---|---|---|
| supMAE > supLP120 | 13/15 (p=.007) | **36/44 (p<.0001)** | **robust — the load-bearing effect** |
| supLP120 < scratch | 10/15 | **28/43 (p=.066)**, meanΔ −2.3 pp | supervised prior hurts |
| supMAE > scratch | 12/15 (p=.035) | **25/42 (p=.28)** | **does NOT survive — was seed-42 luck** |

Paired-t by k (n=15): supLP120 − scratch = −2.1 (k0, p=.044) / −2.3 (k1, p=.13) / −2.3 (k3, p=.056);
supMAE − scratch = +1.9 (k0, p=.034) / −0.1 (k1) / +1.3 (k3, p=.29).

**Interpretation (pooled):** the robust cross-modal signal is **supervised-specific negative transfer
with reconstruction as the antidote** — not "reconstruction beats scratch." Pure-supervised priors
(supLP120) encode skeleton-specific class boundaries that misalign on IMU-derived orientations and
land *below* random init; adding MAE reconstruction (supMAE > supLP120, p<.0001) cancels that damage
back to ≈scratch. This is the external, cross-modal analogue of the mechanism story
([[multiseed-loso-v2]] C2/C3): supervised-only priors don't transfer across modality; reconstruction
regularizes toward transferable low-level kinematics. The single-seed "supMAE +3 pp" headline was an
artifact — the honest claim is the *inversion of prior ranking*, sign-test-robust at p<.0001.

**Boundary (see [[czu-imu-dual]], R6c):** even the supMAE≈scratch rescue is **contingent on the
impoverished target encoding**. When the target is given its full raw signal (raw scratch already
≈ CRC), no prior helps — pooled supMAE ties scratch (17/39, p=.52) and supLP120 is again worst
(6/41, **p<.0001**, paired-t significant every k). So R6b is the *middle* of a three-column arc
(small-gap supLP120 wins → weak cross-modal supLP120 hurts / supMAE rescues → strong cross-modal
no prior helps), not a standalone "reconstruction transfers to IMU" claim.

## supcon extension (2026-07-10, pooled 3 seeds n=15/k)

`trained_models/CZU-IMU-LOSO-supcon-seed{42,43,44}/summary.csv`. supcon − scratch = **−1.44 / −3.35
/ −0.24 pp @ k=0/1/3** (n.s. to marginal, k=1 p=.053); sign test supcon>scratch 19/45 (p=.37, a wash
trending negative — same reading as supLP120's own −2.3 pp mean here). supcon vs supLP120 is
statistically even (26/44, p=.29). **The two label-aware objectives (softmax supLP120, contrastive
supcon) land in the same negative-to-neutral band here**, distinctly below supMAE — the only
objective at this setting with a reconstruction component. This is the evidence that rules out
"something specific to softmax classification": what predicts failure at this gap is the *absence*
of reconstruction, not the presence of any particular supervised recipe. Feeds `paper_results.md`
R6b + the R6e five-setting map's supcon paragraph.

## Caveats (do not overclaim)

1. **The pp margin over scratch is not real; the ranking inversion is.** Pooled, supMAE does **not**
   beat scratch (25/42 folds, p=.28) — the single-seed +3 pp was seed-42 luck. What survives 3-seed
   pooling is the *prior-vs-prior* ordering (supMAE > supLP120, 36/44, p<.0001) and supLP120 <
   scratch (p≈.04–.06). Report the inversion of prior ranking, not a reconstruction-beats-scratch gap.
2. **Deep ≪ CRC (56–61% vs 87–94%)** — the orientation-only encoding (no accel, yaw drift) is a
   lossy bottleneck for inertial; raw-signal CRC is far stronger in absolute terms. Prior-vs-scratch
   comparison is valid (shared representation); absolute numbers are not a claim about IMU ceiling.
3. **Does NOT reproduce "mae = negative transfer."** On NTU→Xsens, mae is *the* negative-transfer
   culprit; here mae ≈ scratch and supLP120 is the one that hurts. Different pattern — the
   cross-modal negative transfer here is carried by the *supervised* prior, not mae.

## CRC published-baseline (raw inertial)

CRC on raw 6-axis statistical moments (mean/std/var/skew/kurtosis × 10 sensors × 6 ch = 300-dim),
same LOSO splits: **86.8 / 90.8 / 94.4%** @k0/1/3. Comparable to the CZU paper's cross-subject
inertial range (~65–85%, T5–T7); higher here (all-10-sensor moments + 4-subject LOSO dictionary).
`trained_models/CZU-IMU-LOSO/crc_baseline/`.

See also [[czu-imu-dual]] (R6c boundary) · [[czu-skeleton-loso]] · [[multiseed-loso-v2]] · [[loso-fulltrain-calibrate]].
