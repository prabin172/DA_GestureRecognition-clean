---
type: result
status: active
updated: 2026-07-03
---

# Literature landscape (Undermind deep-research, 2026-07-03)

Source: `Undermind - Studies on cross-modal HAR pretraining objectives...(1).pdf` (repo root, 40 pp, free tier — 173 refs). Online: https://app.undermind.ai/report/d4cd1404f846f003d92027229ece442552832214cbb5c7a4965f62d3295a1b89

## Headline (the gap is OURS)
**No existing study directly compares supervised vs reconstruction vs hybrid skeleton/vision pretraining for skeleton→IMU HAR under systematically varied domain gaps** (within- vs cross-modal, subject-held-out few-shot, novel classes). The field *trends* toward our hypothesis but has never measured it: under low gaps supervised skeleton pretraining helps; as gaps grow, researchers *abandon* pure supervision for reconstruction/contrastive/hybrid objectives — "the field has voted with its feet" — yet **explicit negative-transfer measurement is missing**. This matches the [[paper-framing]] A1 characterization exactly.

## Four clusters of prior work

### 1. Direct skeleton/pose→IMU transfer (closest competitors)
| work | what it did | what it lacks |
|---|---|---|
| Moya Rueda & Fink (ICPR 2020) | supervised pose-sequence pretraining → IMU fine-tune, gains on 3 benchmarks | low-gap only; no objective comparison, no OOD/novel-class |
| Awasthi et al. (ICPR 2022) | multi-dataset video-pose supervised pretraining → IMU | same |
| PSKD, Ni et al. (ACM MM 2022, 40 cites) | progressive skeleton→accelerometer KD | fully supervised framework; in-distribution eval |
| TAS, Qiao/Maekawa (PerCom 2026 + UbiComp 2025 prelim) | teacher–assistant–student skeleton→single-IMU; adds dense temporal contrastive + attention transfer *because supervised KD is insufficient* | architecture focus; no objective ablation |
| SKELAR (SenSys 2025) | skeleton pretraining via **coarse joint-angle reconstruction**, attention matching to IMU/WiFi; MASD dataset (20 subj, 27 activities) | adopts reconstruction from the outset — never compares against supervised |
| Zolfaghari et al. (ABC 2024) | pose→sensor generator jointly trained with classification (hybrid recon+sup) on MM-Fit | doesn't isolate reconstruction vs supervised benefit |

### 2. Virtual-IMU synthesis (data augmentation, not representation transfer)
IMUTube (IMWUT 2020, 112 cites), "Yet it moves" (Rey 2020), Video2IMU (BSN 2022), CROMOSim (CPHS 2022 / TMC 2024), Multi³Net / Multi3Net+ (ISWC 2024 / 2025), MuJo+FiMAD (PerCom 2025), Xiao virtual wearable sensors (2020). All **reconstruction-style pose→IMU mappings**; consistently note synthetic-trained models need a small amount of real IMU for calibration. No objective comparisons.

### 3. Multimodal contrastive / foundation models
IMU2CLIP (2022, 58 cites), IMG2IMU (2022 / TMC 2025: image-pretrained beats sensor-pretrained by ~9.6 F1), UniMTS (2024, 35 cites), AURA-FM (2025), MoBind (2026), C3T UMA (2024), Cheshmi IMU–video SSL (2025, incl. Parkinson's cohort), DCAT (2025), Yang 2023 / Choi 2023 (skeleton+IMU contrastive). Strong zero/few-shot OOD via **contrastive/alignment objectives**; supports "class-supervised pretraining shapes latent spaces too narrowly" but none compare against supervised pretraining on the same data.

### 4. Within-IMU SSL benchmarks — the direct empirical anchors
- **BenchHAR (2026)**: 8 SSL methods, ~258K samples, cross-dataset OOD. **Hybrid reconstruction+contrastive wins generalization**; more unlabeled pretraining data beats more labeled downstream data.
- **PRIMUS (ICASSP 2025)**: hybrid self-sup + multimodal + nearest-neighbor supervision; up to 15% gains with <500 labels/class on OOD.
- Earlier: Haresamudram masked-reconstruction (ISWC 2020, 150 cites), LIMU-BERT (SenSys 2021, 185 cites), wrist-accelerometer SSL (2021).
These prove objective-vs-OOD comparisons matter **within IMU**; nobody extended them to the skeleton→IMU modality gap.

## Design takeaways the report prescribes (we already satisfy ✓)
1. Both low-gap and high-gap conditions ✓ ([[sanity-checks]] NTU→NTU + [[xsens-to-xsens-loso]] Xsens→Xsens vs [[loso-fulltrain-calibrate]] NTU→Xsens).
2. ≥3 pretraining regimes (supervised / reconstruction / hybrid) ✓ + SupCon + DANN.
3. Measure **signs of negative transfer** vs scratch baseline ✓ (supLP120 < scratch at low k) — and analyze representation properties ([[mmd-domain-gap]]; CKA pending).
4. Subject-held-out few-shot + novel-class onboarding ✓ ([[loso-protocol]], [[oov-leave-class-out]]).

## Venue precedent
THMS has published SSL-HAR: Guo & Li "Cross-Model Cross-Stream Learning" (THMS 2023, SkeletonBYOL); Rahimi Taghanaki et al. time-frequency contrastive HAR (THMS 2022, 19 cites). Venue is receptive to this paper shape.

## What the report did NOT cover (free tier + query scope)
- THMS statistical practice at N=5 — still guided only by [[publishability-review]] item 4.
- Controller / recognizer-in-the-loop literature — needs a separate targeted search before writing the controller section ([[paper-framing]] A0).
- Reviewer-defusal precedents.

Feeds directly into `paper_idea.md` (repo root) and [[paper-framing]].
