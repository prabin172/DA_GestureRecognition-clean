#!/usr/bin/env python3
"""Phase 1 / debt 1: dump per-clip posteriors over EXISTING multi-seed v2 checkpoints.

No retraining. Pure forward pass. Reuses model/dataset code from
loso_fulltrain_calibration.py so architecture + preprocessing match training exactly.

For each (seed-dir, subject, method, k):
  - eval set = the split JSON's `eval_files` (identical across methods for a fixed
    seed/subject/k, so downstream McNemar pairing is valid).
  - ckpt: k==0 -> {sub}_{method}_baseFull_bestVal.pth (zero-shot base model)
          k>0  -> {sub}_{method}_k{k}_head_only_last.pth (calibrated)
Writes {seed_dir}/posteriors/{sub}_{method}_k{k}.csv with full logits (for post-hoc
temperature scaling / ECE) + true/pred labels (for McNemar).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Point the LOSO module at the v2 data BEFORE importing it (it reads LOSO_IMU_DIR at import).
os.environ.setdefault("LOSO_IMU_DIR", "Data_Processed/imu_quats_v2")

import scripts.main_experiment.loso_fulltrain_calibration as L  # noqa: E402
from src.models.kinematic_encoder import KinematicEncoder  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default full grid.
DEFAULT_SEED_DIRS = [
    "trained_models/LOSO-fullTrainCalibrate-v2",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed44",
]
DEFAULT_SUBJECTS = ["sub7", "sub8", "sub9", "sub10", "sub11"]
DEFAULT_METHODS = ["scratch", "supLP120", "supMAE", "mae", "supcon"]
DEFAULT_KS = [0, 1, 3]


def seed_from_dirname(seed_dir: Path) -> int:
    name = seed_dir.name
    if name.endswith("-seed43"):
        return 43
    if name.endswith("-seed44"):
        return 44
    return 42  # base v2 dir is seed 42


def eval_df_from_split(split_json: Path, index_df: pd.DataFrame) -> pd.DataFrame:
    meta = json.loads(split_json.read_text())
    eval_files = meta["eval_files"]
    df = index_df[index_df["file"].isin(eval_files)].copy()
    # preserve the split's file order for reproducibility
    order = {f: i for i, f in enumerate(eval_files)}
    df["__o"] = df["file"].map(order)
    return df.sort_values("__o").drop(columns="__o").reset_index(drop=True)


@torch.no_grad()
def dump_config(seed_dir: Path, subject: str, method: str, k: int,
                index_df: pd.DataFrame, label2id: dict, id2label: dict,
                imu_dir: Path, batch_size: int, limit: int | None) -> pd.DataFrame | None:
    num_classes = len(label2id)
    models_dir = seed_dir / "models"
    if k == 0:
        ckpt_path = models_dir / f"{subject}_{method}_baseFull_bestVal.pth"
    else:
        ckpt_path = models_dir / f"{subject}_{method}_k{k}_head_only_last.pth"
    split_json = seed_dir / "splits" / f"{subject}_k{k}_calibration_split.json"
    if not ckpt_path.exists() or not split_json.exists():
        print(f"  SKIP missing: {ckpt_path.name} / {split_json.name}")
        return None

    eval_df = eval_df_from_split(split_json, index_df)
    if limit is not None:
        eval_df = eval_df.head(limit).reset_index(drop=True)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    head = L.GestureHead(num_classes).to(DEVICE)
    head.load_state_dict(ckpt["head"])
    encoder.eval()
    head.eval()

    loader = DataLoader(
        L.IMUClipDataset(eval_df, label2id, imu_dir, L.MAX_FRAMES),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    all_logits, all_true = [], []
    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        z = L.masked_mean(encoder(x, mask=m), m)
        logits = head(z)
        all_logits.append(logits.cpu().numpy())
        all_true.append(y.numpy())
    logits = np.concatenate(all_logits, 0)
    true_id = np.concatenate(all_true, 0)

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred_id = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred_id == true_id).astype(int)

    out = pd.DataFrame({
        "seed": seed_from_dirname(seed_dir),
        "subject": subject,
        "method": method,
        "k": k,
        "file": eval_df["file"].values,
        "true_id": true_id,
        "true_label": [id2label[int(t)] for t in true_id],
        "pred_id": pred_id,
        "conf": conf,
        "correct": correct,
        "logits": [json.dumps([round(float(v), 5) for v in row]) for row in logits],
    })
    acc = 100.0 * correct.mean()
    print(f"  {seed_dir.name:>38} {subject} {method:>8} k{k}: n={len(out):4d} acc={acc:5.1f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump per-clip posteriors over saved LOSO-v2 checkpoints.")
    ap.add_argument("--seed-dirs", default=",".join(DEFAULT_SEED_DIRS))
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--imu-dir", default="Data_Processed/imu_quats_v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="Cap clips per config (smoke test only).")
    args = ap.parse_args()

    seed_dirs = [PROJECT_ROOT / d for d in args.seed_dirs.split(",")]
    subjects = args.subjects.split(",")
    methods = args.methods.split(",")
    ks = [int(k) for k in args.k_values.split(",")]
    imu_dir = PROJECT_ROOT / args.imu_dir
    index_df = pd.read_csv(imu_dir / "index.csv")

    for seed_dir in seed_dirs:
        lm_path = seed_dir / "label_map.json"
        if not lm_path.exists():
            print(f"SKIP {seed_dir} (no label_map.json)")
            continue
        label2id = json.loads(lm_path.read_text())
        id2label = {v: k for k, v in label2id.items()}
        post_dir = seed_dir / "posteriors"
        post_dir.mkdir(exist_ok=True)
        print(f"== {seed_dir.name} ==")
        for subject in subjects:
            for method in methods:
                for k in ks:
                    df = dump_config(seed_dir, subject, method, k, index_df,
                                     label2id, id2label, imu_dir, args.batch_size, args.limit)
                    if df is not None:
                        df.to_csv(post_dir / f"{subject}_{method}_k{k}.csv", index=False)
    print("DONE dump_posteriors")


if __name__ == "__main__":
    main()
