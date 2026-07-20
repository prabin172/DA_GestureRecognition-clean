#!/usr/bin/env python3
"""
MMD domain gap analysis.

For each pretrained encoder, compute multi-scale RBF-kernel MMD^2 between
NTU and Xsens/IMU feature distributions. Produces a ranked table showing
how well each pretraining objective aligns the two domains.

Use the results to operationalize Pillar 3: if low MMD correlates with
higher cross-domain k-shot accuracy, the mismatch (not objective choice)
is the bottleneck.

Output: trained_models/MMD_DomainGap/mmd_table.csv

Usage:
    .venv/bin/python mmd_domain_gap.py
    .venv/bin/python mmd_domain_gap.py --smoke
    .venv/bin/python mmd_domain_gap.py --include-supcon
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_CONFIGS = [
    {"tag": "scratch",  "ckpt": None},
    {"tag": "sup",      "ckpt": "trained_models/SUPERVISED/sup_lr_ntu120_epoch_50.pth"},
    {"tag": "mae",      "ckpt": "trained_models/MAE/mae_geoLoss_epoch_50.pth"},
    {"tag": "supmae",   "ckpt": "trained_models/SUPMAE/supmae_best.pth"},
]

SUPCON_CKPT = "trained_models/ContrastiveNTU/supcon_epoch_50.pth"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClipDataset(Dataset):
    def __init__(self, files, root, max_frames=120):
        self.files = list(files)
        self.root = Path(root)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        arr = np.load(self.root / Path(self.files[i]).name).astype(np.float32).reshape(-1, 68)
        T = min(arr.shape[0], self.max_frames)
        out = np.zeros((self.max_frames, 68), np.float32)
        mask = np.zeros(self.max_frames, np.float32)
        out[:T] = arr[:T]; mask[:T] = 1.0
        return torch.from_numpy(out), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_encoder(ckpt_path):
    enc = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    if ckpt_path is None:
        return enc
    p = PROJECT_ROOT / ckpt_path if not Path(ckpt_path).is_absolute() else Path(ckpt_path)
    ckpt = torch.load(p, map_location=DEVICE)
    for key in ("encoder_state_dict", "encoder", "state_dict"):
        if key in ckpt:
            state = {k.removeprefix("module.").removeprefix("encoder."): v
                     for k, v in ckpt[key].items()}
            enc.load_state_dict(state, strict=False)
            break
    return enc


def masked_mean(x, mask):
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(1, keepdim=True).clamp(1.0)
    return (x * w).sum(1) / denom


@torch.no_grad()
def extract_features(enc, files, root, n_samples, batch_size):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(files), min(n_samples, len(files)), replace=False)
    chosen = [files[i] for i in idx]
    ds = ClipDataset(chosen, root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    enc.eval()
    parts = []
    for x, mask in dl:
        x = x.to(DEVICE); mask = mask.to(DEVICE)
        parts.append(masked_mean(enc(x, mask=mask), mask).cpu())
    return torch.cat(parts, dim=0)


def rbf_mmd_sq(X, Y, sigmas=(0.5, 1.0, 2.0, 5.0)):
    """Multi-scale RBF kernel MMD^2. Averaged across sigma values."""
    X = X.float(); Y = Y.float()
    XX = torch.cdist(X, X).pow(2)
    YY = torch.cdist(Y, Y).pow(2)
    XY = torch.cdist(X, Y).pow(2)
    total = 0.0
    for s in sigmas:
        sc = 2 * s ** 2
        total += ((-XX / sc).exp().mean()
                  + (-YY / sc).exp().mean()
                  - 2 * (-XY / sc).exp().mean()).item()
    return total / len(sigmas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="MMD domain gap analysis between NTU and Xsens features.")
    ap.add_argument("--ntu-root", type=Path, default=Path("Data_Processed/ntu_quats"))
    ap.add_argument("--xsens-root", type=Path, default=Path("Data_Processed/imu_quats_v2"))
    ap.add_argument("--xsens-index", type=Path, default=Path("Data_Processed/imu_quats_v2/index.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("trained_models/MMD_DomainGap"))
    ap.add_argument("--n-samples", type=int, default=500, help="Clips sampled per domain per encoder")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--include-supcon", action="store_true",
                    help="Include supcon checkpoint (only available after pretraining completes)")
    ap.add_argument("--smoke", action="store_true", help="Quick smoke: 50 samples, batch=32")
    args = ap.parse_args()

    if args.smoke:
        args.n_samples = 50
        args.batch_size = 32
        args.out_dir = Path("trained_models/MMD_DomainGap-smoke")

    ntu_root = args.ntu_root if args.ntu_root.is_absolute() else PROJECT_ROOT / args.ntu_root
    xsens_root = args.xsens_root if args.xsens_root.is_absolute() else PROJECT_ROOT / args.xsens_root
    xsens_idx = args.xsens_index if args.xsens_index.is_absolute() else PROJECT_ROOT / args.xsens_index
    out = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ntu_files = [f.name for f in sorted(ntu_root.glob("*.npy"))]
    xsens_df = pd.read_csv(xsens_idx)
    xsens_df = xsens_df[xsens_df["label"].astype(str) != "unknown"]
    if "n_frames_30hz" in xsens_df.columns:
        xsens_df = xsens_df[xsens_df["n_frames_30hz"] >= 8]
    xsens_files = xsens_df["file"].tolist()

    configs = list(ENCODER_CONFIGS)
    if args.include_supcon:
        configs.append({"tag": "supcon", "ckpt": SUPCON_CKPT})

    print(f"NTU pool: {len(ntu_files)} | Xsens pool: {len(xsens_files)} | "
          f"n_samples: {args.n_samples} | device: {DEVICE}")

    records = []
    for cfg in configs:
        tag, ckpt = cfg["tag"], cfg["ckpt"]
        if ckpt and not (PROJECT_ROOT / ckpt).exists():
            print(f"Skipping {tag}: checkpoint not found ({ckpt})")
            continue
        print(f"\n[{tag}] extracting features...")
        enc = load_encoder(ckpt)
        Z_ntu = extract_features(enc, ntu_files, ntu_root, args.n_samples, args.batch_size)
        Z_xsens = extract_features(enc, xsens_files, xsens_root, args.n_samples, args.batch_size)
        mmd = rbf_mmd_sq(Z_ntu, Z_xsens)
        print(f"  MMD^2 = {mmd:.6f}  (n_ntu={len(Z_ntu)}, n_xsens={len(Z_xsens)})")
        records.append({"method": tag, "mmd_sq": round(mmd, 6),
                        "n_ntu": len(Z_ntu), "n_xsens": len(Z_xsens)})
        del enc; torch.cuda.empty_cache()

    df = pd.DataFrame(records).sort_values("mmd_sq")
    out_csv = out / "mmd_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n=== MMD Domain Gap Results ===\n{df.to_string(index=False)}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
