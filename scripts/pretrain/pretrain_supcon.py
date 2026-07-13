#!/usr/bin/env python3
"""
Supervised Contrastive (SupCon) pretraining on NTU-120 quaternion data.

Pretrain KinematicEncoder with SupCon loss (Khosla et al., 2020) using two
augmented views per clip (temporal crop + small quaternion noise). Projection
head is discarded after training; encoder is saved as {"encoder_state_dict": ...}
so it plugs directly into loso_fulltrain_calibration.py via:

    --extra-method supcon=trained_models/ContrastiveNTU/supcon_epoch_50.pth

Usage:
    .venv/bin/python pretrain_supcon.py
    .venv/bin/python pretrain_supcon.py --smoke
"""

from __future__ import annotations
import argparse, random, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_RE = re.compile(r"A(\d{3})")


# ---------------------------------------------------------------------------
# Dataset — returns two augmented views per clip
# ---------------------------------------------------------------------------

class NTUAugDataset(Dataset):
    def __init__(self, files, root, max_frames=120, crop_min=0.7, noise_std=0.02, seed=0):
        self.files = [Path(f) for f in files]
        self.root = Path(root)
        self.max_frames = max_frames
        self.crop_min = crop_min
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def _augment(self, arr, T):
        """Random temporal crop + quaternion noise → renormalize."""
        crop_len = int(self.rng.integers(max(1, int(T * self.crop_min)), T + 1))
        start = int(self.rng.integers(0, max(1, T - crop_len + 1)))
        out = np.zeros((self.max_frames, 68), np.float32)
        new_mask = np.zeros(self.max_frames, np.float32)
        out[:crop_len] = arr[start:start + crop_len]
        new_mask[:crop_len] = 1.0
        noise = self.rng.normal(0, self.noise_std, out.shape).astype(np.float32)
        noisy = (out + noise * new_mask[:, None]).reshape(-1, 17, 4)
        norms = np.linalg.norm(noisy, axis=-1, keepdims=True).clip(1e-8)
        noisy = (noisy / norms).reshape(self.max_frames, 68)
        return noisy, new_mask

    def __getitem__(self, i):
        f = self.files[i]
        arr = np.load(self.root / f.name).astype(np.float32).reshape(-1, 68)
        T = min(arr.shape[0], self.max_frames)
        arr = arr[:T]
        m = A_RE.search(f.stem)
        y = int(m.group(1)) - 1
        v1, m1 = self._augment(arr, T)
        v2, m2 = self._augment(arr, T)
        return (
            torch.from_numpy(v1), torch.from_numpy(m1),
            torch.from_numpy(v2), torch.from_numpy(m2),
            torch.tensor(y, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Projection head (discarded after pretraining)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), p=2, dim=-1)


# ---------------------------------------------------------------------------
# SupCon loss (Khosla et al., 2020)
# ---------------------------------------------------------------------------

def supcon_loss(features, labels, temperature=0.07):
    """
    features: (2*B, D) L2-normalized; first B = view1, second B = view2
    labels:   (2*B,)
    """
    N = features.shape[0]
    sim = torch.mm(features, features.T) / temperature

    self_mask = torch.eye(N, dtype=torch.bool, device=features.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_eq & ~self_mask

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos_mask.sum(1).float().clamp(1.0)
    # use where() to avoid -inf * 0 = nan at masked diagonal entries
    loss = -(torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))).sum(1) / n_pos
    return loss.mean()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def masked_mean(x, mask):
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(1, keepdim=True).clamp(1.0)
    return (x * w).sum(1) / denom


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_files(root, limit_per_class, seed):
    all_files = sorted(root.glob("*.npy"))
    if not limit_per_class:
        return all_files
    by_class = defaultdict(list)
    for f in all_files:
        m = A_RE.search(f.stem)
        if m:
            by_class[int(m.group(1))].append(f)
    rng = np.random.default_rng(seed)
    out = []
    for cls_files in by_class.values():
        cls_files = list(cls_files)
        rng.shuffle(cls_files)
        out.extend(cls_files[:limit_per_class])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    root = args.ntu_root if args.ntu_root.is_absolute() else PROJECT_ROOT / args.ntu_root
    out = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    files = collect_files(root, args.limit_per_class, args.seed)
    print(f"SupCon NTU pretrain | files={len(files)} epochs={args.epochs} "
          f"batch={args.batch_size} temp={args.temperature} device={DEVICE}")

    ds = NTUAugDataset(files, root, crop_min=args.crop_min,
                       noise_std=args.noise_std, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    enc = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    proj = ProjectionHead().to(DEVICE)
    opt = optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
                      lr=args.lr, weight_decay=0.05)
    params = list(enc.parameters()) + list(proj.parameters())

    for ep in range(1, args.epochs + 1):
        enc.train(); proj.train()
        loss_sum = 0.0; n = 0
        for x1, m1, x2, m2, y in tqdm(dl, desc=f"SupCon epoch {ep}/{args.epochs}"):
            x1 = x1.to(DEVICE); m1 = m1.to(DEVICE)
            x2 = x2.to(DEVICE); m2 = m2.to(DEVICE)
            y = y.to(DEVICE)
            B = x1.shape[0]
            opt.zero_grad(set_to_none=True)
            # No AMP: SupCon loss is already float32 and is numerically sensitive
            # at temperature=0.07; AMP overflow in encoder corrupts parameters
            z1 = proj(masked_mean(enc(x1, mask=m1), m1))
            z2 = proj(masked_mean(enc(x2, mask=m2), m2))
            feats = torch.cat([z1, z2], dim=0)
            lbls = torch.cat([y, y], dim=0)
            loss = supcon_loss(feats, lbls, args.temperature)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            loss_sum += float(loss.item()) * B; n += B
        print(f"Epoch {ep}/{args.epochs} | loss={loss_sum / max(n, 1):.4f}")

    ckpt_path = out / f"supcon_epoch_{args.epochs}.pth"
    torch.save({"encoder_state_dict": enc.state_dict()}, ckpt_path)
    print(f"Saved encoder checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SupCon pretraining on NTU-120 quaternion sequences.")
    ap.add_argument("--ntu-root", type=Path, default=Path("Data_Processed/ntu_quats"))
    ap.add_argument("--out-dir", type=Path, default=Path("trained_models/ContrastiveNTU"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--crop-min", type=float, default=0.7)
    ap.add_argument("--noise-std", type=float, default=0.02)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke test: 2 epochs, 5 clips/class, batch=64")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2
        args.limit_per_class = 5
        args.batch_size = 64
        args.num_workers = 2
        args.out_dir = Path("trained_models/ContrastiveNTU-smoke")
    train(args)


if __name__ == "__main__":
    main()
