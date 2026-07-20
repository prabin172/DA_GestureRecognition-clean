#!/usr/bin/env python3
"""Phase 1 / debt 2: layer-wise CKA (linear + RBF), NTU vs Xsens-v2, per encoder objective.

Upgrades the C3 "representation mismatch" claim from ASSERTED to SHOWN: for each
pretraining objective (scratch, supLP120, supMAE, mae), measure how aligned the
NTU-skeleton and Xsens-v2-IMU representations are at each encoder depth.

Reuses the KinematicEncoder + the data/encoder-loading pattern from
mmd_domain_gap_symmetric.py (dead, not carried over). Only v2 Xsens is used (swing/local excluded by decision).

Outputs under --out-dir (default trained_models/Phase1-analysis):
  - cka_results.csv     : (encoder, layer, cka_linear, cka_rbf)
  - cka_heatmap.png     : encoders x layers, linear CKA
  - cka_vs_benefit.png  : per-encoder mean CKA vs k=1 cross-domain accuracy benefit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES = 120
MIN_FRAMES = 8

ENCODER_CONFIGS = [
    {"tag": "scratch", "ckpt": None},
    {"tag": "supLP120", "ckpt": "trained_models/SUPERVISED/sup_lr_ntu120_epoch_50.pth"},
    {"tag": "supMAE", "ckpt": "trained_models/SUPMAE/supmae_best.pth"},
    {"tag": "mae", "ckpt": "trained_models/MAE/mae_geoLoss_epoch_50.pth"},
    {"tag": "supcon", "ckpt": "trained_models/ContrastiveNTU/supcon_epoch_50.pth"},
]
LAYER_NAMES = ["proj", "L0", "L1", "L2"]
SEED_DIRS = [
    "trained_models/LOSO-fullTrainCalibrate-v2",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed44",
]

# T3: gap-axis measurement. Each target claimed by paper_results.md R6e as small/middle/large
# gap from NTU. All share the (T,17,4) LRQ schema so the same loader works for every one.
TARGETS = [
    ("xsens_v2", "Data_Processed/imu_quats_v2"),
    ("czu_skeleton", "Data_Processed/czu_skeleton_lrq"),
    ("czu_imu_quat", "Data_Processed/czu_imu_quats"),
    ("utd_skeleton", "Data_Processed/utd_skeleton_lrq"),
]


class ClipDataset(Dataset):
    """Loads (T,17,4) npy -> (120,68) padded + mask. Root-relative filenames."""
    def __init__(self, files: list[str], root: Path):
        self.files = files
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        clip = np.load(self.root / self.files[i]).astype(np.float32)
        t = clip.shape[0]
        clip = clip.reshape(t, 68)
        out = np.zeros((MAX_FRAMES, 68), dtype=np.float32)
        mask = np.zeros((MAX_FRAMES,), dtype=np.int64)
        use = min(t, MAX_FRAMES)
        out[:use] = clip[:use]
        mask[:use] = 1
        return torch.from_numpy(out), torch.from_numpy(mask)


def load_encoder(ckpt: str | None) -> KinematicEncoder:
    enc = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    if ckpt is None:
        enc.eval()
        return enc
    p = PROJECT_ROOT / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
    obj = torch.load(p, map_location=DEVICE)
    for key in ("encoder_state_dict", "encoder", "state_dict"):
        if key in obj:
            state = {k.removeprefix("module.").removeprefix("encoder."): v
                     for k, v in obj[key].items()}
            enc.load_state_dict(state, strict=False)
            break
    enc.eval()
    return enc


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(1, keepdim=True).clamp(1.0)
    return (x * w).sum(1) / denom


@torch.no_grad()
def extract_layer_feats(enc: KinematicEncoder, files: list[str], root: Path,
                        batch_size: int) -> dict[str, np.ndarray]:
    """Return {layer_name: (N,512)} masked-mean-pooled activations at each depth."""
    captured: dict[str, torch.Tensor] = {}
    handles = []
    handles.append(enc.input_projection.register_forward_hook(
        lambda m, i, o: captured.__setitem__("proj", o)))
    for li in range(len(enc.transformer_encoder.layers)):
        handles.append(enc.transformer_encoder.layers[li].register_forward_hook(
            (lambda name: lambda m, i, o: captured.__setitem__(name, o))(f"L{li}")))

    ds = ClipDataset(files, root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    acc: dict[str, list] = {n: [] for n in LAYER_NAMES}
    for x, mask in dl:
        x = x.to(DEVICE); mask = mask.to(DEVICE)
        captured.clear()
        enc(x, mask=mask)
        for name in LAYER_NAMES:
            acc[name].append(masked_mean(captured[name], mask).cpu().numpy())
    for h in handles:
        h.remove()
    return {n: np.concatenate(acc[n], 0) for n in LAYER_NAMES}


# ---- CKA ----
def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    u = K.mean(0, keepdims=True)
    return K - u - u.T + K.mean()


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    return float((_center(K) * L).sum())


def _cka(K: np.ndarray, L: np.ndarray) -> float:
    denom = np.sqrt(_hsic(K, K) * _hsic(L, L))
    return _hsic(K, L) / denom if denom > 0 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    return _cka(X @ X.T, Y @ Y.T)


def rbf_cka(X: np.ndarray, Y: np.ndarray) -> float:
    def gram(Z):
        sq = np.maximum(0.0, (Z * Z).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2 * Z @ Z.T)
        med = np.median(sq[np.triu_indices_from(sq, k=1)])
        sigma2 = med if med > 0 else 1.0
        return np.exp(-sq / (2 * sigma2))
    return _cka(gram(X), gram(Y))


def load_target_files(root: Path, n_samples: int, rng: np.random.Generator) -> list[str]:
    """Sample up to n_samples clip filenames from a target LRQ dir's index.csv."""
    xdf = pd.read_csv(root / "index.csv")
    xdf = xdf[xdf["label"].astype(str) != "unknown"]
    if "n_frames_30hz" in xdf.columns:
        xdf = xdf[xdf["n_frames_30hz"] >= MIN_FRAMES]
    files = xdf["file"].tolist()
    n = min(n_samples, len(files))
    return [files[i] for i in rng.choice(len(files), n, replace=False)]


def run_multi_target(args, ntu_root: Path, out_dir: Path) -> None:
    """T3: layer-wise CKA between NTU and EACH target (gap-axis measurement, inference-only)."""
    rng = np.random.default_rng(0)
    ntu_all = sorted(f.name for f in ntu_root.glob("*.npy"))

    target_files: dict[str, list[str]] = {}
    target_roots: dict[str, Path] = {}
    for tag, rel_root in TARGETS:
        troot = PROJECT_ROOT / rel_root
        target_roots[tag] = troot
        target_files[tag] = load_target_files(troot, args.n_samples, rng)
        print(f"target={tag}: n={len(target_files[tag])} (pool root={rel_root})")

    max_n = max(len(f) for f in target_files.values())
    max_n = min(max_n, len(ntu_all))
    ntu_sample = [ntu_all[i] for i in rng.choice(len(ntu_all), max_n, replace=False)]

    rows = []
    for cfg in ENCODER_CONFIGS:
        enc = load_encoder(cfg["ckpt"])
        ntu_feats_full = extract_layer_feats(enc, ntu_sample, ntu_root, args.batch_size)
        for tag, _ in TARGETS:
            n_t = len(target_files[tag])
            ntu_feats_t = {layer: ntu_feats_full[layer][:n_t] for layer in LAYER_NAMES}
            tgt_feats = extract_layer_feats(enc, target_files[tag], target_roots[tag], args.batch_size)
            for layer in LAYER_NAMES:
                lin = linear_cka(ntu_feats_t[layer], tgt_feats[layer])
                rbf = rbf_cka(ntu_feats_t[layer], tgt_feats[layer])
                rows.append({"target": tag, "encoder": cfg["tag"], "layer": layer,
                             "cka_linear": round(lin, 4), "cka_rbf": round(rbf, 4)})
                print(f"  {tag:>14} {cfg['tag']:>8} {layer}: linear={lin:.4f} rbf={rbf:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "cka_by_target.csv", index=False)

    # Summary: mean linear CKA over L0-L2 (depth layers, excludes proj) per target x encoder.
    depth = res[res["layer"] != "proj"]
    summary = depth.groupby(["target", "encoder"])["cka_linear"].mean().unstack("encoder")
    summary = summary.reindex(index=[t for t, _ in TARGETS],
                               columns=[c["tag"] for c in ENCODER_CONFIGS])
    summary.round(4).to_csv(out_dir / "cka_by_target_summary.csv")
    print("\nMean linear CKA (L0-L2) by target x encoder:")
    print(summary.round(4).to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(summary.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(summary.columns))); ax.set_xticklabels(summary.columns)
    ax.set_yticks(range(len(summary.index))); ax.set_yticklabels(summary.index)
    for i in range(summary.shape[0]):
        for j in range(summary.shape[1]):
            ax.text(j, i, f"{summary.values[i, j]:.3f}", ha="center", va="center", color="w")
    ax.set_title("Mean linear CKA (L0-L2): NTU vs each target")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out_dir / "cka_by_target_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nDONE cka_by_target -> {out_dir}")


def compute_benefits() -> dict[str, float]:
    """k=1 cross-domain accuracy benefit (method - scratch), pooled over seeds+subjects."""
    frames = []
    for sd in SEED_DIRS:
        p = PROJECT_ROOT / sd / "summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df[df["k"] == 1]
    mean_acc = df.groupby("method")["final_acc"].mean()
    if "scratch" not in mean_acc:
        return {}
    base = mean_acc["scratch"]
    return {m: float(mean_acc[m] - base) for m in mean_acc.index}


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer-wise NTU-vs-Xsens-v2 CKA per encoder.")
    ap.add_argument("--ntu-root", default="Data_Processed/ntu_quats")
    ap.add_argument("--xsens-root", default="Data_Processed/imu_quats_v2")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-dir", default="trained_models/Phase1-analysis")
    ap.add_argument("--smoke", action="store_true", help="scratch encoder only, 128 clips.")
    ap.add_argument("--multi-target", action="store_true",
                     help="T3: CKA between NTU and each of TARGETS (Xsens-v2, CZU skeleton, "
                          "CZU IMU quat, UTD skeleton) instead of the single --xsens-root target.")
    args = ap.parse_args()

    ntu_root = PROJECT_ROOT / args.ntu_root
    xsens_root = PROJECT_ROOT / args.xsens_root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.multi_target:
        run_multi_target(args, ntu_root, out_dir)
        return

    n = 128 if args.smoke else args.n_samples
    configs = ENCODER_CONFIGS[:1] if args.smoke else ENCODER_CONFIGS

    rng = np.random.default_rng(0)
    ntu_all = sorted(f.name for f in ntu_root.glob("*.npy"))
    xdf = pd.read_csv(xsens_root / "index.csv")
    xdf = xdf[xdf["label"].astype(str) != "unknown"]
    if "n_frames_30hz" in xdf.columns:
        xdf = xdf[xdf["n_frames_30hz"] >= MIN_FRAMES]
    xsens_all = xdf["file"].tolist()
    n = min(n, len(ntu_all), len(xsens_all))  # CKA needs equal sample counts
    ntu_files = [ntu_all[i] for i in rng.choice(len(ntu_all), n, replace=False)]
    xsens_files = [xsens_all[i] for i in rng.choice(len(xsens_all), n, replace=False)]
    print(f"CKA on N={n} clips/domain (NTU pool {len(ntu_all)}, Xsens-v2 pool {len(xsens_all)}).")

    rows = []
    for cfg in configs:
        enc = load_encoder(cfg["ckpt"])
        ntu_feats = extract_layer_feats(enc, ntu_files, ntu_root, args.batch_size)
        xsens_feats = extract_layer_feats(enc, xsens_files, xsens_root, args.batch_size)
        for layer in LAYER_NAMES:
            lin = linear_cka(ntu_feats[layer], xsens_feats[layer])
            rbf = rbf_cka(ntu_feats[layer], xsens_feats[layer])
            rows.append({"encoder": cfg["tag"], "layer": layer,
                         "cka_linear": round(lin, 4), "cka_rbf": round(rbf, 4)})
            print(f"  {cfg['tag']:>8} {layer}: linear={lin:.4f} rbf={rbf:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "cka_results.csv", index=False)

    if not args.smoke:
        # heatmap: encoders x layers (linear CKA)
        piv = res.pivot(index="encoder", columns="layer", values="cka_linear").reindex(
            index=[c["tag"] for c in ENCODER_CONFIGS], columns=LAYER_NAMES)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(LAYER_NAMES))); ax.set_xticklabels(LAYER_NAMES)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w")
        ax.set_title("Linear CKA: NTU vs Xsens-v2")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(out_dir / "cka_heatmap.png", dpi=120)
        plt.close(fig)

        # cka vs cross-domain benefit
        benefits = compute_benefits()
        if benefits:
            mean_cka = res.groupby("encoder")["cka_linear"].mean().to_dict()
            fig, ax = plt.subplots(figsize=(5, 4))
            for tag in [c["tag"] for c in ENCODER_CONFIGS]:
                if tag in benefits and tag in mean_cka:
                    ax.scatter(mean_cka[tag], benefits[tag])
                    ax.annotate(tag, (mean_cka[tag], benefits[tag]))
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xlabel("mean linear CKA (NTU vs Xsens-v2)")
            ax.set_ylabel("k=1 acc benefit vs scratch (pp)")
            ax.set_title("Representation alignment vs transfer benefit")
            fig.tight_layout()
            fig.savefig(out_dir / "cka_vs_benefit.png", dpi=120)
            plt.close(fig)

    print(f"DONE cka_analysis -> {out_dir}")


if __name__ == "__main__":
    main()
