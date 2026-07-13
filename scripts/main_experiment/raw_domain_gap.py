#!/usr/bin/env python3
"""Raw, method-independent domain-gap quantification between dataset PAIRS.

Unlike mmd_domain_gap.py / cka_analysis.py, this does NOT route data
through any pretrained (or even randomly-init) encoder -- it measures how
different the raw datasets are themselves, using the same hand-crafted
statistical-moment feature (mean/std/var/skew/kurtosis per of 68 flattened LRQ
channels -> 340-d) already used for the CRC published-baseline reproductions
(crc_baseline.py (CZU)::clip_features). All 5 datasets share the (T,17,4)
LRQ schema so one feature extractor works for all of them.

For every pair among {ntu, xsens_v2, czu_skeleton, czu_imu_quat, utd_skeleton}
(10 pairs), computes on pooled-standardized features:
  - MMD^2        RBF kernel, median-heuristic bandwidth (scale-fair, fixes the
                 fixed-sigma flaw in the old encoder-space MMD).
  - Frechet dist Gaussian closed form on a PCA(50) projection (numerically
                 stable covariance at n~1000).
  - proxy A-dist Ben-David et al. 2007: 5-fold CV logistic regression domain
                 classifier -> error eps -> d_A = 2*(1-2*eps) in [0,2].

Outputs -> trained_models/RawDomainGap/
  raw_domain_gap.csv        pair x {mmd2, frechet, a_distance, clf_acc, n_a, n_b}
  raw_domain_gap_mmd.png    5x5 heatmap
  raw_domain_gap_frechet.png
  raw_domain_gap_adist.png
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from scripts.external.czu.crc_baseline import clip_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MIN_FRAMES = 8
N_SAMPLES = 1000
PCA_DIM = 50
SEED = 0

DATASETS = [
    ("ntu", "Data_Processed/ntu_quats", None),
    ("xsens_v2", "Data_Processed/imu_quats_v2", "unknown"),
    ("czu_skeleton", "Data_Processed/czu_skeleton_lrq", "unknown"),
    ("czu_imu_quat", "Data_Processed/czu_imu_quats", "unknown"),
    ("utd_skeleton", "Data_Processed/utd_skeleton_lrq", "unknown"),
]


def list_files(root: Path, drop_label: str | None, n_samples: int, rng: np.random.Generator) -> list[str]:
    index = root / "index.csv"
    if index.exists():
        df = pd.read_csv(index)
        if drop_label is not None:
            df = df[df["label"].astype(str) != drop_label]
        if "n_frames_30hz" in df.columns:
            df = df[df["n_frames_30hz"] >= MIN_FRAMES]
        files = df["file"].tolist()
    else:
        files = sorted(f.name for f in root.glob("*.npy"))
    n = min(n_samples, len(files))
    idx = rng.choice(len(files), n, replace=False)
    return [files[i] for i in idx]


def extract_features(root: Path, files: list[str]) -> np.ndarray:
    feats = [clip_features(root / f) for f in files]
    return np.stack(feats, axis=0)  # (N, 340)


def median_heuristic_sigma2(Z: np.ndarray) -> float:
    n = min(len(Z), 500)
    idx = np.random.default_rng(SEED).choice(len(Z), n, replace=False)
    sub = Z[idx]
    sq = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(sq[sq > 0])
    return max(med, 1e-8)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.concatenate([X, Y], axis=0)
    sigma2 = median_heuristic_sigma2(Z)

    def kernel(A, B):
        sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-sq / (2 * sigma2))

    Kxx = kernel(X, X); Kyy = kernel(Y, Y); Kxy = kernel(X, Y)
    m, n = len(X), len(Y)
    t1 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    t2 = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    t3 = Kxy.mean()
    return float(t1 + t2 - 2 * t3)


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    pca = PCA(n_components=min(PCA_DIM, X.shape[0] - 1, Y.shape[0] - 1, X.shape[1]), random_state=SEED)
    Z = pca.fit_transform(np.concatenate([X, Y], axis=0))
    Xp, Yp = Z[: len(X)], Z[len(X):]
    mu1, mu2 = Xp.mean(0), Yp.mean(0)
    C1 = np.cov(Xp, rowvar=False)
    C2 = np.cov(Yp, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(C1 @ C2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(C1 + C2 - 2 * covmean))


def proxy_a_distance(X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
    Z = np.concatenate([X, Y], axis=0)
    labels = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clf, Z, labels, cv=cv)
    acc = float((pred == labels).mean())
    eps = 1.0 - acc
    d_a = 2.0 * (1.0 - 2.0 * eps)
    return d_a, acc


def pooled_standardize(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Z = np.concatenate([X, Y], axis=0)
    mu, sd = Z.mean(0), Z.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (X - mu) / sd, (Y - mu) / sd


def heatmap(df: pd.DataFrame, col: str, tags: list[str], out_path: Path, title: str):
    n = len(tags)
    M = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = tags.index(r["a"]), tags.index(r["b"])
        M[i, j] = M[j, i] = r[col]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(tags)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--out-dir", default="trained_models/RawDomainGap")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_samples = 64 if args.smoke else args.n_samples

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    feats: dict[str, np.ndarray] = {}
    for tag, rel_root, drop_label in DATASETS:
        root = PROJECT_ROOT / rel_root
        files = list_files(root, drop_label, n_samples, rng)
        feats[tag] = extract_features(root, files)
        print(f"{tag:>14}: n={len(files)} (pool root={rel_root})")

    tags = [t for t, _, _ in DATASETS]
    rows = []
    for a, b in itertools.combinations(tags, 2):
        Xs, Ys = pooled_standardize(feats[a], feats[b])
        mmd2 = mmd2_rbf(Xs, Ys)
        fd = frechet_distance(Xs, Ys)
        d_a, acc = proxy_a_distance(Xs, Ys)
        rows.append({
            "a": a, "b": b, "n_a": len(feats[a]), "n_b": len(feats[b]),
            "mmd2": round(mmd2, 5), "frechet": round(fd, 3),
            "a_distance": round(d_a, 4), "clf_acc": round(acc, 4),
        })
        print(f"  {a:>14} vs {b:<14} mmd2={mmd2:.5f} frechet={fd:8.3f} "
              f"a_dist={d_a:.4f} (clf_acc={acc:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_domain_gap.csv", index=False)

    heatmap(df, "mmd2", tags, out_dir / "raw_domain_gap_mmd.png", "Raw-feature MMD^2 (RBF, median heuristic)")
    heatmap(df, "frechet", tags, out_dir / "raw_domain_gap_frechet.png", "Raw-feature Frechet distance (PCA-50)")
    heatmap(df, "a_distance", tags, out_dir / "raw_domain_gap_adist.png", "Proxy A-distance (logreg domain classifier)")

    print(f"\nWrote {out_dir / 'raw_domain_gap.csv'}")


if __name__ == "__main__":
    main()
