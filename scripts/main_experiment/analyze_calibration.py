#!/usr/bin/env python3
"""Phase 1 / analysis: ECE + reliability diagrams + clip-level McNemar from dumped posteriors.

Reads {seed_dir}/posteriors/*.csv (produced by dump_posteriors.py).
Outputs under --out-dir (default trained_models/Phase1-analysis):
  - ece_results.csv        : ECE per (method,k), before/after post-hoc temperature scaling
  - mcnemar_results.csv     : paired clip-level McNemar, scratch vs each prior, per k
  - reliability_k{k}.png    : reliability diagrams, methods side by side, per k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SEED_DIRS = [
    "trained_models/LOSO-fullTrainCalibrate-v2",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed44",
]
METHOD_ORDER = ["scratch", "supLP120", "supMAE", "mae", "supcon"]


def load_posteriors(seed_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for sd in seed_dirs:
        pdir = sd / "posteriors"
        if not pdir.exists():
            continue
        for f in sorted(pdir.glob("*.csv")):
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("No posteriors CSVs found. Run dump_posteriors.py first.")
    df = pd.concat(frames, ignore_index=True)
    df["logits"] = df["logits"].apply(json.loads)
    return df


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, list]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(confidences)
    diag = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if m.sum() == 0:
            diag.append((0.5 * (lo + hi), np.nan, 0))
            continue
        acc = correct[m].mean()
        conf = confidences[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
        diag.append((conf, acc, int(m.sum())))
    return float(e), diag


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit scalar T>0 minimizing NLL (post-hoc, in-sample; illustrative)."""
    def nll(logT):
        T = np.exp(logT)
        z = logits / T
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(labels)), labels].mean()
    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    return float(np.exp(res.x))


def softmax_conf(logits: np.ndarray, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return p.max(1), p.argmax(1)


def run_ece(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    ks = sorted(df["k"].unique())
    for k in ks:
        fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(4 * len(METHOD_ORDER), 4), squeeze=False)
        for j, method in enumerate(METHOD_ORDER):
            sub = df[(df["method"] == method) & (df["k"] == k)]
            if len(sub) == 0:
                continue
            logits = np.array(sub["logits"].tolist(), dtype=np.float64)
            labels = sub["true_id"].to_numpy()
            conf0, pred0 = softmax_conf(logits, 1.0)
            correct0 = (pred0 == labels).astype(float)
            ece0, diag0 = ece(conf0, correct0)
            T = temperature_scale(logits, labels)
            conf1, pred1 = softmax_conf(logits, T)
            correct1 = (pred1 == labels).astype(float)
            ece1, _ = ece(conf1, correct1)
            rows.append({"method": method, "k": int(k), "n_clips": len(sub),
                         "acc": round(100.0 * correct0.mean(), 3),
                         "ece": round(ece0, 4), "temperature": round(T, 3),
                         "ece_tempscaled": round(ece1, 4)})
            ax = axes[0][j]
            centers = [d[0] for d in diag0]
            accs = [d[1] for d in diag0]
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.bar(centers, accs, width=1.0 / 15, alpha=0.7, edgecolor="k")
            ax.set_title(f"{method} k{k}\nECE={ece0:.3f} T={T:.2f}")
            ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / f"reliability_k{k}.png", dpi=120)
        plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "ece_results.csv", index=False)
    return res


def run_mcnemar(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Paired McNemar: scratch vs each prior. Pairing key = (seed,subject,k,file)."""
    df = df.copy()
    df["key"] = df["seed"].astype(str) + "|" + df["subject"] + "|" + df["k"].astype(str) + "|" + df["file"]
    rows = []
    ks = sorted(df["k"].unique())
    for k in ks:
        base = df[(df["method"] == "scratch") & (df["k"] == k)].set_index("key")["correct"]
        for prior in [m for m in METHOD_ORDER if m != "scratch"]:
            pr = df[(df["method"] == prior) & (df["k"] == k)].set_index("key")["correct"]
            common = base.index.intersection(pr.index)
            b = int(((base.loc[common] == 1) & (pr.loc[common] == 0)).sum())  # scratch right, prior wrong
            c = int(((base.loc[common] == 0) & (pr.loc[common] == 1)).sum())  # scratch wrong, prior right
            n = b + c
            p = binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
            rows.append({"k": int(k), "prior": prior, "n_pairs": len(common),
                         "scratch_only_correct_b": b, "prior_only_correct_c": c,
                         "net_prior_gain": c - b, "mcnemar_p": round(p, 5)})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "mcnemar_results.csv", index=False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="ECE + reliability + McNemar from dumped posteriors.")
    ap.add_argument("--seed-dirs", default=",".join(DEFAULT_SEED_DIRS))
    ap.add_argument("--out-dir", default="trained_models/Phase1-analysis")
    args = ap.parse_args()

    seed_dirs = [PROJECT_ROOT / d for d in args.seed_dirs.split(",")]
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_posteriors(seed_dirs)
    print(f"Loaded {len(df)} clip-posteriors across {df['seed'].nunique()} seeds.")
    ece_res = run_ece(df, out_dir)
    print("\n== ECE ==\n", ece_res.to_string(index=False))
    mc_res = run_mcnemar(df, out_dir)
    print("\n== McNemar (scratch vs prior) ==\n", mc_res.to_string(index=False))
    print(f"\nDONE analyze_calibration -> {out_dir}")


if __name__ == "__main__":
    main()
