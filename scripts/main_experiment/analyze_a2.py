#!/usr/bin/env python3
"""Phase 2 / A2 analysis: prior benefit vs number of fine-tuning subjects (N).

Reads trained_models/A2-subjectScaling/N{0,1,2,3,4}/summary.csv and produces, per k:
  - mean accuracy per method vs N
  - prior benefit (method - scratch) vs N
Prediction: prior benefit is largest at N=0/1 and washes out as N grows.

Outputs under --out-dir (default trained_models/A2-subjectScaling/analysis):
  - a2_results.csv        : (N, method, k, mean_acc, n_folds) + benefit vs scratch
  - a2_benefit_vs_N.png   : benefit curves, one panel per k
  - a2_acc_vs_N.png       : raw accuracy curves, one panel per k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METHOD_ORDER = ["scratch", "supLP120", "supMAE", "mae"]


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 subject-count scaling analysis.")
    ap.add_argument("--root", default="trained_models/A2-subjectScaling")
    ap.add_argument("--n-values", default="0,1,2,3,4")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = PROJECT_ROOT / args.root
    out_dir = Path(args.out_dir) if args.out_dir else root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_values = [int(x) for x in args.n_values.split(",")]

    frames = []
    for N in n_values:
        p = root / f"N{N}" / "summary.csv"
        if not p.exists():
            print(f"WARN missing {p}")
            continue
        d = pd.read_csv(p)
        d["N"] = N
        frames.append(d)
    if not frames:
        raise SystemExit("No A2 summary.csv found.")
    df = pd.concat(frames, ignore_index=True)
    # collapse calibration_mode: use the head_only / primary row per (N,subject,method,k)
    # final_acc is per-fold; average across the 5 held-out subjects.
    agg = (df.groupby(["N", "method", "k"])
             .agg(mean_acc=("final_acc", "mean"),
                  std_acc=("final_acc", "std"),
                  n_folds=("final_acc", "size"))
             .reset_index())

    # benefit vs scratch at matching (N,k)
    scratch = agg[agg["method"] == "scratch"].set_index(["N", "k"])["mean_acc"]
    agg["benefit_vs_scratch"] = agg.apply(
        lambda r: r["mean_acc"] - scratch.get((r["N"], r["k"]), np.nan), axis=1)
    agg = agg.round(3)
    agg.to_csv(out_dir / "a2_results.csv", index=False)
    print(agg.to_string(index=False))

    ks = sorted(agg["k"].unique())
    # benefit curves
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
    for ci, k in enumerate(ks):
        ax = axes[0][ci]
        for method in METHOD_ORDER:
            sub = agg[(agg["method"] == method) & (agg["k"] == k)].sort_values("N")
            if len(sub) == 0 or method == "scratch":
                continue
            ax.plot(sub["N"], sub["benefit_vs_scratch"], marker="o", label=method)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_title(f"k={k}: prior benefit vs N")
        ax.set_xlabel("# fine-tuning subjects (N)")
        ax.set_ylabel("acc benefit vs scratch (pp)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "a2_benefit_vs_N.png", dpi=120)
    plt.close(fig)

    # raw accuracy curves
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
    for ci, k in enumerate(ks):
        ax = axes[0][ci]
        for method in METHOD_ORDER:
            sub = agg[(agg["method"] == method) & (agg["k"] == k)].sort_values("N")
            if len(sub) == 0:
                continue
            ax.plot(sub["N"], sub["mean_acc"], marker="o", label=method)
        ax.set_title(f"k={k}: accuracy vs N")
        ax.set_xlabel("# fine-tuning subjects (N)")
        ax.set_ylabel("mean held-out accuracy (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "a2_acc_vs_N.png", dpi=120)
    plt.close(fig)
    print(f"\nDONE analyze_a2 -> {out_dir}")


if __name__ == "__main__":
    main()
