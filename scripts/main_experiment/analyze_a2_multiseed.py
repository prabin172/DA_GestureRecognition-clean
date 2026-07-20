#!/usr/bin/env python3
"""A2 subject-count scaling: 3-seed pooled analysis + paired-t stats.

Closes a real reproducibility gap: the numbers in wiki/results/a2-subject-scaling.md
and paper_results.md R4c were produced by an "ad hoc pooling script" that was never
committed (see that page's caveats section). This script reproduces them from the
raw per-seed run dirs and is the first committed source for the pooled A2 numbers.

Reads trained_models/A2-subjectScaling{,-seed43,-seed44}/N{0..4}/summary.csv (3 seeds
x 5 N-values x 5 held-out subjects = pools to n=15 per (N,method,k) cell), matching
temp_czu_multiseed_analyze.py's paired_cells/paired_t pattern (cells keyed on
(test_subject, k, seed) so method vs scratch comparisons are properly paired, not
just independent means).

Outputs -> trained_models/A2-subjectScaling-pooled/
  a2_pooled_results.csv   (N, method, k, mean_acc, n, benefit_vs_scratch) -- must
                           match the existing file byte-for-byte on mean_acc/benefit
                           columns; this script's job is to make that file reproducible
  a2_pooled_stats.csv     (N, method, k, paired_t_p, mean_diff, n_pairs) vs scratch
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS = {42: "", 43: "-seed43", 44: "-seed44"}
N_VALUES = [0, 1, 2, 3, 4]
KS = [0, 1, 3]
METHOD_ORDER = ["scratch", "supLP120", "supMAE", "mae"]


def load_pooled(root: Path) -> pd.DataFrame:
    rows = []
    for seed, suf in SEEDS.items():
        for N in N_VALUES:
            p = root.parent / f"{root.name}{suf}" / f"N{N}" / "summary.csv"
            if not p.exists():
                print(f"WARN missing {p}")
                continue
            d = pd.read_csv(p)[["test_subject", "method", "k", "final_acc"]].copy()
            d["seed"] = seed
            d["N"] = N
            rows.append(d)
    if not rows:
        raise SystemExit("No A2 summary.csv found in any seed dir.")
    return pd.concat(rows, ignore_index=True)


def paired_cells(df: pd.DataFrame, a: str, b: str, N: int, k: int):
    """Aligned (method a, method b) final_acc arrays over (subject,seed) cells at fixed (N,k)."""
    sub = df[(df.N == N) & (df.k == k)]
    keys = ["test_subject", "seed"]
    da = sub[sub.method == a].set_index(keys)["final_acc"]
    db = sub[sub.method == b].set_index(keys)["final_acc"]
    idx = da.index.intersection(db.index)
    return da.loc[idx].values, db.loc[idx].values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="trained_models/A2-subjectScaling")
    ap.add_argument("--out-dir", default="trained_models/A2-subjectScaling-pooled")
    args = ap.parse_args()

    root = PROJECT_ROOT / args.root
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pooled(root)

    # mean_acc + benefit_vs_scratch, matching a2_pooled_results.csv schema exactly
    agg = (df.groupby(["N", "method", "k"])
             .agg(mean_acc=("final_acc", "mean"), n=("final_acc", "size"))
             .reset_index())
    scratch_mean = agg[agg.method == "scratch"].set_index(["N", "k"])["mean_acc"]
    agg["benefit_vs_scratch"] = agg.apply(
        lambda r: r["mean_acc"] - scratch_mean.get((r["N"], r["k"]), np.nan), axis=1)
    agg["mean_acc"] = agg["mean_acc"].round(2)
    agg["benefit_vs_scratch"] = agg["benefit_vs_scratch"].round(2)
    agg = agg[["N", "method", "k", "mean_acc", "n", "benefit_vs_scratch"]]
    agg.to_csv(out_dir / "a2_pooled_results.csv", index=False)
    print("=== pooled mean_acc / benefit_vs_scratch ===")
    print(agg.to_string(index=False))

    # paired-t vs scratch, per (N, method, k), cells = (subject,seed), n<=15
    stat_rows = []
    for N in N_VALUES:
        for method in METHOD_ORDER:
            if method == "scratch":
                continue
            for k in KS:
                xa, xb = paired_cells(df, method, "scratch", N, k)
                if len(xa) < 2:
                    continue
                diff = xa - xb
                t, p = stats.ttest_rel(xa, xb)
                stat_rows.append(dict(N=N, method=method, k=k,
                                       mean_diff=round(float(diff.mean()), 2),
                                       paired_t_p=float(p), n_pairs=len(xa)))
    stats_df = pd.DataFrame(stat_rows)
    stats_df.to_csv(out_dir / "a2_pooled_stats.csv", index=False)
    print("\n=== paired-t vs scratch (n_pairs = seeds x subjects, up to 15) ===")
    print(stats_df.to_string(index=False))
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
