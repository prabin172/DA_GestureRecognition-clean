#!/usr/bin/env python3
"""Phase 3 / C6 -- CROSS-SETTING extension (2026-07-21): reproduce Locks 1+2 ONLY (no Lock 3 /
iso-safety) of the randomized robustness protocol across every transfer setting for which
compatible per-clip posteriors + label maps exist, not just NTU->Xsens.

This is `controller_robust.py`'s Lock 1 + Lock 2 logic (simulate(), make_prim_of_id(), pack(),
load(), the shared 120-random-vocab draw), refactored to be generic over a SETTINGS dict instead
of one hardcoded NTU->Xsens RUNDIRS/label_map. controller_robust.py itself is untouched -- its
locked `trained_models/Phase3-controller/robust/` output (cited in paper_results.md R5) is not
read or written by this script; NTU->Xsens is re-simulated here too (cheap, ~1 min) so every
setting in the cross-setting table comes from the identical code path/output schema.

Settings and what each one's posteriors came from:
  ntu_xsens     -- trained_models/LOSO-fullTrainCalibrate-v2[-seed43/44]/posteriors/ (pre-existing)
  czu_skeleton  -- trained_models/CZU-skeleton-LOSO[-seed43/44]/posteriors/ (dumped 2026-07-21 via
                   dump_posteriors.py, pure forward pass over existing checkpoints, no retrain)
  czu_imu_quat  -- trained_models/CZU-IMU-LOSO[-seed43/44]/posteriors/ (same, quat-only crossmodal)
  utd_skeleton  -- trained_models/UTD-skeleton-LOSO-seed{42,43,44}/posteriors/ (same)
  czu_dual_raw  -- trained_models/CZU-IMU-DUAL-controller[-seed43/44]/posteriors/ (dualbranch.py
                   --mode dual never persisted checkpoints/posteriors before 2026-07-21; this
                   setting required an actual retrain, done via
                   scripts/orchestration/10_czu_dual_controller_retrain.sh, --dump-posteriors-dir
                   added to dualbranch.py for exactly this purpose. Existing locked
                   CZU-IMU-DUAL*/summary.csv accuracy numbers, cited in paper_results.md R6c, are
                   untouched -- new out-roots only.)

Each setting's own n_classes/label space is used to draw its own 120 random 7-"System Input"
assignments (same rng recipe/seed as controller_robust.py, applied per setting).

Outputs -> trained_models/Phase3-controller/crosssetting/<setting>/
  vocab_sweep.csv, vocab_ordering.csv    [Lock 1]
  costmodel_sweep.csv, costmodel_summary.csv   [Lock 2]
Plus trained_models/Phase3-controller/crosssetting/cross_setting_summary.csv (the table this
script was written to produce): per setting, best/worst method under Lock 1 and Lock 2, whether
they agree.
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "trained_models/Phase3-controller/crosssetting")
METHODS = ["scratch", "mae", "supMAE", "supLP120", "supcon"]

PRIMS = ["next", "previous", "approach", "grasp", "release", "confirm", "cancel"]
CRIT_PRIMS = {PRIMS.index("grasp"), PRIMS.index("release")}
MISSION = [0, 0, 1, 2, 3, 5, 0, 2, 3, 4, 5, 6]

SETTINGS = {
    "ntu_xsens": dict(
        label="NTU -> Xsens (main)",
        rundirs=["trained_models/LOSO-fullTrainCalibrate-v2",
                 "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
                 "trained_models/LOSO-fullTrainCalibrate-v2-seed44"],
        label_map="trained_models/LOSO-fullTrainCalibrate-v2/label_map.json",
    ),
    "czu_skeleton": dict(
        label="CZU-MHAD skeleton (R6)",
        rundirs=["trained_models/CZU-skeleton-LOSO",
                 "trained_models/CZU-skeleton-LOSO-seed43",
                 "trained_models/CZU-skeleton-LOSO-seed44"],
        label_map="trained_models/CZU-skeleton-LOSO/label_map.json",
    ),
    "czu_imu_quat": dict(
        label="CZU-MHAD IMU orientation-only (R6b)",
        rundirs=["trained_models/CZU-IMU-LOSO",
                 "trained_models/CZU-IMU-LOSO-seed43",
                 "trained_models/CZU-IMU-LOSO-seed44"],
        label_map="trained_models/CZU-IMU-LOSO/label_map.json",
    ),
    "utd_skeleton": dict(
        label="UTD-MHAD skeleton (R6d)",
        rundirs=["trained_models/UTD-skeleton-LOSO-seed42",
                 "trained_models/UTD-skeleton-LOSO-seed43",
                 "trained_models/UTD-skeleton-LOSO-seed44"],
        label_map="trained_models/UTD-skeleton-LOSO-seed42/label_map.json",
    ),
    "czu_dual_raw": dict(
        label="CZU-MHAD dual-branch raw+quat (R6c)",
        rundirs=["trained_models/CZU-IMU-DUAL-controller",
                 "trained_models/CZU-IMU-DUAL-controller-seed43",
                 "trained_models/CZU-IMU-DUAL-controller-seed44"],
        label_map="trained_models/CZU-IMU-LOSO/label_map.json",  # same CZU label space/index
    ),
}


def load_posteriors(rundirs):
    frames = []
    for d in rundirs:
        files = glob.glob(os.path.join(ROOT, d, "posteriors", "*.csv"))
        if not files:
            raise FileNotFoundError(f"No posteriors/*.csv under {d} -- dump them first.")
        frames += [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def pack(df, n_classes):
    packed = {}
    for method in METHODS:
        packed[method] = {}
        for k in (1, 3):
            sl = df[(df.method == method) & (df.k == k)]
            if len(sl) == 0:
                raise ValueError(f"No rows for method={method} k={k} -- incomplete posteriors.")
            g = {}
            for gid in range(n_classes):
                rows = sl[sl.true_id == gid]
                g[gid] = (rows.conf.to_numpy(np.float32), rows.pred_id.to_numpy(np.int64))
            packed[method][k] = dict(by_g=g)
    return packed


def make_prim_of_id(vocab_ids, n_classes):
    arr = -np.ones(n_classes, dtype=np.int64)
    for prim_idx, gid in enumerate(vocab_ids):
        arr[gid] = prim_idx
    return arr


def simulate(slice_data, prim_of_id, tau, C_crit, costs, rng, n_missions):
    """Identical logic to controller_robust.py's simulate() -- single pass yields both the
    hard-safety outcome (Lock 1) and the soft-cost outcome (Lock 2)."""
    T_exec, T_reject, T_correct, MAX_REJECT = costs
    by_g = slice_data["by_g"]
    gest_of_prim = {int(prim_of_id[g]): g for g in range(len(prim_of_id)) if prim_of_id[g] >= 0}
    hard_ok_n = 0
    tot_cost = 0.0
    for _ in range(n_missions):
        cost = 0.0; had_crit = False; aborted = False
        for intended in MISSION:
            g = gest_of_prim[intended]
            conf_a, pred_a = by_g[g]
            nrej = 0
            while True:
                i = rng.integers(len(conf_a))
                conf = conf_a[i]; pp = prim_of_id[pred_a[i]]
                if conf < tau:
                    cost += T_reject; nrej += 1
                    if nrej > MAX_REJECT:
                        aborted = True; break
                    continue
                cost += T_exec
                if pp == intended:
                    break
                if (intended in CRIT_PRIMS) or (pp in CRIT_PRIMS):
                    cost += C_crit; had_crit = True; break
                cost += T_correct; break
            if aborted:
                break
        tot_cost += cost
        if not had_crit and not aborted:
            hard_ok_n += 1
    return dict(hard_success=hard_ok_n / n_missions, mean_cost=tot_cost / n_missions)


def run_setting(name, cfg, vocabs, missions, seed, missions_seed_base):
    print(f"\n=== {name} ({cfg['label']}) ===")
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(out, exist_ok=True)

    labelmap = json.load(open(os.path.join(ROOT, cfg["label_map"])))
    n_classes = len(labelmap)
    if n_classes < 7:
        raise ValueError(f"{name}: only {n_classes} classes, need >=7 for a System Input draw.")

    df = load_posteriors(cfg["rundirs"])
    packed = pack(df, n_classes)
    costs = (1.0, 1.0, 3.0, 5)

    rng = np.random.default_rng(seed)
    all_ids = np.arange(n_classes)
    vocab_ids_list = [list(rng.choice(all_ids, size=7, replace=False)) for _ in range(vocabs)]
    print(f"  {n_classes} classes, drew {vocabs} random 7-gesture assignments")

    # ---------- LOCK 1 ----------
    vocab_rows = []
    for v, vids in enumerate(vocab_ids_list):
        prim_of_id = make_prim_of_id(vids, n_classes)
        for k in (1, 3):
            for tau in (0.0, 0.9):
                for method in METHODS:
                    r = simulate(packed[method][k], prim_of_id, tau, C_crit=20.0, costs=costs,
                                rng=np.random.default_rng(missions_seed_base + 1000 + v),
                                n_missions=missions)
                    vocab_rows.append(dict(vocab=v, method=method, k=k, tau=tau, **r))
    vdf = pd.DataFrame(vocab_rows)
    vdf.to_csv(os.path.join(out, "vocab_sweep.csv"), index=False)

    ord_rows = []
    for k in (1, 3):
        for tau in (0.0, 0.9):
            piv = vdf[(vdf.k == k) & (vdf.tau == tau)].pivot_table(
                index="vocab", columns="method", values="hard_success")
            for a in METHODS:
                for b in METHODS:
                    if a == b:
                        continue
                    d = piv[a] - piv[b]
                    ord_rows.append(dict(k=k, tau=tau, a=a, b=b,
                                         median_delta=round(float(d.median()), 4),
                                         frac_a_ge_b=round(float((d >= 0).mean()), 4)))
    odf = pd.DataFrame(ord_rows)
    odf.to_csv(os.path.join(out, "vocab_ordering.csv"), index=False)

    # ---------- LOCK 2 ----------
    cc_rows = []
    for v, vids in enumerate(vocab_ids_list):
        prim_of_id = make_prim_of_id(vids, n_classes)
        for C_crit in [2.0, 5.0, 10.0, 20.0, 50.0, 1e6]:
            for k in (1, 3):
                for method in METHODS:
                    r = simulate(packed[method][k], prim_of_id, tau=0.0, C_crit=C_crit, costs=costs,
                                rng=np.random.default_rng(missions_seed_base + 2000 + v),
                                n_missions=missions)
                    cc_rows.append(dict(vocab=v, method=method, k=k, C_crit=C_crit,
                                        mean_cost=r["mean_cost"], hard_success=r["hard_success"]))
    ccdf = pd.DataFrame(cc_rows)
    ccdf.to_csv(os.path.join(out, "costmodel_sweep.csv"), index=False)
    cc_summary = ccdf.groupby(["method", "k", "C_crit"])["mean_cost"].agg(
        mean="mean", median="median",
        q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
    ).reset_index()
    cc_summary.to_csv(os.path.join(out, "costmodel_summary.csv"), index=False)

    # ---------- per-setting worst/best summary ----------
    lock1_k1t0 = vdf[(vdf.k == 1) & (vdf.tau == 0.0)].groupby("method")["hard_success"].mean()
    lock1_worst, lock1_best = lock1_k1t0.idxmin(), lock1_k1t0.idxmax()
    lock2_c50 = cc_summary[(cc_summary.k == 1) & (cc_summary.C_crit == 50.0)].set_index("method")["median"]
    lock2_worst, lock2_best = lock2_c50.idxmax(), lock2_c50.idxmin()  # highest cost = worst
    print(f"  Lock1 (k=1,tau=0 mean hard_success): worst={lock1_worst} ({lock1_k1t0[lock1_worst]:.3f}) "
          f"best={lock1_best} ({lock1_k1t0[lock1_best]:.3f})")
    print(f"  Lock2 (k=1,C_crit=50 median cost):    worst={lock2_worst} ({lock2_c50[lock2_worst]:.2f}) "
          f"best={lock2_best} ({lock2_c50[lock2_best]:.2f})")
    return dict(setting=name, label=cfg["label"], n_classes=n_classes,
                lock1_worst=lock1_worst, lock1_worst_val=round(float(lock1_k1t0[lock1_worst]), 4),
                lock1_best=lock1_best, lock1_best_val=round(float(lock1_k1t0[lock1_best]), 4),
                lock2_worst=lock2_worst, lock2_worst_val=round(float(lock2_c50[lock2_worst]), 2),
                lock2_best=lock2_best, lock2_best_val=round(float(lock2_c50[lock2_best]), 2),
                locks_agree_worst=bool(lock1_worst == lock2_worst),
                locks_agree_best=bool(lock1_best == lock2_best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS.keys()),
                    help="Comma-separated subset of: " + ",".join(SETTINGS.keys()))
    ap.add_argument("--vocabs", type=int, default=120)
    ap.add_argument("--missions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7, help="Vocab-draw seed, matches controller_robust.py's default.")
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)

    names = args.settings.split(",")
    rows = []
    for name in names:
        if name not in SETTINGS:
            raise ValueError(f"Unknown setting {name!r}, choose from {list(SETTINGS)}")
        rows.append(run_setting(name, SETTINGS[name], args.vocabs, args.missions, args.seed,
                                missions_seed_base=1000 * (list(SETTINGS).index(name) + 1)))

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUT_ROOT, "cross_setting_summary.csv")
    # merge with any already-written rows from a prior partial run (e.g. czu_dual_raw run later)
    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        prev = prev[~prev.setting.isin(summary.setting)]
        summary = pd.concat([prev, summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
