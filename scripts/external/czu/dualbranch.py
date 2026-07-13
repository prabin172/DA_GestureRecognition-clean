#!/usr/bin/env python3
"""Dual-branch cross-modal: NTU-pretrained QUAT prior + from-scratch RAW-inertial branch.

Answers the question R6b couldn't: once the TARGET model uses its full raw signal (accel+gyro,
what CRC exploits), (a) does a learned encoder reach CRC-level accuracy, and (b) does the NTU
skeleton prior STILL add value on top of that strong target model?

Two branches:
  - RAW branch: from-scratch KinematicEncoder(feature_dim=60) on Data_Processed/czu_imu_raw
    (10 sensors x 6ch). This is the target-only signal NTU never had.
  - QUAT branch: KinematicEncoder(feature_dim=68) on Data_Processed/czu_imu_quats, initialized
    from an NTU-pretrained ckpt (scratch/mae/supMAE/supLP120). Carries the transferred prior.
Fuse: concat pooled embeddings (512+512) -> Linear head. Both branches share the frame mask
(clips are frame-aligned by construction).

Protocol mirrors loso_fulltrain_calibration.py EXACTLY (AdamW, enc lr 1e-4 / head lr 1e-3,
class-weighted CE, 80 base epochs best-val-selected, head-only 30-epoch calibrate) and REUSES the
byte-identical LOSO split JSONs from trained_models/CZU-IMU-LOSO/splits so results are paired with
R6b. Modes:
  --mode dual   : raw + quat(prior)      [prior lift on strong base]
  --mode raw    : raw only               [strong scratch baseline vs CRC]
  --mode quat   : quat only              [== R6b, sanity re-check]

Output -> trained_models/CZU-IMU-DUAL/<mode>_<prior>/summary.csv (+ per-fold logs)
"""
import os, sys, json, csv, argparse, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.models.kinematic_encoder import KinematicEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUAT_DIR = ROOT / "Data_Processed/czu_imu_quats"
RAW_DIR = ROOT / "Data_Processed/czu_imu_raw"
QUAT_DIM = 68
RAW_DIM = 60          # overridable via --raw-dim (target-strength dial)
SPLIT_SRC = ROOT / "trained_models/CZU-IMU-LOSO/splits"     # reuse byte-identical splits
SUBJECTS = ["cx", "cyy", "myj", "qyh", "zyh"]
KS = [0, 1, 3]
MAX_FRAMES = 120
MIN_FRAMES = 8
BASE_EPOCHS = 80
CALIB_EPOCHS = 30
VAL_FRAC = 0.20
BATCH_SIZE = 128
BASE_ENCODER_LR = 1e-4
BASE_HEAD_LR = 1e-3
CALIB_HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_SEED = 42
N_TRAIN_SUBJECTS: Optional[int] = None  # T5: # non-held-out subjects for base fine-tuning (None = all 4)

PRIOR_CKPT = {
    "scratch": None,
    "supLP120": "trained_models/SUPERVISED/sup_lr_ntu120_epoch_50.pth",
    "supMAE": "trained_models/SUPMAE/supmae_best.pth",
    "mae": "trained_models/MAE/mae_geoLoss_epoch_50.pth",
    "supcon": "trained_models/ContrastiveNTU/supcon_epoch_50.pth",
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def subject_seed(subject, offset=0):
    d = "".join(c for c in subject if c.isdigit())
    return BASE_SEED + int(d or 0) * 1000 + offset


# ---------------- data ----------------
class DualDataset(Dataset):
    def __init__(self, df, label2id, use_quat, use_raw):
        self.df = df.reset_index(drop=True); self.label2id = label2id
        self.use_quat = use_quat; self.use_raw = use_raw

    def __len__(self): return len(self.df)

    def _load(self, d, fn, feat):
        clip = np.load(d / fn).astype(np.float32).reshape(-1, feat)
        T = clip.shape[0]
        out = np.zeros((MAX_FRAMES, feat), np.float32); mask = np.zeros((MAX_FRAMES,), np.int64)
        u = min(T, MAX_FRAMES); out[:u] = clip[:u]; mask[:u] = 1
        return out, mask

    def __getitem__(self, i):
        r = self.df.iloc[i]; fn = r["file"]
        q, mq = self._load(QUAT_DIR, fn, QUAT_DIM) if self.use_quat else (np.zeros((MAX_FRAMES, QUAT_DIM), np.float32), None)
        rw, mr = self._load(RAW_DIR, fn, RAW_DIM) if self.use_raw else (np.zeros((MAX_FRAMES, RAW_DIM), np.float32), None)
        mask = mq if mq is not None else mr
        y = np.int64(self.label2id[str(r["label"])])
        return (torch.from_numpy(q), torch.from_numpy(rw), torch.from_numpy(mask),
                torch.tensor(y, dtype=torch.long))


class DualModel(nn.Module):
    def __init__(self, num_classes, use_quat, use_raw):
        super().__init__()
        self.use_quat, self.use_raw = use_quat, use_raw
        dim = 0
        if use_quat:
            self.quat_enc = KinematicEncoder(feature_dim=QUAT_DIM, embed_dim=512); dim += 512
        if use_raw:
            self.raw_enc = KinematicEncoder(feature_dim=RAW_DIM, embed_dim=512); dim += 512
        self.head = nn.Linear(dim, num_classes)

    @staticmethod
    def _pool(zt, m):
        w = m.float().unsqueeze(-1)
        return (zt * w).sum(1) / m.float().sum(1, keepdim=True).clamp(1.0)

    def embed(self, q, rw, m):
        parts = []
        if self.use_quat: parts.append(self._pool(self.quat_enc(q, mask=m), m))
        if self.use_raw:  parts.append(self._pool(self.raw_enc(rw, mask=m), m))
        return torch.cat(parts, dim=1)

    def forward(self, q, rw, m):
        return self.head(self.embed(q, rw, m))

    def encoder_params(self):
        ps = []
        if self.use_quat: ps += list(self.quat_enc.parameters())
        if self.use_raw:  ps += list(self.raw_enc.parameters())
        return ps


def load_prior_into(enc: KinematicEncoder, ckpt_rel: str):
    obj = torch.load(ROOT / ckpt_rel, map_location=DEVICE)
    # same resolution order as loso_fulltrain_calibration.load_encoder_weights
    if "encoder_state_dict" in obj:
        sd = obj["encoder_state_dict"]
    elif "encoder" in obj:
        sd = obj["encoder"]
    elif "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        raise KeyError(f"No encoder weights in {ckpt_rel}")
    cleaned = {}
    for k, v in sd.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        k2 = k2[len("encoder."):] if k2.startswith("encoder.") else k2
        cleaned[k2] = v
    enc.load_state_dict(cleaned)  # strict: prior must fully populate the 68-dim quat encoder
    print(f"    prior loaded: {len(cleaned)} tensors (strict) from {ckpt_rel}")


def load_df():
    df = pd.read_csv(QUAT_DIR / "index.csv")
    df = df[df["n_frames_30hz"] >= MIN_FRAMES].copy()
    df["subject"] = df["session"].astype(str).str.split("-").str[0]
    return df[df["label"] != "unknown"].reset_index(drop=True)


def strat_split(df, seed):
    rng = np.random.default_rng(seed); tr, va = [], []
    for _, ch in df.groupby("label"):
        idx = ch.index.to_numpy().copy(); rng.shuffle(idx)
        if len(idx) <= 1: tr.append(df.loc[idx]); continue
        nv = min(max(1, round(len(idx) * VAL_FRAC)), len(idx) - 1)
        va.append(df.loc[idx[:nv]]); tr.append(df.loc[idx[nv:]])
    return (pd.concat(tr).sample(frac=1, random_state=seed).reset_index(drop=True),
            pd.concat(va).sample(frac=1, random_state=seed + 1).reset_index(drop=True))


def class_weights(df, label2id):
    c = np.zeros(len(label2id), np.float32)
    for lab, n in df["label"].value_counts().items(): c[label2id[str(lab)]] = float(n)
    c = np.clip(c, 1.0, None); w = 1.0 / c; return torch.tensor(w / w.mean(), device=DEVICE)


def make_loader(df, label2id, use_quat, use_raw, shuffle):
    return DataLoader(DualDataset(df, label2id, use_quat, use_raw), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = tot = 0
    for q, rw, m, y in loader:
        q, rw, m, y = q.to(DEVICE), rw.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
        pred = model(q, rw, m).argmax(1)
        correct += int((pred == y).sum()); tot += int(y.numel())
    return 100.0 * correct / max(tot, 1)


def train_epoch(model, loader, opt, crit):
    model.train(); tl = 0; nb = 0
    for q, rw, m, y in loader:
        q, rw, m, y = q.to(DEVICE), rw.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(q, rw, m), y); loss.backward(); opt.step()
        tl += float(loss.item()); nb += 1
    return tl / max(nb, 1)


def run_fold(subject, prior, mode, label2id, df):
    use_quat = mode in ("dual", "quat"); use_raw = mode in ("dual", "raw")
    num_classes = len(label2id)
    seed = subject_seed(subject); set_seed(seed)
    # base train on the non-held-out subjects (T5: optionally truncated to first N)
    pool = df[df["subject"] != subject]
    if N_TRAIN_SUBJECTS is not None:
        pool_subjects = sorted(pool["subject"].unique().tolist())
        keep = pool_subjects[:N_TRAIN_SUBJECTS]
        pool = pool[pool["subject"].isin(keep)]
        print(f"  [T5] subj={subject} N={N_TRAIN_SUBJECTS} finetune_subjects={keep} clips={len(pool)}")

    model = DualModel(num_classes, use_quat, use_raw).to(DEVICE)
    if use_quat and PRIOR_CKPT[prior] is not None:
        load_prior_into(model.quat_enc, PRIOR_CKPT[prior])

    if len(pool) == 0:
        # N=0: no CZU fine-tuning. Pretrained init (random head) straight into k-shot calibration.
        base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  BASE subj={subject} mode={mode} prior={prior} N=0 (no finetune, pretrained init)")
    else:
        tr, va = strat_split(pool, seed)
        crit = nn.CrossEntropyLoss(weight=class_weights(tr, label2id))
        opt = optim.AdamW([{"params": model.encoder_params(), "lr": BASE_ENCODER_LR},
                           {"params": model.head.parameters(), "lr": BASE_HEAD_LR}], weight_decay=WEIGHT_DECAY)
        trl = make_loader(tr, label2id, use_quat, use_raw, True)
        val = make_loader(va, label2id, use_quat, use_raw, False)
        best_acc, best_sd = -1, None
        print(f"  BASE subj={subject} mode={mode} prior={prior} train={len(tr)} val={len(va)}")
        for ep in range(1, BASE_EPOCHS + 1):
            loss = train_epoch(model, trl, opt, crit); va_acc = evaluate(model, val)
            if va_acc > best_acc:
                best_acc = va_acc; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep % 20 == 0 or ep == 1:
                print(f"    ep{ep:03d} loss={loss:.3f} val={va_acc:.2f} best={best_acc:.2f}")
        model.load_state_dict(best_sd)
        base_sd = {k: v.clone() for k, v in best_sd.items()}

    results = []
    for k in KS:
        sp = json.load(open(SPLIT_SRC / f"{subject}_k{k}_calibration_split.json"))
        eval_df = df[df["file"].isin(sp["eval_files"])]
        set_seed(subject_seed(subject, 100000 + k * 100))
        model.load_state_dict(base_sd)
        if k == 0:
            acc = evaluate(model, make_loader(eval_df, label2id, use_quat, use_raw, False))
        else:
            calib_df = df[df["file"].isin(sp["calib_files"])]
            for p in model.encoder_params(): p.requires_grad = False  # head-only calibrate
            copt = optim.AdamW([{"params": model.head.parameters(), "lr": CALIB_HEAD_LR}], weight_decay=WEIGHT_DECAY)
            ccrit = nn.CrossEntropyLoss(weight=class_weights(calib_df, label2id))
            cl = make_loader(calib_df, label2id, use_quat, use_raw, True)
            el = make_loader(eval_df, label2id, use_quat, use_raw, False)
            acc = 0.0
            for ep in range(1, CALIB_EPOCHS + 1):
                train_epoch(model, cl, copt, ccrit); acc = evaluate(model, el)
            for p in model.encoder_params(): p.requires_grad = True
        print(f"  SUMMARY {subject} {mode}/{prior} k={k} acc={acc:.2f}")
        results.append(dict(test_subject=subject, mode=mode, prior=prior, k=k,
                            eval_samples=len(eval_df), final_acc=round(acc, 3)))
    return results


def main():
    global BASE_SEED, RAW_DIR, RAW_DIM, N_TRAIN_SUBJECTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dual", "raw", "quat"], required=True)
    ap.add_argument("--priors", default="scratch,supLP120,supMAE,mae")
    ap.add_argument("--seed", type=int, default=BASE_SEED,
                    help="Base seed (default 42 = existing runs). Use 43/44 for multi-seed.")
    ap.add_argument("--out-root", default="trained_models/CZU-IMU-DUAL",
                    help="Output root. Use a NEW dir for multi-seed/dial runs so existing results stay intact.")
    ap.add_argument("--raw-dir", default=str(RAW_DIR),
                    help="Dir for the raw/target branch signal (target-strength dial).")
    ap.add_argument("--raw-dim", type=int, default=RAW_DIM, help="Feature dim of the raw/target branch.")
    ap.add_argument("--n-train-subjects", type=int, default=None,
                    help="T5: number of non-held-out subjects to use for base fine-tuning "
                         "(nested, first N of sorted remaining). N=0 => no fine-tune, pretrained "
                         "init straight into calibration. Default None = all 4.")
    args = ap.parse_args()
    BASE_SEED = args.seed
    RAW_DIR = Path(args.raw_dir); RAW_DIM = args.raw_dim
    N_TRAIN_SUBJECTS = args.n_train_subjects
    print(f"CONFIG mode={args.mode} seed={BASE_SEED} out_root={args.out_root} raw_dir={RAW_DIR} raw_dim={RAW_DIM} n_train_subjects={N_TRAIN_SUBJECTS}")
    df = load_df()
    label2id = {l: i for i, l in enumerate(sorted(df["label"].unique()))}
    priors = args.priors.split(",") if args.mode != "raw" else ["scratch"]
    for prior in priors:
        out = ROOT / args.out_root / f"{args.mode}_{prior}"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in SUBJECTS:
            rows += run_fold(s, prior, args.mode, label2id, df)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        print(f"\nWrote {out}/summary.csv")
        if args.mode == "raw":
            break


if __name__ == "__main__":
    main()
