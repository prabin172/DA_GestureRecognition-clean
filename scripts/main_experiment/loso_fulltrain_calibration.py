"""
loso_fulltrain_calibration.py

Pilot experiment:
  1. Initialize an encoder from a pretraining/adaptation method.
  2. Supervised train on all available clips from the 4 non-held-out subjects,
     using only those 4 subjects for validation/checkpoint selection.
  3. Calibrate on k labeled clips/class from the held-out subject.
  4. Evaluate on the remaining held-out subject clips.

This is meant to answer the practical deployment question:
  historical users -> new user -> few labeled calibration clips.

Important:
  - The held-out subject is not used during base training or base checkpoint
    selection.
  - Calibration does use labeled clips from the held-out subject by design.
  - Calibration summaries report final-epoch performance; best test during
    calibration is logged only as a diagnostic.

Run from project root:
  .venv/bin/python loso_fulltrain_calibration.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "Data_Processed" / "imu_quats_v2" / "index.csv").exists() and (p / "src").exists():
            return p
    raise RuntimeError("Could not find project root.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMU data dir: override with env LOSO_IMU_DIR (e.g. the v2 position-derived set).
_imu_env = os.environ.get("LOSO_IMU_DIR")
IMU_DIR = (
    (Path(_imu_env) if Path(_imu_env).is_absolute() else PROJECT_ROOT / _imu_env)
    if _imu_env else PROJECT_ROOT / "Data_Processed" / "imu_quats"
)
IMU_INDEX = IMU_DIR / "index.csv"

MAX_FRAMES = 120
MIN_FRAMES = 8
DROP_UNKNOWN = True

# Set to None for all five folds after the pilot fold looks sane.
RUN_ONLY_TEST_SUBJECT: Optional[str] = None

K_VALUES = [0, 1, 3, 5, 10]
CALIBRATION_MODES = ["head_only"]
# Optional later:
# CALIBRATION_MODES = ["head_only", "full_light"]

BASE_EPOCHS = 80
CALIB_EPOCHS = 30
VAL_FRAC = 0.20

BATCH_SIZE = 128
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = False

BASE_ENCODER_LR = 1e-4
BASE_HEAD_LR = 1e-3
CALIB_HEAD_LR = 1e-3
CALIB_ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True

BASE_SEED = 42

OUT_DIR = PROJECT_ROOT / "trained_models" / "LOSO-fullTrainCalibrate"
LOG_DIR = OUT_DIR / "Logs"
MODEL_DIR = OUT_DIR / "models"
SPLIT_DIR = OUT_DIR / "splits"

SUMMARY_CSV = OUT_DIR / "summary.csv"
BASE_SUMMARY_CSV = OUT_DIR / "base_training_summary.csv"


METHODS = [
    {
        "tag": "scratch",
        "init": "scratch",
        "ckpt": None,
        "note": "random init, then base-trained on 4 subjects",
    },
    {
        "tag": "supLP120",
        "init": "pretrained",
        "ckpt": Path("trained_models/SUPERVISED/sup_lr_ntu120_epoch_50.pth"),
        "note": "source supervised NTU-120",
    },
    {
        "tag": "supMAE",
        "init": "pretrained",
        "ckpt": Path("trained_models/SUPMAE/supmae_best.pth"),
        "note": "hybrid supervised + MAE checkpoint used by active LOSO scripts",
    },
    {
        "tag": "mae",
        "init": "pretrained",
        "ckpt": Path("trained_models/MAE/mae_geoLoss_epoch_50.pth"),
        "note": "source MAE geodesic",
    },
    {
        "tag": "supcon",
        "init": "pretrained",
        "ckpt": Path("trained_models/ContrastiveNTU/supcon_epoch_50.pth"),
        "note": "source supervised contrastive (SupCon), NTU-120",
    },
    {
        "tag": "targetSupDANN_all120_adaptedInit",
        "init": "pretrained",
        "ckpt_template": "trained_models/TARGET_SUPERVISED_DANN_LOSO/models/targetSupDANN_all120_holdout{subject}_bestVal.pth",
        "note": "fold-specific target-supervised DANN encoder, selected by 4-subject validation",
    },
    {
        "tag": "sourceSupDANN_all120_adaptedInit",
        "init": "pretrained",
        "ckpt_template": "trained_models/SOURCE_SUPERVISED_DANN_ALL120_PRETRAIN_ADAPT_LOSO/models/sourceSupDANN_all120_holdout{subject}_bestVal.pth",
        "note": "fold-specific source-supervised DANN pretrain+adapt encoder, selected by 4-subject validation",
    },
]

for d in [OUT_DIR, LOG_DIR, MODEL_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def subject_seed(subject: str, offset: int = 0) -> int:
    digits = "".join(ch for ch in subject if ch.isdigit())
    return BASE_SEED + int(digits or 0) * 1000 + offset


def stable_tag_offset(tag: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(tag)) % 997


# -----------------------------
# Data
# -----------------------------
class IMUClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id: Dict[str, int], imu_dir: Path, max_frames: int):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.imu_dir = imu_dir
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        clip = np.load(self.imu_dir / row["file"]).astype(np.float32)  # [T, 17, 4]
        t = clip.shape[0]
        clip = clip.reshape(t, 68)

        out = np.zeros((self.max_frames, 68), dtype=np.float32)
        mask = np.zeros((self.max_frames,), dtype=np.int64)
        use_t = min(t, self.max_frames)
        out[:use_t] = clip[:use_t]
        mask[:use_t] = 1

        y = np.int64(self.label2id[str(row["label"])])
        return torch.from_numpy(out), torch.from_numpy(mask), torch.tensor(y, dtype=torch.long)


def load_full_df() -> pd.DataFrame:
    df = pd.read_csv(IMU_INDEX)
    df = df[df["n_frames_30hz"] >= MIN_FRAMES].copy()
    df["subject"] = df["session"].apply(lambda s: str(s).split("-")[0])
    if DROP_UNKNOWN:
        df = df[df["label"] != "unknown"].copy()
    return df.reset_index(drop=True)


def build_global_label_map(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    with (OUT_DIR / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    return label2id, id2label


def stratified_train_val_split(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []

    for _, chunk in df.groupby("label"):
        idx = chunk.index.to_numpy().copy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            train_parts.append(df.loc[idx])
            continue
        n_val = max(1, int(round(len(idx) * val_frac)))
        n_val = min(n_val, len(idx) - 1)
        val_parts.append(df.loc[idx[:n_val]])
        train_parts.append(df.loc[idx[n_val:]])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_df, val_df


def select_k_shot_within_subject(
    subject_df: pd.DataFrame,
    labels_all: List[str],
    k: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int], List[int]]:
    if k == 0:
        empty = subject_df.iloc[0:0].copy().reset_index(drop=True)
        test_df = subject_df.copy().reset_index(drop=True)
        train_counts = {lab: 0 for lab in labels_all}
        test_counts = {lab: int((test_df["label"] == lab).sum()) for lab in labels_all}
        return empty, test_df, train_counts, test_counts, []

    rng = np.random.default_rng(seed)
    calib_parts = []
    chosen_indices: List[int] = []

    for lab in labels_all:
        chunk = subject_df[subject_df["label"] == lab]
        if len(chunk) == 0:
            continue
        take = min(k, len(chunk))
        idx = rng.choice(chunk.index.to_numpy(), size=take, replace=False)
        chosen_indices.extend(idx.tolist())
        calib_parts.append(subject_df.loc[idx])

    if len(calib_parts) == 0:
        raise ValueError("No calibration samples selected.")

    calib_df = pd.concat(calib_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    eval_df = subject_df.drop(index=chosen_indices).copy().reset_index(drop=True)

    train_counts = {lab: int((calib_df["label"] == lab).sum()) for lab in labels_all}
    test_counts = {lab: int((eval_df["label"] == lab).sum()) for lab in labels_all}
    return calib_df, eval_df, train_counts, test_counts, sorted(chosen_indices)


def make_loader(df: pd.DataFrame, label2id: Dict[str, int], shuffle: bool) -> DataLoader:
    return DataLoader(
        IMUClipDataset(df, label2id, IMU_DIR, MAX_FRAMES),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )


# -----------------------------
# Model helpers
# -----------------------------
class GestureHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Linear(512, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(dim=1, keepdim=True).clamp(1.0)
    return (x * w).sum(dim=1) / denom


def make_class_weights(df: pd.DataFrame, label2id: Dict[str, int]) -> Optional[torch.Tensor]:
    if len(df) == 0:
        return None
    counts = np.zeros(len(label2id), dtype=np.float32)
    for lab, c in df["label"].value_counts().items():
        counts[label2id[str(lab)]] = float(c)
    counts = np.clip(counts, 1.0, None)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def resolve_method_ckpt(method: Dict, test_subject: str) -> Optional[Path]:
    if method.get("ckpt") is not None:
        return Path(method["ckpt"])
    if method.get("ckpt_template") is not None:
        return Path(str(method["ckpt_template"]).format(subject=test_subject))
    return None


def load_encoder_weights(encoder: KinematicEncoder, ckpt_rel_path: Path) -> None:
    ckpt_path = (PROJECT_ROOT / ckpt_rel_path).resolve() if not ckpt_rel_path.is_absolute() else ckpt_rel_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location=DEVICE)
    if "encoder_state_dict" in obj:
        state = obj["encoder_state_dict"]
    elif "encoder" in obj:
        state = obj["encoder"]
    elif "state_dict" in obj:
        state = obj["state_dict"]
    else:
        raise KeyError(f"No encoder weights found in checkpoint: {ckpt_path}")

    cleaned = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("encoder."):
            kk = kk[len("encoder.") :]
        cleaned[kk] = v

    encoder.load_state_dict(cleaned)


def clone_state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def safe_torch_save(obj: dict, path: Path, retries: int = 5, sleep_s: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(retries):
        try:
            torch.save(obj, path)
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"Could not save checkpoint: {path}") from last_err


@torch.no_grad()
def evaluate(encoder: nn.Module, head: nn.Module, loader: DataLoader, num_classes: int) -> Tuple[float, float]:
    encoder.eval()
    head.eval()

    correct = 0
    total = 0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        zt = encoder(x, mask=m)
        z = masked_mean(zt, m)
        logits = head(z)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()
        for c in range(num_classes):
            tp[c] += int(np.sum((p_np == c) & (y_np == c)))
            fp[c] += int(np.sum((p_np == c) & (y_np != c)))
            fn[c] += int(np.sum((p_np != c) & (y_np == c)))

    acc = 100.0 * correct / max(total, 1)
    f1s = []
    present = (tp + fn) > 0
    for c in range(num_classes):
        if not present[c]:
            continue
        precision = tp[c] / max(tp[c] + fp[c], 1)
        recall = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return acc, float(np.mean(f1s)) if f1s else 0.0


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    encoder.train()
    head.train()
    running_loss = 0.0
    n_batches = 0

    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        zt = encoder(x, mask=m)
        z = masked_mean(zt, m)
        logits = head(z)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


# -----------------------------
# Experiment stages
# -----------------------------
def initialize_model(method: Dict, test_subject: str, num_classes: int) -> Tuple[nn.Module, nn.Module, Optional[Path]]:
    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    head = GestureHead(num_classes).to(DEVICE)

    ckpt = resolve_method_ckpt(method, test_subject)
    if method["init"] == "pretrained":
        if ckpt is None:
            raise ValueError(f"Method {method['tag']} is pretrained but has no checkpoint.")
        load_encoder_weights(encoder, ckpt)

    return encoder, head, ckpt


def run_base_training(
    method: Dict,
    test_subject: str,
    train_pool_df: pd.DataFrame,
    label2id: Dict[str, int],
) -> Dict:
    seed = subject_seed(test_subject, offset=stable_tag_offset(method["tag"]))
    set_seed(seed)
    num_classes = len(label2id)

    if len(train_pool_df) == 0:
        # A2 N=0: no Xsens fine-tuning. Return the pretrained init (random head) straight
        # into calibration; the k-shot head_only calibration is the only Xsens supervision.
        encoder, head, ckpt = initialize_model(method, test_subject, num_classes)
        run_name = f"{test_subject}_{method['tag']}_baseFull"
        best_path = MODEL_DIR / f"{run_name}_bestVal.pth"
        enc_state = clone_state_dict_cpu(encoder)
        head_state = clone_state_dict_cpu(head)
        safe_torch_save(
            {
                "encoder": enc_state, "head": head_state,
                "test_subject": test_subject, "method": method["tag"],
                "stage": "base_no_finetune_N0",
                "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
                "method_note": method.get("note", ""),
            },
            best_path,
        )
        append_base_summary(
            {
                "test_subject": test_subject, "method": method["tag"], "init": method["init"],
                "train_samples": 0, "val_samples": 0, "best_epoch": 0,
                "best_val_acc": 0.0, "final_val_acc": 0.0, "final_val_macro_f1": 0.0,
                "base_ckpt": str(best_path), "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
                "note": (method.get("note", "") + " | N=0 no finetune").strip(),
            }
        )
        print(f"BASE | subject={test_subject} method={method['tag']} N=0 (no finetune, pretrained init)")
        return {"encoder_state": enc_state, "head_state": head_state,
                "base_ckpt": best_path, "best_epoch": 0, "best_val_acc": 0.0, "pretrained_ckpt": ckpt}

    base_train_df, base_val_df = stratified_train_val_split(train_pool_df, VAL_FRAC, seed)
    train_loader = make_loader(base_train_df, label2id, shuffle=True)
    val_loader = make_loader(base_val_df, label2id, shuffle=False)

    encoder, head, ckpt = initialize_model(method, test_subject, num_classes)
    class_weights = make_class_weights(base_train_df, label2id) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": BASE_ENCODER_LR},
            {"params": head.parameters(), "lr": BASE_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    tag = method["tag"]
    run_name = f"{test_subject}_{tag}_baseFull"
    epoch_log = LOG_DIR / f"{run_name}_epochs.csv"
    with epoch_log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "val_macro_f1", "best_val_acc_so_far"])

    best_val_acc = -1.0
    best_epoch = -1
    best_encoder = None
    best_head = None
    final_val_acc = 0.0
    final_val_f1 = 0.0

    print(f"\nBASE | subject={test_subject} method={tag} train={len(base_train_df)} val={len(base_val_df)}")
    for epoch in range(1, BASE_EPOCHS + 1):
        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, criterion)
        val_acc, val_f1 = evaluate(encoder, head, val_loader, num_classes)
        final_val_acc, final_val_f1 = val_acc, val_f1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_encoder = clone_state_dict_cpu(encoder)
            best_head = clone_state_dict_cpu(head)

        with epoch_log.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{best_val_acc:.6f}"])

        print(f"  epoch {epoch:03d}/{BASE_EPOCHS} loss={train_loss:.4f} val={val_acc:.2f}/{val_f1:.4f}")

    if best_encoder is None or best_head is None:
        best_encoder = clone_state_dict_cpu(encoder)
        best_head = clone_state_dict_cpu(head)

    best_path = MODEL_DIR / f"{run_name}_bestVal.pth"
    safe_torch_save(
        {
            "encoder": best_encoder,
            "head": best_head,
            "test_subject": test_subject,
            "method": tag,
            "stage": "base_full_train_4_subjects",
            "selection_metric": "val_acc_on_train_subject_split",
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "final_val_macro_f1": final_val_f1,
            "train_samples": len(base_train_df),
            "val_samples": len(base_val_df),
            "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
            "method_note": method.get("note", ""),
        },
        best_path,
    )

    append_base_summary(
        {
            "test_subject": test_subject,
            "method": tag,
            "init": method["init"],
            "train_samples": len(base_train_df),
            "val_samples": len(base_val_df),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "final_val_macro_f1": final_val_f1,
            "base_ckpt": str(best_path),
            "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
            "note": method.get("note", ""),
        }
    )

    return {
        "encoder_state": best_encoder,
        "head_state": best_head,
        "base_ckpt": best_path,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "pretrained_ckpt": ckpt,
    }


def run_calibration(
    method: Dict,
    test_subject: str,
    k: int,
    calibration_mode: str,
    base_state: Dict,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label2id: Dict[str, int],
) -> Dict:
    seed = subject_seed(test_subject, offset=100000 + k * 100 + len(calibration_mode))
    set_seed(seed)
    num_classes = len(label2id)

    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    head = GestureHead(num_classes).to(DEVICE)
    encoder.load_state_dict(base_state["encoder_state"])
    head.load_state_dict(base_state["head_state"])

    eval_loader = make_loader(eval_df, label2id, shuffle=False)

    if k == 0 or len(calib_df) == 0:
        acc, f1 = evaluate(encoder, head, eval_loader, num_classes)
        return {
            "test_subject": test_subject,
            "method": method["tag"],
            "k": k,
            "calibration_mode": "none",
            "calib_samples": 0,
            "eval_samples": len(eval_df),
            "final_acc": acc,
            "final_macro_f1": f1,
            "best_diag_acc": acc,
            "best_diag_macro_f1": f1,
            "best_diag_epoch": 0,
            "base_ckpt": str(base_state["base_ckpt"]),
            "pretrained_ckpt": str(base_state["pretrained_ckpt"]) if base_state["pretrained_ckpt"] is not None else "",
        }

    calib_loader = make_loader(calib_df, label2id, shuffle=True)
    class_weights = make_class_weights(calib_df, label2id) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if calibration_mode == "head_only":
        for p in encoder.parameters():
            p.requires_grad = False
        params = [{"params": head.parameters(), "lr": CALIB_HEAD_LR}]
    elif calibration_mode == "full_light":
        params = [
            {"params": encoder.parameters(), "lr": CALIB_ENCODER_LR},
            {"params": head.parameters(), "lr": CALIB_HEAD_LR},
        ]
    else:
        raise ValueError(f"Unknown calibration mode: {calibration_mode}")

    optimizer = optim.AdamW(params, weight_decay=WEIGHT_DECAY)

    run_name = f"{test_subject}_{method['tag']}_k{k}_{calibration_mode}"
    epoch_log = LOG_DIR / f"{run_name}_calibration_epochs.csv"
    with epoch_log.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["epoch", "calib_loss", "eval_acc", "eval_macro_f1", "best_diag_acc_so_far"])

    best_diag_acc = -1.0
    best_diag_f1 = 0.0
    best_diag_epoch = -1
    final_acc = 0.0
    final_f1 = 0.0

    print(f"CALIB | subject={test_subject} method={method['tag']} k={k} mode={calibration_mode} calib={len(calib_df)} eval={len(eval_df)}")
    for epoch in range(1, CALIB_EPOCHS + 1):
        calib_loss = train_one_epoch(encoder, head, calib_loader, optimizer, criterion)
        acc, f1 = evaluate(encoder, head, eval_loader, num_classes)
        final_acc, final_f1 = acc, f1

        if acc > best_diag_acc:
            best_diag_acc = acc
            best_diag_f1 = f1
            best_diag_epoch = epoch

        with epoch_log.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{calib_loss:.6f}", f"{acc:.6f}", f"{f1:.6f}", f"{best_diag_acc:.6f}"])

        print(f"  calib epoch {epoch:03d}/{CALIB_EPOCHS} loss={calib_loss:.4f} eval={acc:.2f}/{f1:.4f}")

    last_path = MODEL_DIR / f"{run_name}_last.pth"
    safe_torch_save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "test_subject": test_subject,
            "method": method["tag"],
            "k": k,
            "calibration_mode": calibration_mode,
            "stage": "heldout_subject_calibration",
            "epoch": CALIB_EPOCHS,
            "final_acc": final_acc,
            "final_macro_f1": final_f1,
            "best_diag_acc": best_diag_acc,
            "best_diag_macro_f1": best_diag_f1,
            "best_diag_epoch": best_diag_epoch,
            "base_ckpt": str(base_state["base_ckpt"]),
        },
        last_path,
    )

    return {
        "test_subject": test_subject,
        "method": method["tag"],
        "k": k,
        "calibration_mode": calibration_mode,
        "calib_samples": len(calib_df),
        "eval_samples": len(eval_df),
        "final_acc": final_acc,
        "final_macro_f1": final_f1,
        "best_diag_acc": best_diag_acc,
        "best_diag_macro_f1": best_diag_f1,
        "best_diag_epoch": best_diag_epoch,
        "base_ckpt": str(base_state["base_ckpt"]),
        "pretrained_ckpt": str(base_state["pretrained_ckpt"]) if base_state["pretrained_ckpt"] is not None else "",
        "calibration_ckpt": str(last_path),
    }


# -----------------------------
# CSV writing
# -----------------------------
def ensure_output_dirs() -> None:
    for d in [OUT_DIR, LOG_DIR, MODEL_DIR, SPLIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_csvs() -> None:
    ensure_output_dirs()
    if not BASE_SUMMARY_CSV.exists():
        with BASE_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [
                    "test_subject",
                    "method",
                    "init",
                    "train_samples",
                    "val_samples",
                    "best_epoch",
                    "best_val_acc",
                    "final_val_acc",
                    "final_val_macro_f1",
                    "base_ckpt",
                    "pretrained_ckpt",
                    "note",
                ]
            )

    if not SUMMARY_CSV.exists():
        with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [
                    "test_subject",
                    "method",
                    "k",
                    "calibration_mode",
                    "calib_samples",
                    "eval_samples",
                    "final_acc",
                    "final_macro_f1",
                    "best_diag_acc",
                    "best_diag_macro_f1",
                    "best_diag_epoch",
                    "base_ckpt",
                    "pretrained_ckpt",
                    "calibration_ckpt",
                ]
            )


def append_base_summary(row: Dict) -> None:
    with BASE_SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["test_subject"],
                row["method"],
                row["init"],
                row["train_samples"],
                row["val_samples"],
                row["best_epoch"],
                f"{row['best_val_acc']:.6f}",
                f"{row['final_val_acc']:.6f}",
                f"{row['final_val_macro_f1']:.6f}",
                row["base_ckpt"],
                row["pretrained_ckpt"],
                row["note"],
            ]
        )


def append_calibration_summary(row: Dict) -> None:
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["test_subject"],
                row["method"],
                row["k"],
                row["calibration_mode"],
                row["calib_samples"],
                row["eval_samples"],
                f"{row['final_acc']:.6f}",
                f"{row['final_macro_f1']:.6f}",
                f"{row['best_diag_acc']:.6f}",
                f"{row['best_diag_macro_f1']:.6f}",
                row["best_diag_epoch"],
                row["base_ckpt"],
                row["pretrained_ckpt"],
                row.get("calibration_ckpt", ""),
            ]
        )


def save_split_metadata(
    test_subject: str,
    k: int,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_counts: Dict[str, int],
    eval_counts: Dict[str, int],
    chosen_indices: List[int],
) -> None:
    path = SPLIT_DIR / f"{test_subject}_k{k}_calibration_split.json"
    obj = {
        "test_subject": test_subject,
        "k": k,
        "calib_samples": len(calib_df),
        "eval_samples": len(eval_df),
        "chosen_original_indices": chosen_indices,
        "calib_counts": train_counts,
        "eval_counts": eval_counts,
        "calib_files": calib_df["file"].tolist() if len(calib_df) else [],
        "eval_files": eval_df["file"].tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def parse_extra_method(value: str) -> Dict:
    if "=" in value:
        tag, ckpt = value.split("=", 1)
    elif ":" in value:
        tag, ckpt = value.split(":", 1)
    else:
        raise ValueError(f"Extra method must be tag=checkpoint or tag:checkpoint, got: {value}")
    tag = tag.strip()
    ckpt = ckpt.strip()
    if not tag or not ckpt:
        raise ValueError(f"Invalid extra method: {value}")
    return {
        "tag": tag,
        "init": "pretrained",
        "ckpt": Path(ckpt),
        "note": f"custom pretrained checkpoint: {ckpt}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOSO full-train then held-out-subject calibration.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory for summaries, models, logs, splits.")
    parser.add_argument("--methods", default=None, help="Comma-separated method tags to run after adding extra methods.")
    parser.add_argument(
        "--extra-method",
        action="append",
        default=[],
        help="Add custom pretrained method as tag=checkpoint. Can be repeated.",
    )
    parser.add_argument("--run-only-test-subject", default=None, help="Run only this held-out subject, e.g. sub11.")
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--calib-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--k-values", default=None, help="Comma-separated k values, e.g. 0,1,3,5,10.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit before training.")
    parser.add_argument("--base-seed", type=int, default=None, help="Override BASE_SEED (default 42). Controls k-shot sampling and train/val split.")
    parser.add_argument("--n-train-subjects", type=int, default=None,
                        help="A2: number of non-held-out subjects to use for base fine-tuning (nested, first N of sorted remaining). N=0 => no fine-tune, pretrained init straight into calibration. Default None = all 4.")
    return parser.parse_args()


N_TRAIN_SUBJECTS: Optional[int] = None  # A2: # non-held-out subjects used for base fine-tuning (None = all)


def apply_arg_overrides(args: argparse.Namespace) -> None:
    global OUT_DIR
    global LOG_DIR
    global MODEL_DIR
    global SPLIT_DIR
    global SUMMARY_CSV
    global BASE_SUMMARY_CSV
    global METHODS
    global RUN_ONLY_TEST_SUBJECT
    global BASE_EPOCHS
    global CALIB_EPOCHS
    global BATCH_SIZE
    global K_VALUES
    global BASE_SEED
    global N_TRAIN_SUBJECTS

    OUT_DIR = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    LOG_DIR = OUT_DIR / "Logs"
    MODEL_DIR = OUT_DIR / "models"
    SPLIT_DIR = OUT_DIR / "splits"
    SUMMARY_CSV = OUT_DIR / "summary.csv"
    BASE_SUMMARY_CSV = OUT_DIR / "base_training_summary.csv"

    methods = list(METHODS)
    for value in args.extra_method:
        for part in [p.strip() for p in value.split(",") if p.strip()]:
            methods.append(parse_extra_method(part))

    selected = parse_csv_arg(args.methods)
    if selected is not None:
        lookup = {method["tag"]: method for method in methods}
        missing = [tag for tag in selected if tag not in lookup]
        if missing:
            raise ValueError(f"Unknown selected method tags: {missing}. Available: {sorted(lookup)}")
        methods = [lookup[tag] for tag in selected]
    METHODS = methods

    if args.run_only_test_subject is not None:
        RUN_ONLY_TEST_SUBJECT = args.run_only_test_subject
    if args.base_epochs is not None:
        BASE_EPOCHS = args.base_epochs
    if args.calib_epochs is not None:
        CALIB_EPOCHS = args.calib_epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.k_values is not None:
        K_VALUES = [int(x) for x in parse_csv_arg(args.k_values) or []]
    if args.base_seed is not None:
        BASE_SEED = args.base_seed
    if args.n_train_subjects is not None:
        N_TRAIN_SUBJECTS = args.n_train_subjects


def main() -> None:
    args = parse_args()
    apply_arg_overrides(args)

    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Base epochs: {BASE_EPOCHS} | Calib epochs: {CALIB_EPOCHS} | Batch size: {BATCH_SIZE}")
    ensure_csvs()

    df = load_full_df()
    label2id, _ = build_global_label_map(df)
    labels_all = sorted(label2id.keys())
    subjects = sorted(df["subject"].unique().tolist())

    if RUN_ONLY_TEST_SUBJECT is not None:
        if RUN_ONLY_TEST_SUBJECT not in subjects:
            raise ValueError(f"RUN_ONLY_TEST_SUBJECT={RUN_ONLY_TEST_SUBJECT} not in {subjects}")
        subjects = [RUN_ONLY_TEST_SUBJECT]

    print(f"Subjects: {subjects}")
    print(f"Total clips: {len(df)} | Classes: {len(label2id)}")
    print(f"Methods: {[m['tag'] for m in METHODS]}")
    print(f"K values: {K_VALUES} | calibration modes: {CALIBRATION_MODES}")

    if args.dry_run:
        print("\nDry run complete. No training started.")
        for method in METHODS:
            ckpt = method.get("ckpt", method.get("ckpt_template", ""))
            print(f"  {method['tag']}: init={method['init']} ckpt={ckpt}")
        return

    for test_subject in subjects:
        subject_df = df[df["subject"] == test_subject].copy()
        train_pool_df = df[df["subject"] != test_subject].copy().reset_index(drop=True)

        if N_TRAIN_SUBJECTS is not None:
            pool_subjects = sorted(train_pool_df["subject"].unique().tolist())
            keep = pool_subjects[:N_TRAIN_SUBJECTS]
            train_pool_df = train_pool_df[train_pool_df["subject"].isin(keep)].reset_index(drop=True)
            print(f"[A2] test={test_subject} N={N_TRAIN_SUBJECTS} finetune_subjects={keep} clips={len(train_pool_df)}")

        split_cache = {}
        for k in K_VALUES:
            split_seed = subject_seed(test_subject, offset=50000 + k)
            calib_df, eval_df, train_counts, eval_counts, chosen_indices = select_k_shot_within_subject(
                subject_df=subject_df,
                labels_all=labels_all,
                k=k,
                seed=split_seed,
            )
            save_split_metadata(test_subject, k, calib_df, eval_df, train_counts, eval_counts, chosen_indices)
            split_cache[k] = (calib_df, eval_df)

        for method in METHODS:
            base_state = run_base_training(method, test_subject, train_pool_df, label2id)

            for k in K_VALUES:
                calib_df, eval_df = split_cache[k]
                if k == 0:
                    row = run_calibration(method, test_subject, k, "none", base_state, calib_df, eval_df, label2id)
                    append_calibration_summary(row)
                    print(f"SUMMARY | {test_subject} {method['tag']} k=0 acc={row['final_acc']:.2f} f1={row['final_macro_f1']:.4f}")
                    continue

                for calibration_mode in CALIBRATION_MODES:
                    row = run_calibration(method, test_subject, k, calibration_mode, base_state, calib_df, eval_df, label2id)
                    append_calibration_summary(row)
                    print(
                        f"SUMMARY | {test_subject} {method['tag']} k={k} {calibration_mode} "
                        f"final={row['final_acc']:.2f}/{row['final_macro_f1']:.4f} "
                        f"best_diag={row['best_diag_acc']:.2f}@{row['best_diag_epoch']}"
                    )

    print(f"\nDone. Summary: {SUMMARY_CSV}")
    print(f"Base summary: {BASE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
