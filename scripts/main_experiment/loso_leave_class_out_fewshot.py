"""
loso_leave_class_out_fewshot.py

OOV / leave-class-out few-shot protocol for wearable gesture recognition:
  1. Hold out one target subject.
  2. Treat one target gesture label as OOV.
  3. Base-train on the other 4 subjects after removing the OOV label.
  4. Expand the 21-way classifier to 22 classes.
  5. Calibrate the new held-out subject head-only with k clips/class.
  6. Evaluate on the remaining held-out subject clips, including OOV clips.

This tests whether a representation can support adding a new unseen gesture
class for a new user from a few labeled examples.

Run from project root:
  .venv/bin/python loso_leave_class_out_fewshot.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    raise RuntimeError("Could not find project root. Run this from the DA_GestureRecognition repo.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMU_DIR = PROJECT_ROOT / "Data_Processed" / "imu_quats"
IMU_INDEX = IMU_DIR / "index.csv"

MAX_FRAMES = 120
MIN_FRAMES = 8
DROP_UNKNOWN = True

RUN_ONLY_TEST_SUBJECT: Optional[str] = None
RUN_ONLY_OOV_LABELS: Optional[List[str]] = None

METHODS = ["scratch", "supLP120", "mae", "supMAE"]
K_VALUES = [1, 3, 5, 10]

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
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True

# Strict protocol: held-out evaluation clips are touched only after calibration.
CALIB_EVAL_EVERY_EPOCH = False

BASE_SEED = 42

OUT_DIR = PROJECT_ROOT / "trained_models" / "LOSO-LeaveClassOutFewShot"
LOG_DIR = OUT_DIR / "Logs"
MODEL_DIR = OUT_DIR / "models"
SPLIT_DIR = OUT_DIR / "splits"
PLOT_DIR = OUT_DIR / "plots"

SUMMARY_CSV = OUT_DIR / "summary.csv"
BASE_SUMMARY_CSV = OUT_DIR / "base_training_summary.csv"
LABEL_MAP_JSON = OUT_DIR / "label_map.json"

METHOD_CONFIGS: Dict[str, Dict] = {
    "scratch": {
        "tag": "scratch",
        "init": "scratch",
        "ckpt": None,
        "note": "random encoder initialization",
    },
    "supLP120": {
        "tag": "supLP120",
        "init": "pretrained",
        "ckpt": Path("trained_models/SUPERVISED/sup_lr_ntu120_epoch_50.pth"),
        "note": "source supervised NTU-120 encoder",
    },
    "mae": {
        "tag": "mae",
        "init": "pretrained",
        "ckpt": Path("trained_models/MAE/mae_geoLoss_epoch_50.pth"),
        "note": "source MAE geodesic encoder",
    },
    "supMAE": {
        "tag": "supMAE",
        "init": "pretrained",
        "ckpt": Path("trained_models/SUPMAE/supmae_best.pth"),
        "note": "source SupMAE encoder",
    },
}


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


def stable_tag_offset(tag: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(tag)) % 997


def subject_seed(subject: str, offset: int = 0) -> int:
    digits = "".join(ch for ch in subject if ch.isdigit())
    return BASE_SEED + int(digits or 0) * 1000 + offset


def combo_seed(test_subject: str, oov_label: str, method_tag: str, offset: int = 0) -> int:
    return subject_seed(test_subject, stable_tag_offset(oov_label) + stable_tag_offset(method_tag) + offset)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


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


def build_full_label_map(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = sorted(str(x) for x in df["label"].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LABEL_MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2)
    return labels, label2id, id2label


def build_known_label_map(labels_all: Sequence[str], oov_label: str) -> Dict[str, int]:
    known_labels = [label for label in labels_all if label != oov_label]
    return {label: i for i, label in enumerate(known_labels)}


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
    labels_all: Sequence[str],
    k: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, int], List[int]]:
    rng = np.random.default_rng(seed)
    calib_parts = []
    chosen_indices: List[int] = []

    for label in labels_all:
        chunk = subject_df[subject_df["label"] == label]
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

    calib_counts = {label: int((calib_df["label"] == label).sum()) for label in labels_all}
    eval_counts = {label: int((eval_df["label"] == label).sum()) for label in labels_all}
    return calib_df, eval_df, calib_counts, eval_counts, sorted(chosen_indices)


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
    for label, count in df["label"].value_counts().items():
        counts[label2id[str(label)]] = float(count)
    counts = np.clip(counts, 1.0, None)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def resolve_method(name: str) -> Dict:
    lookup = {key.lower(): value for key, value in METHOD_CONFIGS.items()}
    key = name.strip().lower()
    if key not in lookup:
        raise ValueError(f"Unknown method {name}. Available: {sorted(METHOD_CONFIGS)}")
    return dict(lookup[key])


def resolve_method_ckpt(method: Dict) -> Optional[Path]:
    if method.get("ckpt") is None:
        return None
    return Path(method["ckpt"])


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
    for key, value in state.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("encoder."):
            clean_key = clean_key[len("encoder.") :]
        cleaned[clean_key] = value

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
        except Exception as exc:
            last_err = exc
            time.sleep(sleep_s * (i + 1))
    raise RuntimeError(f"Could not save checkpoint: {path}") from last_err


def initialize_base_model(method: Dict, num_known_classes: int) -> Tuple[nn.Module, nn.Module, Optional[Path]]:
    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    head = GestureHead(num_known_classes).to(DEVICE)

    ckpt = resolve_method_ckpt(method)
    if method["init"] == "pretrained":
        if ckpt is None:
            raise ValueError(f"Method {method['tag']} is pretrained but has no checkpoint.")
        load_encoder_weights(encoder, ckpt)
    return encoder, head, ckpt


def expand_head_to_full_classes(
    base_head: GestureHead,
    known_label2id: Dict[str, int],
    full_label2id: Dict[str, int],
    oov_label: str,
) -> GestureHead:
    full_head = GestureHead(len(full_label2id)).to(DEVICE)
    with torch.no_grad():
        for label, known_idx in known_label2id.items():
            full_idx = full_label2id[label]
            full_head.net.weight[full_idx].copy_(base_head.net.weight[known_idx])
            full_head.net.bias[full_idx].copy_(base_head.net.bias[known_idx])
        # The OOV row remains randomly initialized by the fresh Linear module.
        _ = oov_label
    return full_head


# -----------------------------
# Metrics
# -----------------------------
def metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: Sequence[int],
) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0

    acc = 100.0 * float(np.mean(y_true == y_pred))
    f1s = []
    for class_id in class_ids:
        tp = int(np.sum((y_pred == class_id) & (y_true == class_id)))
        fp = int(np.sum((y_pred == class_id) & (y_true != class_id)))
        fn = int(np.sum((y_pred != class_id) & (y_true == class_id)))
        if (tp + fn) == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(f1)
    return acc, float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate_arrays(encoder: nn.Module, head: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    head.eval()
    all_true = []
    all_pred = []

    for x, m, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        m = m.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        zt = encoder(x, mask=m)
        z = masked_mean(zt, m)
        logits = head(z)
        pred = logits.argmax(dim=1)

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    if not all_true:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(all_true), np.concatenate(all_pred)


def evaluate_classification(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    class_ids: Sequence[int],
) -> Tuple[float, float]:
    y_true, y_pred = evaluate_arrays(encoder, head, loader)
    return metrics_from_arrays(y_true, y_pred, class_ids)


def evaluate_oov_protocol(
    encoder: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    full_label2id: Dict[str, int],
    oov_label: str,
    id2label: Dict[int, str],
) -> Dict:
    y_true, y_pred = evaluate_arrays(encoder, head, loader)
    all_class_ids = list(range(len(full_label2id)))
    oov_id = full_label2id[oov_label]
    known_class_ids = [idx for label, idx in full_label2id.items() if label != oov_label]

    acc_all, f1_all = metrics_from_arrays(y_true, y_pred, all_class_ids)

    known_mask = y_true != oov_id
    y_known = y_true[known_mask]
    p_known = y_pred[known_mask]
    acc_known, f1_known = metrics_from_arrays(y_known, p_known, known_class_ids)

    oov_mask = y_true == oov_id
    y_oov = y_true[oov_mask]
    p_oov = y_pred[oov_mask]
    acc_oov = 100.0 * float(np.mean(p_oov == oov_id)) if len(y_oov) else 0.0
    recall_oov = acc_oov

    pred_counts = {}
    if len(p_oov):
        unique, counts = np.unique(p_oov, return_counts=True)
        pred_counts = {id2label[int(idx)]: int(count) for idx, count in zip(unique, counts)}

    return {
        "final_acc_all": acc_all,
        "final_macro_f1_all": f1_all,
        "final_acc_known": acc_known,
        "final_macro_f1_known": f1_known,
        "final_acc_oov": acc_oov,
        "final_recall_oov": recall_oov,
        "eval_known_samples": int(len(y_known)),
        "eval_oov_samples": int(len(y_oov)),
        "oov_pred_counts": pred_counts,
    }


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
def run_base_training(
    method: Dict,
    test_subject: str,
    oov_label: str,
    train_pool_df: pd.DataFrame,
    known_label2id: Dict[str, int],
) -> Dict:
    tag = method["tag"]
    seed = combo_seed(test_subject, oov_label, tag)
    set_seed(seed)

    base_df = train_pool_df[train_pool_df["label"] != oov_label].copy().reset_index(drop=True)
    if len(base_df) == 0:
        raise ValueError(f"No base training rows remain for oov_label={oov_label}")

    base_train_df, base_val_df = stratified_train_val_split(base_df, VAL_FRAC, seed)
    train_loader = make_loader(base_train_df, known_label2id, shuffle=True)
    val_loader = make_loader(base_val_df, known_label2id, shuffle=False)

    encoder, head, ckpt = initialize_base_model(method, len(known_label2id))
    class_weights = make_class_weights(base_train_df, known_label2id) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": BASE_ENCODER_LR},
            {"params": head.parameters(), "lr": BASE_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    run_name = f"{test_subject}_oov-{safe_name(oov_label)}_{tag}_base21"
    epoch_log = LOG_DIR / f"{run_name}_epochs.csv"
    with epoch_log.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_acc", "val_macro_f1", "best_val_acc_so_far"])

    best_val_acc = -1.0
    best_epoch = -1
    best_encoder = None
    best_head = None
    final_val_acc = 0.0
    final_val_f1 = 0.0
    known_class_ids = list(range(len(known_label2id)))

    print(
        f"\nBASE | subject={test_subject} oov={oov_label} method={tag} "
        f"train={len(base_train_df)} val={len(base_val_df)} known_classes={len(known_label2id)}"
    )
    for epoch in range(1, BASE_EPOCHS + 1):
        train_loss = train_one_epoch(encoder, head, train_loader, optimizer, criterion)
        val_acc, val_f1 = evaluate_classification(encoder, head, val_loader, known_class_ids)
        final_val_acc, final_val_f1 = val_acc, val_f1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_encoder = clone_state_dict_cpu(encoder)
            best_head = clone_state_dict_cpu(head)

        with epoch_log.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{best_val_acc:.6f}"]
            )
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
            "oov_label": oov_label,
            "method": tag,
            "stage": "base_train_21_known_classes",
            "known_label2id": known_label2id,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "final_val_acc": final_val_acc,
            "final_val_macro_f1": final_val_f1,
            "base_train_samples": len(base_train_df),
            "base_val_samples": len(base_val_df),
            "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
        },
        best_path,
    )

    row = {
        "test_subject": test_subject,
        "oov_label": oov_label,
        "method": tag,
        "num_known_classes": len(known_label2id),
        "base_train_samples": len(base_train_df),
        "base_val_samples": len(base_val_df),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "final_val_macro_f1": final_val_f1,
        "base_ckpt": str(best_path),
        "pretrained_ckpt": str(ckpt) if ckpt is not None else "",
    }
    append_base_summary(row)

    return {
        "encoder_state": best_encoder,
        "head_state": best_head,
        "base_ckpt": best_path,
        "pretrained_ckpt": ckpt,
        "known_label2id": known_label2id,
        "base_train_samples": len(base_train_df),
        "base_val_samples": len(base_val_df),
    }


def run_calibration(
    method: Dict,
    test_subject: str,
    oov_label: str,
    k: int,
    base_state: Dict,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    full_label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Dict:
    tag = method["tag"]
    seed = combo_seed(test_subject, oov_label, tag, offset=100000 + k * 100)
    set_seed(seed)

    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    encoder.load_state_dict(base_state["encoder_state"])
    base_head = GestureHead(len(base_state["known_label2id"])).to(DEVICE)
    base_head.load_state_dict(base_state["head_state"])
    head = expand_head_to_full_classes(base_head, base_state["known_label2id"], full_label2id, oov_label)

    for p in encoder.parameters():
        p.requires_grad = False

    calib_loader = make_loader(calib_df, full_label2id, shuffle=True)
    eval_loader = make_loader(eval_df, full_label2id, shuffle=False)
    class_weights = make_class_weights(calib_df, full_label2id) if USE_CLASS_WEIGHTS else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([{"params": head.parameters(), "lr": CALIB_HEAD_LR}], weight_decay=WEIGHT_DECAY)

    run_name = f"{test_subject}_oov-{safe_name(oov_label)}_{tag}_k{k}_head_only"
    epoch_log = LOG_DIR / f"{run_name}_calibration_epochs.csv"
    with epoch_log.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(["epoch", "calib_loss", "eval_acc_all", "best_diag_acc_all_so_far"])

    best_diag_acc_all = -1.0
    best_diag_epoch = -1

    print(
        f"CALIB | subject={test_subject} oov={oov_label} method={tag} "
        f"k={k} calib={len(calib_df)} eval={len(eval_df)}"
    )
    for epoch in range(1, CALIB_EPOCHS + 1):
        calib_loss = train_one_epoch(encoder, head, calib_loader, optimizer, criterion)
        eval_cell = ""
        best_cell = ""

        if CALIB_EVAL_EVERY_EPOCH:
            metrics = evaluate_oov_protocol(encoder, head, eval_loader, full_label2id, oov_label, id2label)
            eval_cell = f"{metrics['final_acc_all']:.6f}"
            if metrics["final_acc_all"] > best_diag_acc_all:
                best_diag_acc_all = metrics["final_acc_all"]
                best_diag_epoch = epoch
            best_cell = f"{best_diag_acc_all:.6f}"

        with epoch_log.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, f"{calib_loss:.6f}", eval_cell, best_cell])

        if CALIB_EVAL_EVERY_EPOCH:
            print(f"  calib epoch {epoch:03d}/{CALIB_EPOCHS} loss={calib_loss:.4f} eval_all={eval_cell}")
        else:
            print(f"  calib epoch {epoch:03d}/{CALIB_EPOCHS} loss={calib_loss:.4f}")

    metrics = evaluate_oov_protocol(encoder, head, eval_loader, full_label2id, oov_label, id2label)
    if not CALIB_EVAL_EVERY_EPOCH:
        best_diag_acc_all = metrics["final_acc_all"]
        best_diag_epoch = CALIB_EPOCHS

    print(
        f"  final eval all={metrics['final_acc_all']:.2f} "
        f"known={metrics['final_acc_known']:.2f} oov_recall={metrics['final_recall_oov']:.2f}"
    )

    last_path = MODEL_DIR / f"{run_name}_last.pth"
    safe_torch_save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "test_subject": test_subject,
            "oov_label": oov_label,
            "method": tag,
            "k": k,
            "calibration_mode": "head_only",
            "stage": "heldout_subject_oov_head_only_calibration",
            "epoch": CALIB_EPOCHS,
            "metrics": metrics,
            "base_ckpt": str(base_state["base_ckpt"]),
            "pretrained_ckpt": str(base_state["pretrained_ckpt"]) if base_state["pretrained_ckpt"] is not None else "",
            "known_label2id": base_state["known_label2id"],
            "full_label2id": full_label2id,
        },
        last_path,
    )

    return {
        "test_subject": test_subject,
        "oov_label": oov_label,
        "method": tag,
        "k": k,
        "calibration_mode": "head_only",
        "base_train_samples": base_state["base_train_samples"],
        "base_val_samples": base_state["base_val_samples"],
        "calib_samples": len(calib_df),
        "eval_samples": len(eval_df),
        "eval_known_samples": metrics["eval_known_samples"],
        "eval_oov_samples": metrics["eval_oov_samples"],
        "final_acc_all": metrics["final_acc_all"],
        "final_macro_f1_all": metrics["final_macro_f1_all"],
        "final_acc_known": metrics["final_acc_known"],
        "final_macro_f1_known": metrics["final_macro_f1_known"],
        "final_acc_oov": metrics["final_acc_oov"],
        "final_recall_oov": metrics["final_recall_oov"],
        "best_diag_acc_all": best_diag_acc_all,
        "best_diag_epoch": best_diag_epoch,
        "base_ckpt": str(base_state["base_ckpt"]),
        "calibration_ckpt": str(last_path),
        "pretrained_ckpt": str(base_state["pretrained_ckpt"]) if base_state["pretrained_ckpt"] is not None else "",
        "oov_pred_counts_json": json.dumps(metrics["oov_pred_counts"], sort_keys=True),
    }


# -----------------------------
# CSV and split writing
# -----------------------------
SUMMARY_COLUMNS = [
    "test_subject",
    "oov_label",
    "method",
    "k",
    "calibration_mode",
    "base_train_samples",
    "base_val_samples",
    "calib_samples",
    "eval_samples",
    "eval_known_samples",
    "eval_oov_samples",
    "final_acc_all",
    "final_macro_f1_all",
    "final_acc_known",
    "final_macro_f1_known",
    "final_acc_oov",
    "final_recall_oov",
    "best_diag_acc_all",
    "best_diag_epoch",
    "base_ckpt",
    "calibration_ckpt",
    "pretrained_ckpt",
    "oov_pred_counts_json",
]

BASE_SUMMARY_COLUMNS = [
    "test_subject",
    "oov_label",
    "method",
    "num_known_classes",
    "base_train_samples",
    "base_val_samples",
    "best_epoch",
    "best_val_acc",
    "final_val_acc",
    "final_val_macro_f1",
    "base_ckpt",
    "pretrained_ckpt",
]


def ensure_output_dirs() -> None:
    for d in [OUT_DIR, LOG_DIR, MODEL_DIR, SPLIT_DIR, PLOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_csv(path: Path, columns: Sequence[str]) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(columns)


def ensure_csvs() -> None:
    ensure_output_dirs()
    ensure_csv(BASE_SUMMARY_CSV, BASE_SUMMARY_COLUMNS)
    ensure_csv(SUMMARY_CSV, SUMMARY_COLUMNS)


def append_base_summary(row: Dict) -> None:
    with BASE_SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["test_subject"],
                row["oov_label"],
                row["method"],
                row["num_known_classes"],
                row["base_train_samples"],
                row["base_val_samples"],
                row["best_epoch"],
                f"{row['best_val_acc']:.6f}",
                f"{row['final_val_acc']:.6f}",
                f"{row['final_val_macro_f1']:.6f}",
                row["base_ckpt"],
                row["pretrained_ckpt"],
            ]
        )


def append_summary(row: Dict) -> None:
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["test_subject"],
                row["oov_label"],
                row["method"],
                row["k"],
                row["calibration_mode"],
                row["base_train_samples"],
                row["base_val_samples"],
                row["calib_samples"],
                row["eval_samples"],
                row["eval_known_samples"],
                row["eval_oov_samples"],
                f"{row['final_acc_all']:.6f}",
                f"{row['final_macro_f1_all']:.6f}",
                f"{row['final_acc_known']:.6f}",
                f"{row['final_macro_f1_known']:.6f}",
                f"{row['final_acc_oov']:.6f}",
                f"{row['final_recall_oov']:.6f}",
                f"{row['best_diag_acc_all']:.6f}",
                row["best_diag_epoch"],
                row["base_ckpt"],
                row["calibration_ckpt"],
                row["pretrained_ckpt"],
                row["oov_pred_counts_json"],
            ]
        )


def save_split_metadata(
    test_subject: str,
    oov_label: str,
    k: int,
    calib_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    calib_counts: Dict[str, int],
    eval_counts: Dict[str, int],
    chosen_indices: List[int],
) -> None:
    path = SPLIT_DIR / f"{test_subject}_oov-{safe_name(oov_label)}_k{k}_split.json"
    obj = {
        "test_subject": test_subject,
        "oov_label": oov_label,
        "k": k,
        "calib_samples": len(calib_df),
        "eval_samples": len(eval_df),
        "chosen_original_indices": chosen_indices,
        "calib_counts": calib_counts,
        "eval_counts": eval_counts,
        "calib_files": calib_df["file"].tolist() if len(calib_df) else [],
        "eval_files": eval_df["file"].tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Dry run and aggregation
# -----------------------------
def print_dry_run(
    df: pd.DataFrame,
    labels_all: Sequence[str],
    subjects: Sequence[str],
    oov_labels: Sequence[str],
    methods: Sequence[Dict],
) -> None:
    print("\n" + "=" * 80)
    print("Dry-run summary")
    print("=" * 80)
    print(f"Subjects ({len(subjects)}): {list(subjects)}")
    print(f"Labels ({len(labels_all)}): {list(labels_all)}")
    print(f"Methods ({len(methods)}): {[m['tag'] for m in methods]}")
    print(f"K values: {K_VALUES}")

    counts = df.pivot_table(index="label", columns="subject", values="file", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=list(labels_all), columns=sorted(df["subject"].unique().tolist()), fill_value=0)
    print("\nPer-label counts per subject:")
    print(counts.to_string())

    print("\nK=10 feasibility for selected OOV labels:")
    for label in oov_labels:
        row = counts.loc[label]
        enough = {subject: int(row.get(subject, 0)) >= max(K_VALUES) for subject in subjects}
        status = "OK" if all(enough.values()) else "LOW"
        count_str = ", ".join(f"{subject}:{int(row.get(subject, 0))}" for subject in subjects)
        print(f"  {label:>12s} | {status} | {count_str}")

    expected_base = len(subjects) * len(oov_labels) * len(methods)
    expected_calib = expected_base * len(K_VALUES)
    print("\nExpected work:")
    print(f"  base training runs: {expected_base}")
    print(f"  calibration runs:   {expected_calib}")
    print("  base classifier classes: 21")
    print("  calibration/eval classifier classes: 22")


def aggregate_and_plot_summary() -> None:
    if not SUMMARY_CSV.exists():
        print(f"No summary found yet: {SUMMARY_CSV}")
        return
    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        print(f"Summary is empty: {SUMMARY_CSV}")
        return

    metrics = {
        "final_acc_oov": ("oov_accuracy_vs_k_by_method.png", "OOV recall/accuracy (%)"),
        "final_acc_all": ("overall_accuracy_vs_k_by_method.png", "Overall accuracy (%)"),
        "final_acc_known": ("known_accuracy_vs_k_by_method.png", "Known-class accuracy (%)"),
    }

    grouped = (
        df.groupby(["method", "k"], as_index=False)
        .agg(
            final_acc_all_mean=("final_acc_all", "mean"),
            final_acc_all_std=("final_acc_all", "std"),
            final_acc_known_mean=("final_acc_known", "mean"),
            final_acc_known_std=("final_acc_known", "std"),
            final_acc_oov_mean=("final_acc_oov", "mean"),
            final_acc_oov_std=("final_acc_oov", "std"),
            final_macro_f1_all_mean=("final_macro_f1_all", "mean"),
            n=("final_acc_all", "count"),
        )
        .sort_values(["method", "k"])
    )

    print("\n" + "=" * 80)
    print("Aggregate leave-class-out metrics by method and k")
    print("=" * 80)
    for _, row in grouped.iterrows():
        print(
            f"{row['method']:>8s} k={int(row['k']):>2d} "
            f"all={row['final_acc_all_mean']:6.2f} "
            f"known={row['final_acc_known_mean']:6.2f} "
            f"oov={row['final_acc_oov_mean']:6.2f} "
            f"n={int(row['n'])}"
        )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    for metric, (filename, ylabel) in metrics.items():
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        plt.figure(figsize=(7.5, 4.8))
        for method, part in grouped.groupby("method"):
            part = part.sort_values("k")
            plt.errorbar(
                part["k"],
                part[mean_col],
                yerr=part[std_col].fillna(0.0),
                marker="o",
                capsize=3,
                label=method,
            )
        plt.xlabel("k labeled clips/class from held-out subject")
        plt.ylabel(ylabel)
        plt.title("Leave-class-out few-shot calibration")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_path = PLOT_DIR / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot: {out_path}")

    for k in [1, 3]:
        subset = df[df["k"] == k].copy()
        if subset.empty:
            continue
        heat = subset.pivot_table(index="oov_label", columns="method", values="final_acc_oov", aggfunc="mean")
        if heat.empty:
            continue
        plt.figure(figsize=(max(6.0, 1.2 * len(heat.columns)), 8.5))
        plt.imshow(heat.values, aspect="auto", cmap="viridis")
        plt.colorbar(label="OOV recall/accuracy (%)")
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
        plt.yticks(range(len(heat.index)), heat.index)
        plt.title(f"OOV label x method recall, k={k}")
        plt.tight_layout()
        out_path = PLOT_DIR / f"oov_label_method_heatmap_k{k}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved heatmap: {out_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOSO leave-class-out few-shot protocol.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts and expected run sizes without training.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate/plot an existing summary.csv only.")
    parser.add_argument("--run-only-test-subject", default=None, help="Run only this held-out subject, e.g. sub11.")
    parser.add_argument("--oov-labels", default=None, help="Comma-separated OOV labels to run.")
    parser.add_argument("--methods", default=None, help="Comma-separated methods, e.g. scratch,mae,supMAE.")
    parser.add_argument("--base-epochs", type=int, default=None, help="Override BASE_EPOCHS.")
    parser.add_argument("--calib-epochs", type=int, default=None, help="Override CALIB_EPOCHS.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE.")
    return parser.parse_args()


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def apply_arg_overrides(args: argparse.Namespace) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    global RUN_ONLY_TEST_SUBJECT
    global BASE_EPOCHS
    global CALIB_EPOCHS
    global BATCH_SIZE

    if args.run_only_test_subject is not None:
        RUN_ONLY_TEST_SUBJECT = args.run_only_test_subject
    if args.base_epochs is not None:
        BASE_EPOCHS = args.base_epochs
    if args.calib_epochs is not None:
        CALIB_EPOCHS = args.calib_epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size

    return parse_csv_arg(args.oov_labels), parse_csv_arg(args.methods)


def main() -> None:
    args = parse_args()
    cli_oov_labels, cli_methods = apply_arg_overrides(args)

    if args.aggregate_only:
        aggregate_and_plot_summary()
        return

    ensure_output_dirs()
    df = load_full_df()
    labels_all, full_label2id, id2label = build_full_label_map(df)
    subjects = sorted(df["subject"].unique().tolist())

    if RUN_ONLY_TEST_SUBJECT is not None:
        if RUN_ONLY_TEST_SUBJECT not in subjects:
            raise ValueError(f"RUN_ONLY_TEST_SUBJECT={RUN_ONLY_TEST_SUBJECT} not in {subjects}")
        subjects = [RUN_ONLY_TEST_SUBJECT]

    if cli_oov_labels is not None:
        oov_labels = cli_oov_labels
    elif RUN_ONLY_OOV_LABELS is not None:
        oov_labels = RUN_ONLY_OOV_LABELS
    else:
        oov_labels = list(labels_all)

    missing_labels = [label for label in oov_labels if label not in full_label2id]
    if missing_labels:
        raise ValueError(f"OOV labels not found: {missing_labels}. Available labels: {labels_all}")

    method_names = cli_methods if cli_methods is not None else METHODS
    methods = [resolve_method(name) for name in method_names]

    print(f"Device: {DEVICE}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Subjects: {subjects}")
    print(f"OOV labels: {oov_labels}")
    print(f"Methods: {[m['tag'] for m in methods]}")
    print(f"K values: {K_VALUES}")
    print(f"Epochs: base={BASE_EPOCHS} calibration={CALIB_EPOCHS} | batch={BATCH_SIZE}")
    print(f"Total clips: {len(df)} | Classes: {len(labels_all)}")

    if args.dry_run:
        print_dry_run(df, labels_all, subjects, oov_labels, methods)
        return

    ensure_csvs()

    for test_subject in subjects:
        subject_df = df[df["subject"] == test_subject].copy()
        train_pool_df = df[df["subject"] != test_subject].copy().reset_index(drop=True)

        for oov_label in oov_labels:
            known_label2id = build_known_label_map(labels_all, oov_label)
            split_cache = {}

            for k in K_VALUES:
                split_seed = combo_seed(test_subject, oov_label, "split", offset=50000 + k)
                calib_df, eval_df, calib_counts, eval_counts, chosen_indices = select_k_shot_within_subject(
                    subject_df=subject_df,
                    labels_all=labels_all,
                    k=k,
                    seed=split_seed,
                )
                save_split_metadata(
                    test_subject=test_subject,
                    oov_label=oov_label,
                    k=k,
                    calib_df=calib_df,
                    eval_df=eval_df,
                    calib_counts=calib_counts,
                    eval_counts=eval_counts,
                    chosen_indices=chosen_indices,
                )
                split_cache[k] = (calib_df, eval_df)

            for method in methods:
                base_state = run_base_training(method, test_subject, oov_label, train_pool_df, known_label2id)

                for k in K_VALUES:
                    calib_df, eval_df = split_cache[k]
                    row = run_calibration(
                        method=method,
                        test_subject=test_subject,
                        oov_label=oov_label,
                        k=k,
                        base_state=base_state,
                        calib_df=calib_df,
                        eval_df=eval_df,
                        full_label2id=full_label2id,
                        id2label=id2label,
                    )
                    append_summary(row)
                    print(
                        f"SUMMARY | {test_subject} oov={oov_label} {method['tag']} k={k} "
                        f"all={row['final_acc_all']:.2f} known={row['final_acc_known']:.2f} "
                        f"oov={row['final_acc_oov']:.2f}"
                    )

    print(f"\nDone. Summary: {SUMMARY_CSV}")
    print(f"Base summary: {BASE_SUMMARY_CSV}")
    aggregate_and_plot_summary()


if __name__ == "__main__":
    main()
