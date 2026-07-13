import sys
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.ntu_loader import UnifiedNTUDataset
from src.models.kinematic_encoder import KinematicEncoder

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 8
USE_CLASS_WEIGHTS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = PROJECT_ROOT / "Data_Processed" / "ntu_quats"
SAVE_DIR = PROJECT_ROOT / "trained_models" / "SUPERVISED"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RELEVANT_JSON_PATH = PROJECT_ROOT / "src" / "data" / "ntu_relevant_action_ids.json"

A_RE = re.compile(r"A(\d{3})")


# -----------------------------
# Same linear probe as downstream
# -----------------------------
class GestureHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, z):
        return self.fc(self.norm(z))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(dim=1, keepdim=True).clamp(1.0)
    return (x * w).sum(dim=1) / denom


# -----------------------------
# Dataset wrapper
# -----------------------------
class NTUSubsetDataset(Dataset):
    """
    Wraps UnifiedNTUDataset(mode='supervised') and optionally filters/remaps labels.
    Returns:
      x: (T, 68)
      mask: (T,)
      y: contiguous class id
    """
    def __init__(self, base_dataset, indices, remap_dict, targets):
        self.base_dataset = base_dataset
        self.indices = indices
        self.remap_dict = remap_dict
        self.targets = targets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, mask, y = self.base_dataset[base_idx]

        y = int(y)
        if self.remap_dict is not None:
            # base label is assumed 0..119, convert to NTU action id 1..120 first
            y = self.remap_dict[y + 1]

        return x, mask, torch.tensor(y, dtype=torch.long)


def parse_action_id_from_name(name: str):
    stem = Path(name).stem
    match = A_RE.search(stem)
    if match is None:
        return None
    return int(match.group(1))


def infer_file_list(base_dataset):
    if hasattr(base_dataset, "file_list"):
        return list(base_dataset.file_list)
    raise AttributeError(
        "UnifiedNTUDataset does not expose file_list. "
        "This script expects base_dataset.file_list to exist."
    )


def load_relevant_ids():
    with open(RELEVANT_JSON_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)

    action_ids = [int(x) for x in obj["action_ids"]]
    action_names = {int(k): v for k, v in obj.get("action_names", {}).items()}
    return action_ids, action_names


def make_class_weights(targets, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for t in targets:
        counts[t] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return weights.to(DEVICE)


def build_dataset(mode_name: str):
    """
    mode_name:
      'all120'
      'relevant23'
    """
    base_dataset = UnifiedNTUDataset(data_path=DATA_PATH, mode="supervised")
    file_list = infer_file_list(base_dataset)

    if len(file_list) != len(base_dataset):
        raise ValueError(
            f"file_list length ({len(file_list)}) does not match dataset length ({len(base_dataset)})."
        )

    if mode_name == "all120":
        indices = list(range(len(base_dataset)))
        remap_dict = None

        targets = []
        for fp in file_list:
            action_id = parse_action_id_from_name(fp)
            if action_id is None:
                raise ValueError(f"Could not parse NTU action id from filename: {fp}")
            targets.append(action_id - 1)

        num_classes = 120
        meta = {
            "mode_name": "all120",
            "action_ids": list(range(1, 121))
        }

        dataset = NTUSubsetDataset(
            base_dataset=base_dataset,
            indices=indices,
            remap_dict=remap_dict,
            targets=targets
        )
        return dataset, num_classes, meta

    if mode_name == "relevant23":
        relevant_ids, action_names = load_relevant_ids()
        relevant_ids = sorted(relevant_ids)

        ntu_id_to_local = {aid: i for i, aid in enumerate(relevant_ids)}

        indices = []
        targets = []

        for i, fp in enumerate(file_list):
            action_id = parse_action_id_from_name(fp)
            if action_id is None:
                raise ValueError(f"Could not parse NTU action id from filename: {fp}")

            if action_id in ntu_id_to_local:
                indices.append(i)
                targets.append(ntu_id_to_local[action_id])

        num_classes = len(relevant_ids)
        meta = {
            "mode_name": "relevant23",
            "action_ids": relevant_ids,
            "action_names": action_names
        }

        dataset = NTUSubsetDataset(
            base_dataset=base_dataset,
            indices=indices,
            remap_dict=ntu_id_to_local,
            targets=targets
        )
        return dataset, num_classes, meta

    raise ValueError("mode_name must be 'all120' or 'relevant23'")


def train_one(mode_name: str):
    dataset, num_classes, meta = build_dataset(mode_name)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    head = GestureHead(num_classes=num_classes).to(DEVICE)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    if USE_CLASS_WEIGHTS:
        class_weights = make_class_weights(dataset.targets, num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    print("\n" + "=" * 70)
    print(f"Supervised NTU pretraining: {mode_name}")
    print(f"Device: {DEVICE}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Num classes: {num_classes}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 70)

    final_loss = None

    for epoch in range(EPOCHS):
        encoder.train()
        head.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"{mode_name} | epoch {epoch + 1}/{EPOCHS}")
        for x, mask, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                zt = encoder(x, mask=mask)   # (B, T, 512)
                z = masked_mean(zt, mask)    # (B, 512)
                logits = head(z)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        final_loss = total_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} | train loss: {final_loss:.6f}")

    # --------- ONLY CHANGE: checkpoint filenames ---------
    if mode_name == "all120":
        ckpt_name = f"sup_lr_ntu120_epoch_{EPOCHS}.pth"
    else:
        ckpt_name = f"sup_lr_ntuRelevant23_epoch_{EPOCHS}.pth"
    # ----------------------------------------------------

    ckpt_path = SAVE_DIR / ckpt_name
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "num_classes": num_classes,
            "pretrain_mode": mode_name,
            "action_ids": meta["action_ids"],
            "final_loss": final_loss
        },
        ckpt_path
    )

    print(f"Saved checkpoint: {ckpt_path.name}")


def main():
    train_one("all120")
    train_one("relevant23")
    print(f"\nDone. Checkpoints saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()