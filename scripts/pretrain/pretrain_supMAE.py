# train_supMae.py
# Multi task pretraining: MAE reconstruction + supervised classification
# Shared encoder, separate decoder and classification head

import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.ntu_loader import UnifiedNTUDataset
from src.models.kinematic_encoder import KinematicEncoder
from src.models.kinematic_decoder import KinematicDecoder


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
NUM_WORKERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = PROJECT_ROOT / "Data_Processed" / "ntu_quats"
SAVE_DIR = PROJECT_ROOT / "trained_models" / "SUPMAE"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# MAE masking
MASK_RATIO = 0.70

# Loss balancing
LAMBDA_CE_FINAL = 1.0
CE_WARMUP_EPOCHS = 3
CE_RAMP_EPOCHS = 7

EMA_BETA = 0.98
EPS = 1e-8


# -----------------------------

# -----------------------------
class GestureHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, z):
        return self.fc(self.norm(z))
    
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: (B, T, C)
    # mask: (B, T) with 1 for valid, 0 for pad
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(dim=1, keepdim=True).clamp(1.0)
    return (x * w).sum(dim=1) / denom


def lambda_schedule(epoch_idx: int) -> float:
    # CE starts later, then ramps up
    if epoch_idx < CE_WARMUP_EPOCHS:
        return 0.0
    t = epoch_idx - CE_WARMUP_EPOCHS
    if CE_RAMP_EPOCHS <= 0:
        return LAMBDA_CE_FINAL
    alpha = min(max(t / CE_RAMP_EPOCHS, 0.0), 1.0)
    return alpha * LAMBDA_CE_FINAL


def geodesic_loss(reconstructed: torch.Tensor, target: torch.Tensor, mae_mask: torch.Tensor):
    # reconstructed, target: (B, T, 68)
    # mae_mask: (B, T) with 1 for masked valid frames, 0 otherwise
    B, T, D = reconstructed.shape

    recon_q = reconstructed.view(-1, 17, 4)
    target_q = target.view(-1, 17, 4)

    recon_q = F.normalize(recon_q, p=2, dim=-1)

    dot = torch.abs(torch.sum(recon_q * target_q, dim=-1))
    dot = torch.clamp(dot, 0.0, 1.0)

    dist = 1.0 - dot  # (B*T, 17)

    flat_mask = mae_mask.view(-1)  # (B*T)
    valid = flat_mask == 1.0
    if valid.sum().item() == 0:
        loss = dist.mean() * 0.0
        deg = torch.tensor(0.0, device=reconstructed.device)
        return loss, deg

    loss = dist[valid].mean()

    with torch.no_grad():
        avg_dot = 1.0 - loss
        avg_angle_rad = 2.0 * torch.acos(torch.clamp(avg_dot, 0.0, 1.0))
        avg_angle_deg = torch.rad2deg(avg_angle_rad)

    return loss, avg_angle_deg


def parse_args():
    parser = argparse.ArgumentParser(description="SupMAE pretraining: geodesic MAE + source CE.")
    parser.add_argument("--ntu-root", type=Path, default=DATA_PATH, help="NTU quaternion root.")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR, help="Output checkpoint/log directory.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    parser.add_argument("--lambda-ce-final", type=float, default=LAMBDA_CE_FINAL)
    parser.add_argument("--ce-warmup-epochs", type=int, default=CE_WARMUP_EPOCHS)
    parser.add_argument("--ce-ramp-epochs", type=int, default=CE_RAMP_EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="Optional source clip limit for smoke tests.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_args_to_globals(args) -> None:
    global BATCH_SIZE
    global EPOCHS
    global LEARNING_RATE
    global WEIGHT_DECAY
    global NUM_WORKERS
    global DATA_PATH
    global SAVE_DIR
    global MASK_RATIO
    global LAMBDA_CE_FINAL
    global CE_WARMUP_EPOCHS
    global CE_RAMP_EPOCHS

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    NUM_WORKERS = args.num_workers
    DATA_PATH = resolve_path(args.ntu_root)
    SAVE_DIR = resolve_path(args.save_dir)
    MASK_RATIO = args.mask_ratio
    LAMBDA_CE_FINAL = args.lambda_ce_final
    CE_WARMUP_EPOCHS = args.ce_warmup_epochs
    CE_RAMP_EPOCHS = args.ce_ramp_epochs
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def train(args=None):
    if args is None:
        args = parse_args()
    apply_args_to_globals(args)

    # mode="supervised" returns (x, padding_mask, y)
    dataset = UnifiedNTUDataset(data_path=DATA_PATH, mode="supervised")
    if args.limit is not None and args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))

    # Assumption: dataset labels are contiguous 0..C-1 already
    if hasattr(dataset, "num_classes"):
        num_classes = dataset.num_classes
    else:
        # fallback
        num_classes = 120

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    decoder = KinematicDecoder(embed_dim=512, feature_dim=68).to(DEVICE)
    head = GestureHead(num_classes=num_classes).to(DEVICE)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    ema_mae = None
    ema_ce = None

    latest_path = SAVE_DIR / "supmae_latest.pth"
    best_path = SAVE_DIR / "supmae_best.pth"
    history_path = SAVE_DIR / "supmae_history.csv"
    best_total = float("inf")

    with history_path.open("w", encoding="utf-8") as f:
        f.write("epoch,lambda_ce,avg_total,avg_mae,avg_ce,best_total\n")

    print("\n" + "=" * 70)
    print("SupMAE pretraining (MAE geodesic + CE) ")
    print(f"Device: {DEVICE}")
    print(f"NTU root: {DATA_PATH}")
    print(f"Save dir: {SAVE_DIR}")
    print(f"Dataset size: {len(dataset)} | Batch size: {BATCH_SIZE}")
    print(f"Mask ratio: {MASK_RATIO}")
    print(f"CE warmup: {CE_WARMUP_EPOCHS} | CE ramp: {CE_RAMP_EPOCHS} | lambda_final: {LAMBDA_CE_FINAL}")
    print("=" * 70)

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        head.train()

        lam = lambda_schedule(epoch)

        total_sum = 0.0
        mae_sum = 0.0
        ce_sum = 0.0

        pbar = tqdm(loader, desc=f"supMae | epoch {epoch+1}/{EPOCHS}")
        for x, padding_mask, y in pbar:
            x = x.to(DEVICE, non_blocking=True)                  # (B, T, 68)
            padding_mask = padding_mask.to(DEVICE, non_blocking=True)  # (B, T)
            y = y.to(DEVICE, non_blocking=True).long()           # (B,)

            B, T, D = x.shape

            # -----------------------------
            # MAE masking (same logic as train_mae_geodesic)
            # -----------------------------
            noise = torch.rand(B, T, device=DEVICE)
            noise = noise + (1 - padding_mask) * 10.0
            ids_shuffle = torch.argsort(noise, dim=1)
            num_keep = max(1, int(T * (1 - MASK_RATIO)))
            ids_keep = ids_shuffle[:, :num_keep]

            masked_x = torch.zeros_like(x)
            batch_idx = torch.arange(B, device=DEVICE).unsqueeze(-1)
            masked_x[batch_idx, ids_keep] = x[batch_idx, ids_keep]

            mae_mask = torch.ones(B, T, device=DEVICE)
            mae_mask[batch_idx, ids_keep] = 0.0
            mae_mask = mae_mask * padding_mask

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                # -----------------------------
                # Branch A: MAE reconstruction on masked input
                # -----------------------------
                latent_mae = encoder(masked_x, mask=padding_mask)
                recon = decoder(latent_mae, mask=padding_mask)
                loss_mae, deg_err = geodesic_loss(recon, x, mae_mask)

                # -----------------------------
                # Branch B: CE classification on unmasked input
                # -----------------------------
                latent_full = encoder(x, mask=padding_mask)          # (B, T, 512)
                pooled = masked_mean(latent_full, padding_mask)      # (B, 512)
                logits = head(pooled)
                loss_ce = F.cross_entropy(logits, y)

                # EMA scaling to reduce manual lambda sensitivity
                with torch.no_grad():
                    if ema_mae is None:
                        ema_mae = loss_mae.detach()
                        ema_ce = loss_ce.detach()
                    else:
                        ema_mae = EMA_BETA * ema_mae + (1 - EMA_BETA) * loss_mae.detach()
                        ema_ce = EMA_BETA * ema_ce + (1 - EMA_BETA) * loss_ce.detach()

                scale = (ema_mae / (ema_ce + EPS)).detach()
                loss_ce_scaled = loss_ce * scale

                loss_total = loss_mae + lam * loss_ce_scaled

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            total_sum += float(loss_total.item())
            mae_sum += float(loss_mae.item())
            ce_sum += float(loss_ce.item())

            pbar.set_postfix(
                total=f"{loss_total.item():.4f}",
                mae=f"{loss_mae.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                lam=f"{lam:.3f}",
                deg=f"{deg_err.item():.2f}"
            )

        avg_total = total_sum / max(len(loader), 1)
        avg_mae = mae_sum / max(len(loader), 1)
        avg_ce = ce_sum / max(len(loader), 1)

        print(f"Epoch {epoch+1}/{EPOCHS} | total {avg_total:.6f} | mae {avg_mae:.6f} | ce {avg_ce:.6f} | lam {lam:.3f}")
        current_best = min(best_total, avg_total)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{lam:.6f},{avg_total:.8f},{avg_mae:.8f},{avg_ce:.8f},{current_best:.8f}\n")

        # Save latest (overwrite every epoch)
        torch.save(
            {
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "head_state_dict": head.state_dict(),
                "epoch": epoch + 1,
                "avg_total": avg_total,
                "avg_mae": avg_mae,
                "avg_ce": avg_ce,
                "mask_ratio": MASK_RATIO,
                "lambda_ce": lam,
                "lambda_ce_final": LAMBDA_CE_FINAL,
                "ce_warmup_epochs": CE_WARMUP_EPOCHS,
                "ce_ramp_epochs": CE_RAMP_EPOCHS,
                "ntu_root": str(DATA_PATH),
                "num_classes": num_classes
            },
            latest_path
        )

        # Save best by total loss
        if avg_total < best_total:
            best_total = avg_total
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch + 1,
                    "avg_total": avg_total,
                    "avg_mae": avg_mae,
                    "avg_ce": avg_ce,
                    "mask_ratio": MASK_RATIO,
                    "lambda_ce": lam,
                    "lambda_ce_final": LAMBDA_CE_FINAL,
                    "ce_warmup_epochs": CE_WARMUP_EPOCHS,
                    "ce_ramp_epochs": CE_RAMP_EPOCHS,
                    "ntu_root": str(DATA_PATH),
                    "best_total": best_total,
                    "num_classes": num_classes
                },
                best_path
            )
            print(f"Saved best: epoch {epoch+1} total {best_total:.6f}")

    print(f"\nDone. Saved latest: {latest_path.name} | best: {best_path.name}")


if __name__ == "__main__":
    train()
