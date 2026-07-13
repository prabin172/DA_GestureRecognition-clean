"""
WHY GEODESIC LOSS INSTEAD OF MSE?
1. Double Cover Property: In quaternions, q and -q represent the exact same 
   rotation. MSE treats them as opposites, while Geodesic Loss (Absolute Dot 
   Product) treats them as identical, preventing 'gradient confusion.'
2. Hypersphere Geometry: Quaternions live on a 4D unit sphere (S3). MSE 
   measures the straight-line (chordal) distance through the sphere, while 
   Geodesic Loss measures the true distance along the surface.
3. Unit Constraint: By normalizing predictions before calculating the dot 
   product, we force the model to optimize for valid rotations rather than 
   arbitrary 4D vectors.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Local Imports
from src.data.ntu_loader import UnifiedNTUDataset
from src.models.kinematic_encoder import KinematicEncoder
from src.models.kinematic_decoder import KinematicDecoder

# --- Configuration (Optimized for 24GB VRAM Lab PC) ---
BATCH_SIZE = 256    # Increased for high-VRAM throughput
EPOCHS = 50         # Starting with 50 for the first phase
LEARNING_RATE = 2e-4 
MASK_RATIO = 0.70   # 70% of the movement is hidden
NUM_WORKERS = 8     # High-speed data loading
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_PATH = PROJECT_ROOT / "Data_Processed" / "ntu_quats"
SAVE_DIR = PROJECT_ROOT / "trained_models" / "MAE"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="MAE pretraining with geodesic quaternion loss.")
    parser.add_argument("--ntu-root", type=Path, default=DATA_PATH, help="NTU quaternion root.")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR, help="Output checkpoint/log directory.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    parser.add_argument("--limit", type=int, default=None, help="Optional source clip limit for smoke tests.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def train(args=None):
    if args is None:
        args = parse_args()
    ntu_root = resolve_path(args.ntu_root)
    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Loader
    # mode='mae' ensures we get (quats, padding_mask)
    dataset = UnifiedNTUDataset(data_path=ntu_root, mode='mae')
    if args.limit is not None and args.limit > 0:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )

    # 2. Models
    encoder = KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    decoder = KinematicDecoder(embed_dim=512, feature_dim=68).to(DEVICE)

    # 3. Optimizer & Scaler (AMP)
    # AdamW is preferred for Transformers; weight_decay helps generalization
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize Scaler for Automatic Mixed Precision (FP16)
    scaler = torch.cuda.amp.GradScaler()
    
    # criterion = nn.MSELoss()

    print(f"--- IROS 2026: MAE Pre-training (Geodesic Loss) ---")
    print(f"Device: {DEVICE}")
    print(f"NTU root: {ntu_root}")
    print(f"Save dir: {save_dir}")
    print(f"Dataset Size: {len(dataset)} | Batch Size: {args.batch_size}")
    print(f"Masking Ratio: {args.mask_ratio * 100}%")

    latest_path = save_dir / "mae_latest.pth"
    best_path = save_dir / "mae_best.pth"
    history_path = save_dir / "mae_geodesic_history.csv"
    with history_path.open("w", encoding="utf-8") as f:
        f.write("epoch,avg_loss\n")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (quats, padding_mask) in enumerate(pbar):
            # Move data to GPU
            quats = quats.to(DEVICE, non_blocking=True)
            padding_mask = padding_mask.to(DEVICE, non_blocking=True)
            
            B, T, D = quats.shape

            # --- MAE MASKING LOGIC ---
            # Random noise to determine which frames to keep
            noise = torch.rand(B, T, device=DEVICE) 
            # Force padding frames (mask=0) to have high noise so they are always 'masked'
            noise = noise + (1 - padding_mask) * 10.0 

            # Sort and find indices to keep
            ids_shuffle = torch.argsort(noise, dim=1)
            num_keep = max(1, int(T * (1 - args.mask_ratio)))
            ids_keep = ids_shuffle[:, :num_keep]

            # Prepare masked input (only visible frames remain)
            masked_quats = torch.zeros_like(quats)
            batch_indices = torch.arange(B, device=DEVICE).unsqueeze(-1)
            masked_quats[batch_indices, ids_keep] = quats[batch_indices, ids_keep]

            # --- FORWARD PASS WITH AMP ---
            with torch.cuda.amp.autocast():
                # Encoder gets the sparse, masked sequence
                latent = encoder(masked_quats, mask=padding_mask)
                # Decoder tries to reconstruct the full 120-frame sequence
                reconstructed = decoder(latent, mask=padding_mask)

                # Identify which frames were hidden AND are not padding
                mae_mask = torch.ones(B, T, device=DEVICE)
                mae_mask[batch_indices, ids_keep] = 0.0 # Ignore visible frames in loss
                mae_mask = mae_mask * padding_mask       # Ignore padding frames in loss
                
                # --- GEODESIC LOSS LOGIC ---
                # Reshape to (B*T, 17, 4) to treat each joint rotation as a unit
                recon_q = reconstructed.view(-1, 17, 4)
                target_q = quats.view(-1, 17, 4)

                # 1. Normalize predictions to ensure they are on the unit hypersphere
                recon_q = F.normalize(recon_q, p=2, dim=-1)

                # 2. Calculate absolute dot product to handle double cover (|q . target|)
                # Perfect alignment = 1.0, Opposite (but same rotation) = 1.0
                dot_prod = torch.abs(torch.sum(recon_q * target_q, dim=-1))
                
                # 3. Loss = 1 - dot_product
                # Clamp for numerical stability
                geodesic_dist = 1.0 - torch.clamp(dot_prod, 0.0, 1.0)
                
                # 4. Mask the loss (only compute for hidden frames and non-padding)
                flat_mae_mask = mae_mask.view(-1)
                # Filter to only keep masked/valid frames, then take the mean
                loss = geodesic_dist[flat_mae_mask == 1.0].mean()

                # Convert loss to approximate degrees
                # We multiply by 2 because quaternion space is half-angle space
                with torch.no_grad():
                    avg_angle_rad = 2 * torch.acos(torch.clamp(1.0 - loss, 0.0, 1.0))
                    avg_angle_deg = torch.rad2deg(avg_angle_rad)

            # --- BACKWARD PASS WITH AMP ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            # Postfix shows the geodesic loss value
            pbar.set_postfix(geo_loss=f"{loss.item():.6f}", deg_err=f"{avg_angle_deg.item():.2f}°")

        avg_loss = total_loss / max(len(dataloader), 1)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_loss:.8f}\n")

        latest_obj = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'loss': avg_loss,
            'epoch': epoch + 1,
            'ntu_root': str(ntu_root),
            'mask_ratio': args.mask_ratio,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
        torch.save(latest_obj, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_obj = dict(latest_obj)
            best_obj["best_loss"] = best_loss
            torch.save(best_obj, best_path)
            print(f" -> Best checkpoint saved: {best_path.name} loss={best_loss:.6f}")

        # --- SMART CHECKPOINTING ---
        # Preserve original epoch checkpoint naming for backwards compatibility.
        # Save every 5 epochs or at the very last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = save_dir / f"mae_geoLoss_epoch_{epoch+1}.pth"
            torch.save(latest_obj, checkpoint_path)
            print(f" -> Checkpoint saved: {checkpoint_path.name}")

    print(f"Pre-training Complete. Models saved in: {save_dir}")
    print(f"Best checkpoint: {best_path}")

if __name__ == "__main__":
    train()
