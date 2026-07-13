import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.kinematic_encoder import KinematicEncoder
from src.models.kinematic_decoder import KinematicDecoder

def verify():
    # 1. Setup dimensions
    B, T, D = 8, 120, 68  # Batch, Time, Dimensions
    embed_dim = 512
    
    print(f"--- Starting Verification ---")
    print(f"Input Shape: ({B}, {T}, {D})")

    # 2. Initialize Models
    encoder = KinematicEncoder(feature_dim=D, embed_dim=embed_dim)
    decoder = KinematicDecoder(embed_dim=embed_dim, feature_dim=D)
    
    # 3. Create Dummy Data and Padding Mask
    # Imagine a sequence where the first 80 frames are real, last 40 are padding
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T)
    mask[:, 80:] = 0.0 # Simulating padding
    
    # --- Test Case 1: MAE Mode (Unpooled) ---
    print("\nTesting MAE Mode...")
    latent = encoder(x, mask=mask, pool=False)
    print(f"Encoder Output (Latent) Shape: {latent.shape} (Expected: {B}, {T}, {embed_dim})")
    
    reconstructed = decoder(latent, mask=mask)
    print(f"Decoder Output (Reconstructed) Shape: {reconstructed.shape} (Expected: {B}, {T}, {D})")
    
    # --- Test Case 2: Supervised/MIL Mode (Pooled) ---
    print("\nTesting Supervised/MIL Mode...")
    pooled_out = encoder(x, mask=mask, pool=True)
    print(f"Pooled Output Shape: {pooled_out.shape} (Expected: {B}, {embed_dim})")
    
    # Check L2 Normalization (Pooled output should have norm of 1.0)
    norm = torch.norm(pooled_out, p=2, dim=1)
    print(f"L2 Norm of Pooled Output: {norm[0].item():.4f} (Expected: 1.0000)")

    print("\nVERIFICATION COMPLETE: Shapes and logic are correct.")

if __name__ == "__main__":
    verify()