import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DIFFERENCES FROM PREVIOUS (KINETICS-400) VERSION:
1. Input Dimension: Increased from 66D (filtered SMPL) to 68D (17 local quaternions).
2. Depth: Increased from 1 Transformer layer to 3. This is necessary to learn 
   higher-order physics from the much larger (43k) NTU dataset.
3. Positional Encoding: Added a learnable Parameter to give the model 'temporal 
   awareness' (knowing which frame is 1st vs 100th), which is vital for MAE.
4. Conditional Pooling: The 'pool' flag allows this single model to work for:
    - MAE: Returns (B, 120, 512) sequence for frame reconstruction.
    - Supervised/MIL: Returns (B, 512) summary for action classification.
5. L2 Normalization: Moved inside the pooling logic. This ensures we only 
   constrain the vector length for contrastive tasks, not for reconstruction tasks.
"""

class KinematicEncoder(nn.Module):
    def __init__(self, feature_dim=68, embed_dim=512, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # 1. Input Projection: 68D Quats -> 512D Latent Space
        self.input_projection = nn.Linear(feature_dim, embed_dim)
        
        # 2. Positional Encoding (B, 120, 512)
        # Adds 'time' information to the frames before they enter the transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, 120, embed_dim))
        
        # 3. Transformer Encoder (3 Layers Deep)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        
    def forward(self, x, mask=None, pool=False):
        """
        Input x shape: (B, T, 68)
        mask: Padding mask from the DataLoader (1 for real, 0 for pad)
        pool: If True, returns a single summary vector (B, 512)
        """
        # Step 1: Project to 512D and add time info
        x = self.input_projection(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Step 2: Pass through Transformer
        # We convert the 1/0 mask to a Boolean mask (True means 'Ignore')
        if mask is not None:
            padding_mask = (mask == 0)
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer_encoder(x)
            
        # Step 3: Pooling (The 'Summary' Logic)
        if pool:
            # Average across the time (T) dimension: (B, T, 512) -> (B, 512)
            x = x.mean(dim=1) 
            # L2 Normalize only when pooling (for Contrastive/Classification)
            return F.normalize(x, p=2, dim=1)
        
        # Default: Return the full sequence (For MAE/Self-Supervised)
        return x