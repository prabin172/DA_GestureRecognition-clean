import torch
import torch.nn as nn

# This is the mirrored version of decoder, but lighter weight 
# (1 layer instead of 3) since it only needs to learn the 'inverse' 
# of the encoder's projection, not the full physics.

class KinematicDecoder(nn.Module):
    def __init__(self, embed_dim=512, feature_dim=68, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        
        # 1. Decoder Transformer Layer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 2. Output Projection: 512 -> 68 (Reconstruct the joints)
        self.output_projection = nn.Linear(embed_dim, feature_dim)
        
    def forward(self, x, mask=None):
        # x is the sequence from the Encoder: (B, T, 512)
        x = self.transformer_decoder(x, src_key_padding_mask=(mask == 0) if mask is not None else None)
        
        # Map back to 68D joint space
        # Returns (B, T, 68)
        return self.output_projection(x)
    
    # Meaning of this output
    # The decoder is trying to predict raw quaternion values, but since the output_projection is 
    # a linear layer, it can technically output any real number. However, during training, we will 
    # use a loss function (Mean squared loss) that encourages the output to be close to the original quaternions.
    # In practice, the model will learn to output values that are close to valid quaternions 
    # (which are typically normalized to have a magnitude of 1) because that’s what minimizes the reconstruction loss.
    # We might see cases where output quaternions are not perfectly "unit length", but should be acceptable for 
    # the pre-training phase, as the model is just learning the kinematic patterns.