import torch
import torch.nn as nn
from src.core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("FeatureRefinementTransformer")
class FeatureRefinementTransformer(nn.Module):
    """
    A Transformer-based classification head that operates on the sequence of 
    latent feature residuals extracted from the diffusion trajectory.
    
    Implements:
    1. Learnable CLS Token
    2. Learnable Positional Embeddings
    3. Multi-Head Self-Attention Encoder
    """
    def __init__(
        self, 
        input_dim: int, 
        seq_length: int, 
        projection_dim: int = 512, 
        num_heads: int = 8, 
        num_layers: int = 2, 
        num_classes: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection_dim = projection_dim
        
        # 1. Input Projection & Norm
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, projection_dim)
        
        # 2. Learnable Parameters
        # [1, 1, dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
        # [1, seq_len + 1, dim]
        self.positional_embedding = nn.Parameter(torch.randn(1, seq_length + 1, projection_dim))
        
        # 3. Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dropout=dropout,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        """
        Args:
            x (Tensor): Sequence of features [Batch, Seq_Len, Input_Dim]
        Returns:
            logits (Tensor): Classification logits [Batch, Num_Classes]
            features (Tensor): Encoded sequence [Batch, Seq_Len+1, Projection_Dim]
        """
        B, S, D = x.shape
        
        # Normalize and Project
        x = self.input_norm(x)
        x = self.projection(x) # [B, S, proj_dim]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, proj_dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, S+1, proj_dim]
        
        # Add Positional Embedding
        # Note: If input sequence length varies (dynamic trajectory), slicing might be needed
        if x.shape[1] <= self.positional_embedding.shape[1]:
            x = x + self.positional_embedding[:, :x.shape[1], :]
        else:
             # Fallback or interpolation could be implemented here
             pass
             
        # Transformer Pass
        encoded_x = self.transformer_encoder(x)
        
        # Extract CLS token output (index 0)
        cls_output = encoded_x[:, 0, :]
        
        # Final Classification
        logits = self.fc_head(cls_output)
        
        return logits, encoded_x