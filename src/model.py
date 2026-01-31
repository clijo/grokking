"""1-layer Transformer for modular addition."""

import torch
import torch.nn as nn


class ModularAdditionTransformer(nn.Module):
    """
    1-layer Transformer for learning (a + b) mod p.

    Architecture:
    - Learned positional embeddings (2 positions)
    - Single transformer encoder layer
    - Linear classification head over p classes
    """

    def __init__(
        self,
        p: int,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model

        # Token embeddings: one for each possible value 0..p-1
        self.token_embedding = nn.Embedding(p, d_model)

        # Positional embeddings for 2 positions (a and b)
        self.pos_embedding = nn.Embedding(2, d_model)

        # Single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.output_head = nn.Linear(d_model, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 2) containing (a, b) pairs

        Returns:
            Logits of shape (batch_size, p)
        """
        # Token embeddings: (batch_size, 2, d_model)
        tok_emb = self.token_embedding(x)

        # Positional embeddings: (2, d_model) -> (1, 2, d_model)
        positions = torch.arange(2, device=x.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)

        # Combine embeddings
        h = tok_emb + pos_emb  # (batch_size, 2, d_model)

        # Transformer encoder
        h = self.transformer(h)  # (batch_size, 2, d_model)

        # Pool by taking the mean of both positions
        h = h.mean(dim=1)  # (batch_size, d_model)

        # Classification head
        logits = self.output_head(h)  # (batch_size, p)

        return logits
