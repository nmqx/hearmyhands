"""
GRU-based sign language classifier.

  1. Bidirectional GRU (2 layers) -- past+future context helps a lot for
     moving letters like J and Z and adds capacity at low extra cost.
  2. Masked mean-pooling over real frames instead of "last frame".
     Most sequences are padded; the "last frame" was often zero padding,
     so the original model had to learn to ignore garbage.
  3. Dropout between GRU and classifier (regularization -- your val loss
     was 10x the train loss, classic overfit).
  4. forward() accepts an optional mask; falls back to "use everything"
     if you don't pass one (so single-frame demos still work).
"""

import torch
import torch.nn as nn


class SignLanguageGRU(nn.Module):
    def __init__(
        self,
        input_size=42,
        hidden_size=96,
        num_layers=2,
        num_classes=26,
        dropout=0.3,
        bidirectional=True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Dropout(dropout),
            nn.Linear(out_size, num_classes),
        )

    def forward(self, x, mask=None):
        """
        x   : [B, T, 42]
        mask: [B, T] with 1 for real frames, 0 for padding. Optional.
        """
        h = self.embedding(x)        # [B, T, hidden]
        out, _ = self.gru(h)         # [B, T, hidden*(1+bi)]

        # masked mean pooling
        if mask is None:
            pooled = out.mean(dim=1)
        else:
            m = mask.unsqueeze(-1)             # [B, T, 1]
            denom = m.sum(dim=1).clamp(min=1)  # avoid /0
            pooled = (out * m).sum(dim=1) / denom

        return self.head(pooled)