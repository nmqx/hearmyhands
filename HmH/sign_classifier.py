"""Sequential sign classifier (Ocarina GRU V2): 45 frames × 42 features -> 1 class.

L'archi est dupliquée (et non importée) depuis Modèle_Ocarina/Ocarina_GRU.py
pour ne pas dépendre du chemin accentué Modèle_Ocarina/ au runtime
(problèmes d'encoding observés sur certains systèmes).

V2 (vs V1) :
- Bidirectional GRU (2 layers, hidden=96)
- Masked mean-pool sur les frames réelles (et non output[:, -1, :])
- LayerNorm + Dropout dans la head
- Forward accepte un masque [B, T] (1 = real, 0 = padding)
"""
from __future__ import annotations

import json
import logging
import os
import string

import torch
import torch.nn as nn

log = logging.getLogger("hmh.sign")

DEFAULT_CLASSES = list(string.ascii_uppercase)  # 26 letters fallback


class _OcarinaGRU(nn.Module):
    """Architecture V2 — doit rester strictement alignée avec
    `Modèle_Ocarina/Ocarina_GRU.py::SignLanguageGRU` (mêmes shapes des
    couches pour pouvoir charger le state_dict produit par Train.py)."""

    def __init__(
        self,
        input_size: int = 42,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_classes: int = 26,
        dropout: float = 0.3,
        bidirectional: bool = True,
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embedding(x)         # [B, T, hidden]
        out, _ = self.gru(h)          # [B, T, hidden * (1+bi)]
        if mask is None:
            pooled = out.mean(dim=1)
        else:
            m = mask.unsqueeze(-1)             # [B, T, 1]
            denom = m.sum(dim=1).clamp(min=1)
            pooled = (out * m).sum(dim=1) / denom
        return self.head(pooled)


class SignClassifier:
    SEQ_LEN = 45            # doit matcher MAX_FRAMES de Modèle_Ocarina/Train.py
    INPUT_SIZE = 42

    def __init__(self, model: _OcarinaGRU, classes: list[str], device: torch.device):
        self.model = model
        self.classes = classes
        self.device = device

    @classmethod
    def try_load(cls, weights_path: str, classes_path: str | None = None) -> "SignClassifier | None":
        if not os.path.exists(weights_path):
            log.info("Ocarina weights not found at %s — sign classifier disabled", weights_path)
            return None
        try:
            classes = DEFAULT_CLASSES
            if classes_path and os.path.exists(classes_path):
                with open(classes_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list) and loaded:
                    classes = [str(c) for c in loaded]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _OcarinaGRU(num_classes=len(classes)).to(device)
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            log.info(
                "Ocarina GRU V2 loaded from %s (%d classes, hidden=%d, layers=%d, bidir=%s) on %s",
                weights_path, len(classes),
                model.gru.hidden_size, model.gru.num_layers, model.bidirectional, device,
            )
            return cls(model, classes, device)
        except Exception as exc:
            log.warning("Failed to load Ocarina GRU: %s", exc)
            return None

    @torch.no_grad()
    def predict(self, sequence: list[list[float]]) -> tuple[str, float] | None:
        """sequence : list de frames, chacune un vecteur 42-float [x0,y0,...,x20,y20].

        Côté serveur, la deque `state['buf']` ne pousse une frame QUE quand
        une main est détectée → toutes les SEQ_LEN frames sont "réelles",
        donc le mask est all-ones. Si jamais on injecte du padding plus tard
        (queue plus longue, etc.), le calcul de mask devra être adapté.
        """
        if len(sequence) != self.SEQ_LEN:
            return None
        try:
            x = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        except (ValueError, TypeError):
            return None
        if x.shape != (self.SEQ_LEN, self.INPUT_SIZE):
            return None
        x = x.unsqueeze(0)                                 # [1, T, 42]
        mask = torch.ones(1, self.SEQ_LEN, device=self.device, dtype=torch.float32)
        logits = self.model(x, mask=mask)[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        return self.classes[idx], float(probs[idx].item())
