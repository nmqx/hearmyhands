"""Sequential sign classifier (Ocarina GRU): 60 frames × 42 features -> 1 class.

The architecture mirrors `Modèle_Ocarina/Ocarina_GRU.py` so that state_dicts
saved during training load directly here without depending on the accented
source directory at runtime.
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
    def __init__(self, input_size: int = 42, hidden_size: int = 128,
                 num_layers: int = 2, num_classes: int = 26):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])


class SignClassifier:
    # Doit rester aligné avec MAX_FRAMES de Modèle_Ocarina/Train.py
    SEQ_LEN = 45
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
            log.info("Ocarina GRU loaded from %s (%d classes) on %s",
                     weights_path, len(classes), device)
            return cls(model, classes, device)
        except Exception as exc:
            log.warning("Failed to load Ocarina GRU: %s", exc)
            return None

    @torch.no_grad()
    def predict(self, sequence: list[list[float]]) -> tuple[str, float] | None:
        """sequence: list of frames, each a flat 42-float [x0,y0,…,x20,y20]."""
        if len(sequence) != self.SEQ_LEN:
            return None
        try:
            x = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        except (ValueError, TypeError):
            return None
        if x.shape != (self.SEQ_LEN, self.INPUT_SIZE):
            return None
        x = x.unsqueeze(0)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        return self.classes[idx], float(probs[idx].item())
