"""Tiny MLP that maps 21 hand landmarks (x, y) to one of 23 letters.

Architecture: 42 -> 30 (sigmoid) -> 23 (sigmoid). Weights are loaded from
JSON files exported during training (W1, W2, b1, b2).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

# J and Z are excluded because they require motion in LSF/ASL fingerspelling.
ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

INPUT_SIZE  = 42  # 21 landmarks * (x, y)
OUTPUT_SIZE = len(ALPHABET)

log = logging.getLogger("hmh.letters")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        return np.asarray(json.load(f), dtype=float)


class LetterClassifier:
    def __init__(self, weights_dir: str):
        self.W1 = _load_json(os.path.join(weights_dir, "W1.json"))
        self.W2 = _load_json(os.path.join(weights_dir, "W2.json"))
        self.b1 = _load_json(os.path.join(weights_dir, "b1.json"))
        self.b2 = _load_json(os.path.join(weights_dir, "b2.json"))

    @classmethod
    def try_load(cls, weights_dir: str) -> "LetterClassifier | None":
        try:
            inst = cls(weights_dir)
            log.info("Letter classifier loaded from %s", weights_dir)
            return inst
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Letter classifier disabled: %s", exc)
            return None

    def predict(self, features: list[float]) -> tuple[str, float] | None:
        """Return (letter, confidence) for 42 floats in [0, 1] or None on bad input."""
        if len(features) != INPUT_SIZE:
            return None
        x   = np.asarray(features, dtype=float).reshape(1, -1)
        h   = _sigmoid(x @ self.W1 + self.b1)
        out = _sigmoid(h @ self.W2 + self.b2)[0]
        idx = int(np.argmax(out))
        return ALPHABET[idx], float(out[idx])
