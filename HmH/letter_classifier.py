"""Tiny MLP that maps 21 hand landmarks (x, y) to one of 26 letters.

Architecture: 42 -> 50 (ReLU) -> 26 (softmax). Weights are loaded from
JSON files exported during training (W1, W2, b1, b2). Le passage à ReLU
+ softmax et à l'alphabet complet date de l'export Poids_centered de
mai 2026 (cf. Modèle_MLP/predict.py).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

# Alphabet complet : la nouvelle version du MLP couvre J, X, Z (même si
# en LSF J/Z restent des signes dynamiques qu'on prédira plutôt via le
# GRU Ocarina — ici on garde le résultat brut et le filtrage se fait
# côté client via le seuil de confiance).
ALPHABET = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]

INPUT_SIZE  = 42  # 21 landmarks * (x, y)
OUTPUT_SIZE = len(ALPHABET)

log = logging.getLogger("hmh.letters")


def _load_json(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        return np.asarray(json.load(f), dtype=float)


def _softmax(z: np.ndarray) -> np.ndarray:
    # Softmax numériquement stable (substraction du max avant exp).
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class LetterClassifier:
    def __init__(self, weights_dir: str):
        self.W1 = _load_json(os.path.join(weights_dir, "W1.json"))
        self.W2 = _load_json(os.path.join(weights_dir, "W2.json"))
        self.b1 = _load_json(os.path.join(weights_dir, "b1.json"))
        self.b2 = _load_json(os.path.join(weights_dir, "b2.json"))
        # Validation des shapes — évite des comportements silencieusement
        # faux si on switch d'archi sans repush ce fichier.
        if self.W2.shape[1] != OUTPUT_SIZE:
            raise ValueError(
                f"W2 a {self.W2.shape[1]} sorties, attendu {OUTPUT_SIZE}. "
                "Vérifie l'alphabet et la dernière couche du MLP."
            )

    @classmethod
    def try_load(cls, weights_dir: str) -> "LetterClassifier | None":
        try:
            inst = cls(weights_dir)
            log.info("Letter classifier loaded from %s (hidden=%d, classes=%d)",
                     weights_dir, inst.W1.shape[1], inst.W2.shape[1])
            return inst
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.warning("Letter classifier disabled: %s", exc)
            return None

    def predict(self, features: list[float]) -> tuple[str, float] | None:
        """Return (letter, confidence) for 42 floats or None on bad input.

        Les features sont les landmarks recentrés sur le poignet (point 0),
        cohérent avec le pré-traitement côté serveur (_normalize_hand)
        et avec le script d'entraînement du nouveau MLP.
        """
        if len(features) != INPUT_SIZE:
            return None
        x   = np.asarray(features, dtype=float).reshape(1, -1)
        h   = np.maximum(0.0, x @ self.W1 + self.b1)      # ReLU
        out = _softmax(h @ self.W2 + self.b2)[0]          # Softmax
        idx = int(np.argmax(out))
        return ALPHABET[idx], float(out[idx])
