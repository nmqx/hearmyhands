"""InferenceEngine — pure-Python entry point for all model inference.

Used directly in-process by `hearmyhands/app.py` (fast path, no HTTP) and
wrapped behind HTTP endpoints by `HmH/api.py` (backwards-compatible service).
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "heatnoks"))
sys.path.append(SCRIPT_DIR)

from model import HeatnoksModel  # noqa: E402
from letter_classifier import LetterClassifier  # noqa: E402
from sign_classifier import SignClassifier  # noqa: E402

INPUT_SIZE    = 256
VIS_THRESHOLD = 0.3

CKPT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "heatnoks", "checkpoints", "best.pt"),
    os.path.join(SCRIPT_DIR, "heatnoks", "checkpoints_pretrain", "best.pt"),
]
HAND_TASK_PATH = os.path.join(SCRIPT_DIR, "heatnoks", "hand_landmarker.task")
POIDS_DIR      = os.path.join(SCRIPT_DIR, "Poids")

OCARINA_DIR     = os.path.join(SCRIPT_DIR, "..", "Modèle_Ocarina")
OCARINA_WEIGHTS = os.environ.get("OCARINA_WEIGHTS", os.path.join(OCARINA_DIR, "ocarina_gru_v1.pth"))
OCARINA_CLASSES = os.environ.get("OCARINA_CLASSES", os.path.join(OCARINA_DIR, "ocarina_classes.json"))

_MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_STD_BGR  = np.array([0.225, 0.224, 0.229], dtype=np.float32)

log = logging.getLogger("hmh.inference")


def _load_body_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt_path is None:
        log.warning("No checkpoint found. Tried: %s", CKPT_CANDIDATES)
        return None, device
    model = HeatnoksModel(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Body model loaded from %s on %s", ckpt_path, device)
    return model, device


def _load_hand_detector():
    if not os.path.exists(HAND_TASK_PATH):
        log.warning("Hand landmarker task not found at %s", HAND_TASK_PATH)
        return None
    base_options = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return mp_vision.HandLandmarker.create_from_options(options)


class InferenceEngine:
    """Loads all models once, provides thread-safe inference methods."""

    def __init__(self):
        self.model, self.device = _load_body_model()
        self.hands_detector    = _load_hand_detector()
        self.letter_classifier = LetterClassifier.try_load(POIDS_DIR)
        self.sign_classifier   = SignClassifier.try_load(OCARINA_WEIGHTS, OCARINA_CLASSES)
        # fp16 was slower than fp32 on small batches (RTX 3060 Ti) — opt-in
        self.use_amp = os.environ.get("USE_AMP", "0") == "1" and self.device.type == "cuda"
        # Single worker for MediaPipe → safe to reuse detector across requests
        self._hand_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")
        # Pre-build mean/std on the right device
        self._mean_t = torch.tensor(_MEAN_BGR[::-1].copy(), device=self.device).view(1, 3, 1, 1)
        self._std_t  = torch.tensor(_STD_BGR[::-1].copy(),  device=self.device).view(1, 3, 1, 1)

    # ── Public API ───────────────────────────────────────────────────────────
    def health(self) -> dict:
        return {
            "model_loaded":      self.model is not None,
            "hands_detector":    self.hands_detector is not None,
            "letter_classifier": self.letter_classifier is not None,
            "sign_classifier":   self.sign_classifier is not None,
            "device":            str(self.device),
            "use_amp":           self.use_amp,
        }

    def predict_frame(self, image_bytes: bytes) -> dict | None:
        """Decode JPEG bytes → run body model + MediaPipe (parallel) → result dict."""
        if self.model is None:
            return None
        frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None

        # Kick off hand detection in parallel with the GPU inference
        hands_future = self._hand_executor.submit(self._detect_hands, frame_bgr)

        tensor, side, pad_left, pad_top = self._preprocess(frame_bgr)
        with torch.inference_mode():
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = self.model.predict(tensor)
            else:
                out = self.model.predict(tensor)
        kp_orig = out[0].float().cpu().numpy().copy()
        kp_orig[:, 0] = kp_orig[:, 0] * side - pad_left
        kp_orig[:, 1] = kp_orig[:, 1] * side - pad_top

        hands_data = hands_future.result()
        letter, confidence = self._predict_letter(hands_data, frame_bgr)
        img_h, img_w = frame_bgr.shape[:2]

        return {
            "keypoints":    kp_orig.tolist(),
            "hands":        hands_data,
            "letter":       letter,
            "confidence":   confidence,
            "image_width":  int(img_w),
            "image_height": int(img_h),
        }

    def predict_sign(self, sequence) -> dict | None:
        """Run the Ocarina GRU on a 60-frame sequence of 42 normalized floats."""
        if self.sign_classifier is None:
            return None
        if not isinstance(sequence, list):
            return None
        result = self.sign_classifier.predict(sequence)
        if result is None:
            return None
        sign, conf = result
        return {"sign": sign, "confidence": conf}

    # ── Internals ────────────────────────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray):
        h, w, _ = frame_bgr.shape
        side = max(w, h)
        pad_left = (side - w) // 2
        pad_top  = (side - h) // 2
        padded = cv2.copyMakeBorder(
            frame_bgr,
            pad_top, side - h - pad_top, pad_left, side - w - pad_left,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        resized = cv2.resize(padded, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).float().mul_(1.0 / 255.0).unsqueeze_(0)
        tensor = (tensor - self._mean_t) / self._std_t
        return tensor, side, pad_left, pad_top

    def _detect_hands(self, frame_bgr: np.ndarray) -> list[list[list[float]]]:
        if self.hands_detector is None:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hands_detector.detect(mp_image)
        if not results.hand_landmarks:
            return []
        h, w, _ = frame_bgr.shape
        return [[[lm.x * w, lm.y * h] for lm in hand] for hand in results.hand_landmarks]

    def _predict_letter(self, hands_data, frame_bgr):
        if not hands_data or self.letter_classifier is None or frame_bgr is None:
            return None, None
        h_img, w_img, _ = frame_bgr.shape
        if h_img == 0 or w_img == 0:
            return None, None
        features = [
            coord
            for pt in hands_data[0]
            for coord in (pt[0] / w_img, pt[1] / h_img)
        ]
        result = self.letter_classifier.predict(features)
        return (None, None) if result is None else result
