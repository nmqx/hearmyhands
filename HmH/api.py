"""Model API — receives raw JPEG bytes, returns keypoints + hand landmarks.

Hot path:
- single JPEG decode (cv2/libjpeg-turbo)
- numpy/cv2 preprocessing direct to GPU (no PIL)
- body keypoints (GPU) and MediaPipe hand detection (CPU) run **in parallel**
- single full-frame MediaPipe call instead of two cropped calls
- inference_mode (no autograd state)
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
from flask import Flask, jsonify, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "heatnoks"))
sys.path.append(SCRIPT_DIR)

from model import HeatnoksModel  # noqa: E402
from letter_classifier import LetterClassifier  # noqa: E402
from sign_classifier import SignClassifier  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
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

log = logging.getLogger("hmh.model")

# Mean/std baked into a tensor once (filled later when device is known)
_MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32)  # BGR order (cv2)
_STD_BGR  = np.array([0.225, 0.224, 0.229], dtype=np.float32)


# ── Model loading ────────────────────────────────────────────────────────────
def load_model() -> tuple[HeatnoksModel | None, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt_path is None:
        log.warning("No checkpoint found. Tried: %s", CKPT_CANDIDATES)
        return None, device

    model = HeatnoksModel(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Model loaded from %s on %s", ckpt_path, device)
    return model, device


def load_hand_detector() -> mp_vision.HandLandmarker | None:
    if not os.path.exists(HAND_TASK_PATH):
        log.warning("Hand landmarker task not found at %s", HAND_TASK_PATH)
        return None
    base_options = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return mp_vision.HandLandmarker.create_from_options(options)


app = Flask(__name__)
model, device = load_model()
hands_detector = load_hand_detector()
letter_classifier = LetterClassifier.try_load(POIDS_DIR)
sign_classifier  = SignClassifier.try_load(OCARINA_WEIGHTS, OCARINA_CLASSES)

# fp16 was measured slower than fp32 on small batches (RTX 3060 Ti, batch=1) —
# stay in fp32 unless the user explicitly opts in.
USE_AMP = os.environ.get("USE_AMP", "0") == "1" and device.type == "cuda"
_MEAN_T = torch.tensor(_MEAN_BGR[::-1].copy(), device=device).view(1, 3, 1, 1)  # RGB on device
_STD_T  = torch.tensor(_STD_BGR[::-1].copy(),  device=device).view(1, 3, 1, 1)

# Single worker thread for MediaPipe so we can run it concurrently with the
# GPU inference. MediaPipe Tasks is C++ and releases the GIL, but reusing a
# single detector instance across threads needs serialization — one worker
# guarantees that.
_hand_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hands")


# ── Image processing ─────────────────────────────────────────────────────────
def preprocess_bgr(frame_bgr: np.ndarray):
    """Pad-to-square (cv2) → resize → BGR→RGB → normalize → GPU tensor. No PIL."""
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
    # uint8 → float32 [0, 1] → CHW → GPU → normalize
    tensor = torch.from_numpy(rgb).to(device, non_blocking=True)
    tensor = tensor.permute(2, 0, 1).float().mul_(1.0 / 255.0).unsqueeze_(0)
    tensor = (tensor - _MEAN_T) / _STD_T
    return tensor, side, pad_left, pad_top


def detect_hands_fullframe(frame_bgr: np.ndarray) -> list[list[list[float]]]:
    """Single MediaPipe call on the full frame. Returns list of [[x,y], …] hands."""
    if hands_detector is None:
        return []
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hands_detector.detect(mp_image)
    if not results.hand_landmarks:
        return []
    h, w, _ = frame_bgr.shape
    return [[[lm.x * w, lm.y * h] for lm in hand] for hand in results.hand_landmarks]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/model_predict", methods=["POST"])
def model_predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return jsonify({"error": "empty body"}), 400

    # Single JPEG decode via cv2/libjpeg-turbo
    frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "invalid image"}), 400

    # Kick off MediaPipe hand detection on CPU in parallel with GPU inference
    hands_future = _hand_executor.submit(detect_hands_fullframe, frame_bgr)

    tensor, side, pad_left, pad_top = preprocess_bgr(frame_bgr)
    with torch.inference_mode():
        if USE_AMP:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model.predict(tensor)
        else:
            out = model.predict(tensor)
    kp_orig = out[0].float().cpu().numpy().copy()
    kp_orig[:, 0] = kp_orig[:, 0] * side - pad_left
    kp_orig[:, 1] = kp_orig[:, 1] * side - pad_top

    hands_data = hands_future.result()
    letter, confidence = predict_letter(hands_data, frame_bgr)
    img_h, img_w = frame_bgr.shape[:2]

    return jsonify({
        "keypoints":    kp_orig.tolist(),
        "hands":        hands_data,
        "letter":       letter,
        "confidence":   confidence,
        "image_width":  int(img_w),
        "image_height": int(img_h),
    })


def predict_letter(hands_data, frame_bgr):
    """Run the letter MLP on the first detected hand."""
    if not hands_data or letter_classifier is None or frame_bgr is None:
        return None, None
    h_img, w_img, _ = frame_bgr.shape
    if h_img == 0 or w_img == 0:
        return None, None
    features = [
        coord
        for pt in hands_data[0]
        for coord in (pt[0] / w_img, pt[1] / h_img)
    ]
    result = letter_classifier.predict(features)
    return (None, None) if result is None else result


@app.route("/sign_predict", methods=["POST"])
def sign_predict():
    """Run the Ocarina GRU on a 60-frame sequence of normalized hand landmarks."""
    if sign_classifier is None:
        return jsonify({"error": "sign classifier not loaded"}), 503
    payload = request.get_json(silent=True) or {}
    seq = payload.get("sequence")
    if not isinstance(seq, list):
        return jsonify({"error": "missing 'sequence' (list of frames)"}), 400
    result = sign_classifier.predict(seq)
    if result is None:
        return jsonify({"error": "bad sequence shape (need 60 × 42)"}), 400
    sign, conf = result
    return jsonify({"sign": sign, "confidence": conf})


@app.route("/healthz")
def healthz():
    return jsonify({
        "model_loaded":      model is not None,
        "hands_detector":    hands_detector is not None,
        "letter_classifier": letter_classifier is not None,
        "sign_classifier":   sign_classifier is not None,
        "device":            str(device),
        "use_amp":           USE_AMP,
    })


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False, threaded=True)
