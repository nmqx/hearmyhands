"""Model API — receives raw JPEG bytes, returns keypoints + hand landmarks."""
from __future__ import annotations

import io
import logging
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms.functional as TF
from flask import Flask, jsonify, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "heatnoks"))

from model import HeatnoksModel  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
INPUT_SIZE    = 256
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]
VIS_THRESHOLD = 0.3

CKPT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "heatnoks", "checkpoints", "best.pt"),
    os.path.join(SCRIPT_DIR, "heatnoks", "checkpoints_pretrain", "best.pt"),
]
HAND_TASK_PATH = os.path.join(SCRIPT_DIR, "heatnoks", "hand_landmarker.task")

log = logging.getLogger("hmh.model")


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


# ── Image processing ─────────────────────────────────────────────────────────
def preprocess(img: Image.Image):
    """Pad to square + resize + normalize. Returns tensor and unpad params."""
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top  = (side - h) // 2
    img_padded  = TF.pad(img, (pad_left, pad_top, side - w - pad_left, side - h - pad_top), fill=0)
    img_resized = TF.resize(img_padded, [INPUT_SIZE, INPUT_SIZE])
    tensor = TF.normalize(TF.to_tensor(img_resized), mean=MEAN, std=STD)
    return tensor.unsqueeze(0), side, pad_left, pad_top


def detect_hands(frame_bgr: np.ndarray, kp_orig: np.ndarray) -> list[list[list[float]]]:
    """Crop around each wrist (or projected forearm) and run MediaPipe."""
    if hands_detector is None:
        return []

    s1x, s1y, _ = kp_orig[1]
    s2x, s2y, _ = kp_orig[2]
    sh_dist = float(np.hypot(s1x - s2x, s1y - s2y))
    h, w, _ = frame_bgr.shape
    out: list[list[list[float]]] = []

    for elbow_idx, wrist_idx, shoulder_idx in [(3, 5, 1), (4, 6, 2)]:
        wx, wy, wv = kp_orig[wrist_idx]
        ex, ey, ev = kp_orig[elbow_idx]
        sx, sy, sv = kp_orig[shoulder_idx]

        if wv <= VIS_THRESHOLD and not (ev > VIS_THRESHOLD and sv > VIS_THRESHOLD):
            continue

        search_x, search_y = wx, wy
        dist_fw = float(np.hypot(wx - ex, wy - ey)) if wv > VIS_THRESHOLD else 0.0

        if ev > VIS_THRESHOLD and sv > VIS_THRESHOLD:
            dist_ua = float(np.hypot(ex - sx, ey - sy))
            if wv <= VIS_THRESHOLD or dist_fw < 0.4 * dist_ua:
                # Project forearm beyond the elbow when the wrist is unreliable
                unit_dx = (ex - sx) / (dist_ua + 1e-6)
                unit_dy = (ey - sy) / (dist_ua + 1e-6)
                search_x = ex + unit_dx * dist_ua * 0.9
                search_y = ey + unit_dy * dist_ua * 0.9
                dist_fw  = dist_ua * 0.9

        box_size = int(max(sh_dist * 0.6, dist_fw * 1.6, 160))
        half = box_size // 2
        x1, y1 = max(0, int(search_x - half)), max(0, int(search_y - half))
        x2, y2 = min(w, int(search_x + half)), min(h, int(search_y + half))
        if x2 - x1 <= 20 or y2 - y1 <= 20:
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        results = hands_detector.detect(mp_image)
        if not results.hand_landmarks:
            continue

        ch, cw, _ = crop.shape
        for hand_landmarks in results.hand_landmarks:
            out.append([[lm.x * cw + x1, lm.y * ch + y1] for lm in hand_landmarks])

    return out


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/model_predict", methods=["POST"])
def model_predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return jsonify({"error": "empty body"}), 400

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        return jsonify({"error": f"invalid image: {exc}"}), 400

    tensor, side, pad_left, pad_top = preprocess(img)
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model.predict(tensor)

    kp_orig = out[0].cpu().numpy().copy()
    kp_orig[:, 0] = kp_orig[:, 0] * side - pad_left
    kp_orig[:, 1] = kp_orig[:, 1] * side - pad_top

    hands_data = detect_hands(frame_bgr, kp_orig) if frame_bgr is not None else []

    return jsonify({"keypoints": kp_orig.tolist(), "hands": hands_data})


@app.route("/healthz")
def healthz():
    return jsonify({
        "model_loaded": model is not None,
        "hands_detector": hands_detector is not None,
        "device": str(device),
    })


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False)
