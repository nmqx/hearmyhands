"""HearMyHands web app — Flask + Socket.IO with binary frame transport.

Per Socket.IO connection we keep a rolling buffer of the last N frames of
normalized hand landmarks; once full, we ask the temporal sign classifier
(Ocarina GRU) for a sign prediction every few frames.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from threading import Lock

import requests
from flask import Flask, render_template, request
from flask_socketio import SocketIO

MODEL_API_URL   = os.environ.get("MODEL_API_URL", "http://127.0.0.1:5001/model_predict")
SIGN_API_URL    = os.environ.get("SIGN_API_URL",  "http://127.0.0.1:5001/sign_predict")
REQUEST_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "5"))
SIGN_TIMEOUT    = float(os.environ.get("SIGN_TIMEOUT",  "2"))
MAX_FRAME_BYTES = 2 * 1024 * 1024  # 2 MB cap per frame
SEQ_LEN         = 60               # must match SignClassifier.SEQ_LEN
SIGN_EVERY_N    = 5                # call /sign_predict every N frames once buffer ready

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=MAX_FRAME_BYTES,
)

_http = requests.Session()
_log = logging.getLogger("hmh.web")
_EMPTY = {
    "skeleton": None, "hands": [],
    "letter": None, "confidence": None,
    "sign": None, "sign_confidence": None,
}

# Per-connection state: rolling buffer of normalized hand landmarks + frame counter.
_sessions: dict[str, dict] = {}
_sessions_lock = Lock()

# Disabled after the first 503 — saves the round-trip when no weights present.
_sign_api_disabled = False


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/translate")
def translate():
    return render_template("translate.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@socketio.on("connect")
def _on_connect():
    with _sessions_lock:
        _sessions[request.sid] = {"buf": deque(maxlen=SEQ_LEN), "tick": 0}


@socketio.on("disconnect")
def _on_disconnect():
    with _sessions_lock:
        _sessions.pop(request.sid, None)


@socketio.on("frame")
def handle_frame(image_bytes):
    """Receive raw JPEG bytes, run per-frame predictions, return via ack."""
    if not image_bytes:
        return _EMPTY
    try:
        resp = _http.post(
            MODEL_API_URL,
            data=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        _log.warning("model API unreachable: %s", exc)
        return _EMPTY

    if resp.status_code != 200:
        _log.warning("model API status %s", resp.status_code)
        return _EMPTY

    data   = resp.json()
    hands  = data.get("hands", []) or []
    img_w  = data.get("image_width", 0) or 0
    img_h  = data.get("image_height", 0) or 0

    sign, sign_conf = _maybe_predict_sign(request.sid, hands, img_w, img_h)

    return {
        "skeleton":         data.get("keypoints"),
        "hands":            hands,
        "letter":           data.get("letter"),
        "confidence":       data.get("confidence"),
        "sign":             sign,
        "sign_confidence":  sign_conf,
    }


def _maybe_predict_sign(sid, hands, img_w, img_h):
    """Append the first hand to the session buffer; call /sign_predict every N frames."""
    global _sign_api_disabled
    if _sign_api_disabled:
        return None, None
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return None, None
        if hands and img_w and img_h:
            state["buf"].append(_normalize_hand(hands[0], img_w, img_h))
        state["tick"] += 1
        ready = len(state["buf"]) == SEQ_LEN and state["tick"] % SIGN_EVERY_N == 0
        if not ready:
            return None, None
        sequence = list(state["buf"])

    try:
        r = _http.post(SIGN_API_URL, json={"sequence": sequence}, timeout=SIGN_TIMEOUT)
    except requests.RequestException as exc:
        _log.warning("sign API unreachable: %s", exc)
        return None, None
    if r.status_code == 503:
        _sign_api_disabled = True
        _log.info("Sign classifier not available — disabling sign predictions")
        return None, None
    if r.status_code != 200:
        return None, None
    d = r.json()
    return d.get("sign"), d.get("confidence")


def _normalize_hand(hand, img_w, img_h):
    """Flatten 21 (x, y) landmarks into 42 normalized floats."""
    out = []
    for pt in hand:
        out.append(pt[0] / img_w)
        out.append(pt[1] / img_h)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        allow_unsafe_werkzeug=True,
    )
