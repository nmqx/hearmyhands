"""HearMyHands web app — Flask + Socket.IO with binary frame transport."""
from __future__ import annotations

import logging
import os

import requests
from flask import Flask, render_template
from flask_socketio import SocketIO

MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:5001/model_predict")
REQUEST_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "5"))
MAX_FRAME_BYTES = 2 * 1024 * 1024  # 2 MB cap per frame

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=MAX_FRAME_BYTES,
)

_http = requests.Session()
_log = logging.getLogger("hmh.web")
_EMPTY = {"skeleton": None, "hands": []}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/translate")
def translate():
    return render_template("translate.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@socketio.on("frame")
def handle_frame(image_bytes):
    """Receive raw JPEG bytes and return the model prediction via Socket.IO ack."""
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

    data = resp.json()
    return {"skeleton": data.get("keypoints"), "hands": data.get("hands", [])}


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
    )
