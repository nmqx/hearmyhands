"""HearMyHands web app — Flask + Socket.IO with binary frame transport.

Per Socket.IO connection we keep a rolling buffer of the last N frames of
normalized hand landmarks; once full, we ask the temporal sign classifier
(Ocarina GRU) for a sign prediction every few frames.

Inference is in-process by default (single Python process). Set
USE_HTTP_MODEL=1 to fall back to calling the standalone HmH/api.py service
over HTTP (useful when the model lives on a separate machine).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from threading import Lock

from flask import Flask, render_template, request
from flask_socketio import SocketIO

try:
    import psutil  # type: ignore
    psutil.cpu_percent(interval=None)                  # baseline (premier appel = 0)
    psutil.cpu_percent(interval=None, percpu=True)     # baseline per-core
    _proc = psutil.Process()
    _proc.cpu_percent(interval=None)
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    _proc = None

# Make HmH/ importable regardless of where this is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "..", "HmH"))

USE_HTTP_MODEL  = os.environ.get("USE_HTTP_MODEL", "0") == "1"
MODEL_API_URL   = os.environ.get("MODEL_API_URL", "http://127.0.0.1:5001/model_predict")
SIGN_API_URL    = os.environ.get("SIGN_API_URL",  "http://127.0.0.1:5001/sign_predict")
REQUEST_TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "5"))
SIGN_TIMEOUT    = float(os.environ.get("SIGN_TIMEOUT",  "2"))
MAX_FRAME_BYTES = 2 * 1024 * 1024
SEQ_LEN         = 45          # doit matcher SignClassifier.SEQ_LEN
SIGN_EVERY_N    = 5

# Espace pixel canonique sur lequel le GRU Ocarina a été entraîné
# (mod_json.py force la webcam en 640x480 lors de la capture du dataset)
TRAIN_W, TRAIN_H = 640, 480

app = Flask(__name__)
# async_mode auto: utilise eventlet/gevent en prod (gunicorn) sinon threading (dev werkzeug)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=os.environ.get("SOCKETIO_ASYNC_MODE") or None,
    max_http_buffer_size=MAX_FRAME_BYTES,
)

_log = logging.getLogger("hmh.web")
_EMPTY = {
    "skeleton": None, "hands": [],
    "letter": None, "confidence": None,
    "sign": None, "sign_confidence": None,
}

# Per-connection rolling buffer of normalized hand landmarks + tick counter.
_sessions: dict[str, dict] = {}
_sessions_lock = Lock()

# ── Inference backend ────────────────────────────────────────────────────────
# Default: in-process. Single shared engine, loaded once.
_engine = None
_http   = None
_sign_api_disabled = False

if USE_HTTP_MODEL:
    import requests  # type: ignore
    _http = requests.Session()
    _log.info("Inference backend: HTTP (%s)", MODEL_API_URL)
else:
    from inference import InferenceEngine  # type: ignore
    _engine = InferenceEngine()
    _log.info("Inference backend: in-process (%s)", _engine.health())


def _run_frame(image_bytes: bytes):
    if _engine is not None:
        return _engine.predict_frame(image_bytes)
    try:
        resp = _http.post(
            MODEL_API_URL, data=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as exc:
        _log.warning("model API unreachable: %s", exc)
        return None
    if resp.status_code != 200:
        _log.warning("model API status %s", resp.status_code)
        return None
    return resp.json()


def _run_sign(sequence):
    global _sign_api_disabled
    if _sign_api_disabled:
        return None
    if _engine is not None:
        if _engine.sign_classifier is None:
            _sign_api_disabled = True
            _log.info("Sign classifier not loaded — disabling sign predictions")
            return None
        return _engine.predict_sign(sequence)
    try:
        r = _http.post(SIGN_API_URL, json={"sequence": sequence}, timeout=SIGN_TIMEOUT)
    except Exception as exc:
        _log.warning("sign API unreachable: %s", exc)
        return None
    if r.status_code == 503:
        _sign_api_disabled = True
        _log.info("Sign classifier not available — disabling sign predictions")
        return None
    if r.status_code != 200:
        return None
    return r.json()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/translate")
def translate():
    return render_template("translate.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/learn/cards")
def learn_cards():
    return render_template("learn_cards.html")


@app.route("/learn/library")
def learn_library():
    return render_template("learn_library.html")


@app.route("/videotest")
def videotest():
    return render_template("videotest.html")


@app.route("/api/video/<letter>")
def api_video(letter):
    """CDN local pour les vidéos d'apprentissage.

    Sert le fichier .mp4 en bypassant le système static de Flask et le cache
    Cloudflare. Appelé depuis /learn/play/<letter> (wrapper HTML qui ajoute
    autoplay+loop).
    """
    from flask import send_from_directory, abort
    letter = letter.upper()
    if not (len(letter) == 1 and 'A' <= letter <= 'Z'):
        abort(404)
    video_dir = os.path.join(_HERE, "static", "learn")
    if not os.path.exists(os.path.join(video_dir, f"{letter}.mp4")):
        abort(404)
    resp = send_from_directory(video_dir, f"{letter}.mp4", mimetype="video/mp4")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/learn/play/<letter>")
def learn_play(letter):
    """Mini-page wrapper qui joue la vidéo en boucle dans un <video>.

    On l'utilise comme src d'un <iframe> côté /learn/cards : le navigateur
    rend la vidéo dans son contexte propre (qui marche, contrairement au
    <video> embedded dans la page principale qui restait noir).
    """
    from flask import abort, Response
    letter = letter.upper()
    if not (len(letter) == 1 and 'A' <= letter <= 'Z'):
        abort(404)
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;background:#000;height:100%;overflow:hidden}"
        "video{width:100%;height:100%;object-fit:cover;display:block}</style>"
        "</head><body>"
        f"<video src='/api/video/{letter}' autoplay loop muted playsinline controls></video>"
        "</body></html>"
    )
    return Response(html, mimetype="text/html")


@app.route("/healthz")
def healthz():
    if _engine is not None:
        return _engine.health()
    return {"backend": "http", "model_api": MODEL_API_URL}


@app.route("/monitor")
def monitor():
    return render_template("monitor.html")


@app.route("/stats")
def stats():
    if not _PSUTIL:
        return {"error": "psutil not installed"}, 503
    mem  = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    load = os.getloadavg()
    return {
        "ts":            time.time(),
        "cpu_total":     psutil.cpu_percent(interval=None),
        "cpu_per_core":  psutil.cpu_percent(interval=None, percpu=True),
        "cpu_count":     psutil.cpu_count(),
        "mem_total":     mem.total,
        "mem_used":      mem.used,
        "mem_percent":   mem.percent,
        "disk_total":    disk.total,
        "disk_used":     disk.used,
        "disk_percent":  disk.percent,
        "load_1":        load[0],
        "load_5":        load[1],
        "load_15":       load[2],
        "uptime":        time.time() - psutil.boot_time(),
        "app_rss":       _proc.memory_info().rss,
        "app_cpu":       _proc.cpu_percent(interval=None),
        "app_threads":   _proc.num_threads(),
    }


# ── Socket.IO ────────────────────────────────────────────────────────────────
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
    """Decode + predict + ack."""
    if not image_bytes:
        return _EMPTY
    data = _run_frame(image_bytes)
    if data is None:
        return _EMPTY

    hands  = data.get("hands", []) or []
    img_w  = data.get("image_width", 0) or 0
    img_h  = data.get("image_height", 0) or 0

    sign, sign_conf = _maybe_predict_sign(request.sid, hands, img_w, img_h)

    return {
        "skeleton":        data.get("keypoints"),
        "hands":           hands,
        "letter":          data.get("letter"),
        "confidence":      data.get("confidence"),
        "sign":            sign,
        "sign_confidence": sign_conf,
    }


def _maybe_predict_sign(sid, hands, img_w, img_h):
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

    result = _run_sign(sequence)
    if result is None:
        return None, None
    return result.get("sign"), result.get("confidence")


def _normalize_hand(hand, img_w, img_h):
    """Pré-traitement à l'identique de Modèle_Ocarina/Dataset.py:

    - rescale les landmarks depuis l'image runtime vers l'espace pixel
      d'entraînement (640x480)
    - centre tous les points sur le poignet (point 0) — invariance en
      translation
    Retourne une liste plate de 42 floats [x0, y0, x1, y1, …].
    """
    if not hand or not img_w or not img_h:
        return [0.0] * 42
    sx = TRAIN_W / img_w
    sy = TRAIN_H / img_h
    scaled = [(pt[0] * sx, pt[1] * sy) for pt in hand]
    wx, wy = scaled[0]
    out = []
    for x, y in scaled:
        out.append(x - wx)
        out.append(y - wy)
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
