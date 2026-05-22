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
import re
import sqlite3
import sys
import time
from collections import deque
from threading import Lock

from flask import Flask, jsonify, render_template, request
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
# Un appel GRU tous les SIGN_EVERY_N frames quand le client demande
# explicitement (predict_sign=True). Aligné avec PRED_EVERY_N_FRAMES=3 de
# demo.py pour avoir une UX réactive (~10 Hz à 30 fps de webcam).
SIGN_EVERY_N    = 3
# Nombre minimum de frames AVEC main détectée avant de tenter une prédiction.
# Sinon le GRU tournerait sur un buffer quasi-vide -> bruit pur.
MIN_REAL_FRAMES = 5

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

# Cache de la dernière frame brute reçue par session, pour la vue debug
# /webcam. On garde uniquement les octets JPEG, qui suffisent au rendu :
# pas de décodage côté serveur, juste un passe-plat avec base64.
_last_frames: dict[str, tuple[float, bytes]] = {}
_last_frames_lock = Lock()
_LAST_FRAME_MAX_AGE_S = 10.0      # une session inactive disparaît au bout de 10s

# IP du client par sid, alimentée au connect Socket.IO. Affichée dans
# /webcam à la place du SID hash pour identifier visuellement qui est qui
# en démo (utile pour aller dire « ouais c'est ton flux, est-ce que ça
# marche pour toi ? »).
_session_ips: dict[str, str] = {}
_session_ips_lock = Lock()


def _client_ip():
    """Vraie IP du client derrière nginx + Cloudflare.

    Ordre de préférence :
      1. CF-Connecting-IP (Cloudflare donne l'IP d'origine ici)
      2. X-Forwarded-For (premier maillon = client)
      3. request.remote_addr (fallback, sera l'IP locale si derrière proxy)
    """
    ip = request.headers.get("CF-Connecting-IP")
    if ip:
        return ip
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"

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


def _run_sign(sequence, mask=None):
    global _sign_api_disabled
    if _sign_api_disabled:
        return None
    if _engine is not None:
        if _engine.sign_classifier is None:
            _sign_api_disabled = True
            _log.info("Sign classifier not loaded — disabling sign predictions")
            return None
        return _engine.predict_sign(sequence, mask)
    try:
        payload = {"sequence": sequence}
        if mask is not None:
            payload["mask"] = mask
        r = _http.post(SIGN_API_URL, json=payload, timeout=SIGN_TIMEOUT)
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


# ── QR code vers le site (page de démo) ─────────────────────────────────────
# Généré une seule fois au démarrage, gardé en mémoire comme PNG base64.
# URL cible configurable via env, défaut = https://hearmyhands.asia
QR_TARGET_URL = os.environ.get("QR_TARGET_URL", "https://hearmyhands.asia")
_QR_PNG_B64 = None  # populated lazily on first /qr hit


def _build_qr_png_b64():
    """Génère le PNG du QR code et le retourne en base64. Lazy — ne tente
    l'import qu'à la première demande pour ne pas casser le boot si
    qrcode n'est pas installé."""
    import base64, io
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        # box_size = taille d'un module en pixels dans le PNG. 24 -> ~720 px de
        # côté pour une URL courte, largement assez pour un zoom plein écran
        # sur projecteur de démo.
        box_size=24, border=2,
    )
    qr.add_data(QR_TARGET_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1c23", back_color="#ffffff")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route("/qr")
def qr_page():
    global _QR_PNG_B64
    if _QR_PNG_B64 is None:
        _QR_PNG_B64 = _build_qr_png_b64()
    if _QR_PNG_B64 is None:
        return ("Erreur : la lib 'qrcode' n'est pas installée sur le serveur. "
                "Installe via `pip install qrcode[pil]`."), 503
    return render_template("qr.html", qr_png=_QR_PNG_B64, target=QR_TARGET_URL)


@app.route("/translate")
def translate():
    return render_template("translate.html")


@app.route("/learn")
def learn():
    return render_template("learn.html")


@app.route("/learn/cards")
@app.route("/learn/cards/<letter>")
def learn_cards(letter=None):
    # La lettre dans l'URL est lue côté JS (window.location). Côté serveur
    # on rend juste le même template — le client gère le routing.
    return render_template("learn_cards.html")


@app.route("/learn/library")
def learn_library():
    return render_template("learn_library.html")


def _no_store(resp):
    # Force le browser à refetch les pages quiz (HTML) à chaque visite, sinon
    # une vieille version cachée référence un quiz.js obsolète et on n'a plus
    # les fix récents.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/learn/quiz")
def learn_quiz():
    from flask import make_response
    return _no_store(make_response(render_template("learn_quiz.html")))


@app.route("/learn/quiz/<mode>")
def learn_quiz_game(mode):
    from flask import abort, make_response
    if mode not in ("hardcore", "10sec", "survival"):
        abort(404)
    return _no_store(make_response(render_template("learn_quiz_game.html", mode=mode)))


# ── Leaderboard quiz (SQLite local) ─────────────────────────────────────
QUIZ_DB_PATH = os.environ.get("QUIZ_DB_PATH", os.path.join(_HERE, "quiz_scores.db"))
QUIZ_MODES   = ("hardcore", "10sec", "survival")
# Borne supérieure raisonnable selon le mode (anti-abus naïf)
QUIZ_MAX_SCORE = {"hardcore": 10, "10sec": 10, "survival": 9999}
_PSEUDO_RE = re.compile(r"[^\w\-. ]+", re.UNICODE)


def _quiz_db():
    """Connexion SQLite par appel (thread-safe naturellement). WAL pour
    de la concurrence raisonnable même avec gevent/eventlet en prod."""
    conn = sqlite3.connect(QUIZ_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _quiz_db_init():
    with _quiz_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                mode       TEXT      NOT NULL,
                pseudo     TEXT      NOT NULL,
                score      INTEGER   NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mode_score "
                     "ON quiz_scores(mode, score DESC, created_at ASC)")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass


_quiz_db_init()


def _sanitize_pseudo(raw: str) -> str:
    """Garde lettres/chiffres/tirets/points/espace, max 20 chars."""
    if not raw:
        return ""
    cleaned = _PSEUDO_RE.sub("", str(raw)).strip()
    return cleaned[:20] or "Anonyme"


@app.route("/api/quiz/leaderboard/<mode>")
def api_quiz_leaderboard(mode):
    from flask import abort
    if mode not in QUIZ_MODES:
        abort(404)
    limit = request.args.get("limit", default=10, type=int)
    limit = max(1, min(50, limit))
    try:
        with _quiz_db() as conn:
            rows = conn.execute(
                "SELECT pseudo, score, created_at FROM quiz_scores "
                "WHERE mode = ? ORDER BY score DESC, created_at ASC LIMIT ?",
                (mode, limit),
            ).fetchall()
    except sqlite3.Error as exc:
        _log.warning("quiz leaderboard read failed: %s", exc)
        return jsonify({"error": "db"}), 503
    return jsonify({
        "mode": mode,
        "entries": [{"pseudo": r["pseudo"], "score": r["score"],
                     "ts": r["created_at"]} for r in rows],
    })


@app.route("/api/quiz/score", methods=["POST"])
def api_quiz_submit():
    data = request.get_json(silent=True) or {}
    mode   = data.get("mode")
    pseudo = _sanitize_pseudo(data.get("pseudo", ""))
    score_raw = data.get("score")
    if mode not in QUIZ_MODES:
        return jsonify({"error": "invalid mode"}), 400
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid score"}), 400
    if score < 0 or score > QUIZ_MAX_SCORE.get(mode, 9999):
        return jsonify({"error": "score out of range"}), 400
    try:
        with _quiz_db() as conn:
            cur = conn.execute(
                "INSERT INTO quiz_scores (mode, pseudo, score) VALUES (?, ?, ?)",
                (mode, pseudo, score),
            )
            score_id = cur.lastrowid
            # Calcule le rang du nouveau score dans le top 10
            rank_row = conn.execute(
                "SELECT COUNT(*) + 1 AS rank FROM quiz_scores "
                "WHERE mode = ? AND (score > ? OR (score = ? AND id < ?))",
                (mode, score, score, score_id),
            ).fetchone()
            rank = rank_row["rank"] if rank_row else None
    except sqlite3.Error as exc:
        _log.warning("quiz score write failed: %s", exc)
        return jsonify({"error": "db"}), 503
    return jsonify({"ok": True, "rank": rank, "pseudo": pseudo, "score": score})


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
    # Fingerprint = mtime du fichier vidéo. Ajouté en query string sur la src
    # du <video> pour qu'un re-encode invalide tous les caches navigateur
    # (sans ça, max-age=86400 sur /api/video/* fait que le browser garde
    # l'ancienne version mp4v pendant 24 h).
    video_path = os.path.join(_HERE, "static", "learn", f"{letter}.mp4")
    try:
        v_tag = int(os.path.getmtime(video_path))
    except OSError:
        v_tag = 0
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>html,body{margin:0;background:#000;height:100%;overflow:hidden}"
        "video{width:100%;height:100%;object-fit:cover;display:block}</style>"
        "</head><body>"
        f"<video id='v' src='/api/video/{letter}?v={v_tag}' autoplay muted playsinline controls></video>"
        "<script>"
        "(function(){"
        "var v=document.getElementById('v');"
        # Parse les params du hash : #quiz&speed=2&once
        # - speed=N    -> playbackRate
        # - once       -> pas de loop, lecture unique
        # - sinon      -> loop manuel (event 'ended' relance)
        "var h=(location.hash||'').replace(/^#/,'').toLowerCase();"
        "var params=h.split('&').reduce(function(a,kv){"
        "  var p=kv.split('='); if(p[0]) a[p[0]]=p[1]||true; return a;"
        "}, {});"
        "var speed=parseFloat(params.speed||1);"
        "if(isFinite(speed)&&speed>0){ v.playbackRate=speed; v.defaultPlaybackRate=speed; }"
        "var once=!!params.once;"
        # Loop par défaut (pas en mode 'once')
        "function relance(){ if(!once){ v.currentTime=0; v.play(); } }"
        "v.addEventListener('ended', relance);"
        "v.addEventListener('timeupdate', function(){"
        "  if(v.duration && v.duration - v.currentTime < 0.15){"
        "    if(once){ /* laisse finir, l'event ended s'occupera de signaler */ }"
        "    else { v.currentTime=0; v.play(); }"
        "  }"
        "});"
        # Notifie le parent que la vidéo s'est terminée (utile en mode once)
        "v.addEventListener('ended', function(){"
        "  try{ parent.postMessage({hmh:'video_ended'}, '*'); }catch(e){}"
        "});"
        # Le parent peut envoyer { hmh: 'pause' } pour stopper net (blackout)
        "window.addEventListener('message', function(ev){"
        "  var d=ev.data; if(!d||typeof d!=='object') return;"
        "  if(d.hmh==='pause'){ v.pause(); }"
        "  if(d.hmh==='play'){ v.play(); }"
        "});"
        "})();"
        "</script>"
        "</body></html>"
    )
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/healthz")
def healthz():
    if _engine is not None:
        return _engine.health()
    return {"backend": "http", "model_api": MODEL_API_URL}


@app.route("/monitor")
def monitor():
    return render_template("monitor.html")


# ── Auth basique pour les routes debug ──────────────────────────────────────
# Le mot de passe vit en variable d'env WEBCAM_DEBUG_PASS, défaut "admin".
# On protège ET la page HTML ET l'API JSON — sinon n'importe qui pourrait
# tout simplement hitter /api/debug/sessions et obtenir les frames.
from functools import wraps
from flask import Response

WEBCAM_DEBUG_PASS = os.environ.get("WEBCAM_DEBUG_PASS", "admin")


def _require_webcam_auth(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        # On accepte n'importe quel username, seul le password compte.
        if not auth or auth.password != WEBCAM_DEBUG_PASS:
            return Response(
                "Accès debug protégé.", 401,
                {"WWW-Authenticate": 'Basic realm="HearMyHands debug"'},
            )
        return view(*args, **kwargs)
    return wrapper


@app.route("/webcam")
@_require_webcam_auth
def webcam_debug():
    """Vue debug : affiche en grille les webcams des sessions actives.

    À utiliser uniquement en démo / debug — c'est une vue privée des
    flux webcam de toutes les personnes connectées au même moment.
    Protégé par Basic Auth (password = $WEBCAM_DEBUG_PASS, défaut 'admin').
    """
    return render_template("webcam.html")


@app.route("/api/debug/sessions")
@_require_webcam_auth
def api_debug_sessions():
    """Liste des sessions Socket.IO actives + leur dernière frame.

    Polling JSON depuis /webcam. La frame est encodée en base64 (data URI
    directement utilisable comme src d'un <img>).
    """
    import base64
    now = time.time()
    out = []
    with _last_frames_lock:
        # Cleanup des sessions trop vieilles au passage
        stale = [sid for sid, (ts, _) in _last_frames.items()
                 if now - ts > _LAST_FRAME_MAX_AGE_S]
        for sid in stale:
            _last_frames.pop(sid, None)
        items = list(_last_frames.items())
    # On consulte les IPs hors du verrou frames pour ne pas créer de
    # contention entre les deux dicts.
    with _session_ips_lock:
        ips = {sid: _session_ips.get(sid, "?") for sid, _ in items}
    for sid, (ts, jpeg) in items:
        age = now - ts
        out.append({
            "sid":    sid[:8],                        # gardé pour debug avancé
            "ip":     ips.get(sid, "?"),
            "age_ms": int(age * 1000),
            "size_b": len(jpeg),
            "image":  "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
        })
    # Trie par âge croissant (les plus actifs en premier)
    out.sort(key=lambda s: s["age_ms"])
    return {"sessions": out, "count": len(out), "server_ts": now}


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
        # buf  : deque de 42-vecteurs, padding à zéro si pas de main détectée
        # mask : deque alignée, 1.0 si frame avec main, 0.0 sinon
        # tick : compteur de frames pour le throttling SIGN_EVERY_N
        _sessions[request.sid] = {
            "buf":  deque(maxlen=SEQ_LEN),
            "mask": deque(maxlen=SEQ_LEN),
            "tick": 0,
        }
    # Mémorise l'IP pour la vue debug
    with _session_ips_lock:
        _session_ips[request.sid] = _client_ip()


@socketio.on("disconnect")
def _on_disconnect():
    with _sessions_lock:
        _sessions.pop(request.sid, None)
    with _last_frames_lock:
        _last_frames.pop(request.sid, None)
    with _session_ips_lock:
        _session_ips.pop(request.sid, None)


@socketio.on("ping_test")
def _on_ping_test(client_ts=None):
    """Echo immédiat — sert au bouton debug ping côté client.

    Pas d'inférence, juste un round-trip Socket.IO pour mesurer le RTT.
    """
    return {"server_ts": time.time(), "client_ts": client_ts}


@socketio.on("frame")
def handle_frame(image_bytes, flags=None):
    """Decode + predict + ack.

    flags : dict optionnel, transmis par le client. Champs reconnus :
      - predict_sign (bool, default True) :
            False  -> on remplit toujours le buffer GRU (zéro coût) mais on
                      n'appelle PAS l'inférence GRU. Utile quand le client
                      est en mode statique, ou en mode dynamique sous le
                      seuil (où la prédiction est inutile).
            True   -> comportement classique : un appel GRU tous les
                      SIGN_EVERY_N frames si le buffer est plein.
    """
    if not image_bytes:
        return _EMPTY
    # Cache de la dernière frame brute pour la vue debug /webcam. On le fait
    # avant tout traitement pour rester visible même si l'inférence échoue.
    with _last_frames_lock:
        _last_frames[request.sid] = (time.time(), bytes(image_bytes))
    data = _run_frame(image_bytes)
    if data is None:
        return _EMPTY

    hands       = data.get("hands", []) or []
    handedness  = data.get("handedness", []) or []
    img_w       = data.get("image_width", 0) or 0
    img_h       = data.get("image_height", 0) or 0

    if flags is None or not isinstance(flags, dict):
        flags = {}
    want_sign = bool(flags.get("predict_sign", True))
    sign, sign_conf = _maybe_predict_sign(
        request.sid, hands, handedness, img_w, img_h, want_sign,
    )

    return {
        "skeleton":        data.get("keypoints"),
        "hands":           hands,
        "letter":          data.get("letter"),
        "confidence":      data.get("confidence"),
        "sign":            sign,
        "sign_confidence": sign_conf,
    }


def _maybe_predict_sign(sid, hands, handedness, img_w, img_h, want_sign=True):
    """Maintient toujours le buffer + mask, appelle le GRU si want_sign.

    Sémantique du buffer (alignée sur Modèle_Ocarina/demo.py) :
    - À CHAQUE frame on push une entrée. Si une main est détectée, le
      vecteur normalisé V2 et mask=1. Sinon, vecteur nul et mask=0.
    - La timeline n'est donc jamais compressée : un trou (main hors champ)
      est reflété tel quel dans la séquence, et le mask permet au GRU
      d'ignorer les paddings via masked mean-pool.

    handedness : liste alignée sur hands, valeurs "Right" / "Left".
    """
    if _sign_api_disabled:
        return None, None
    with _sessions_lock:
        state = _sessions.get(sid)
        if state is None:
            return None, None

        # 1) Push systématique au buffer et au mask
        if hands and img_w and img_h:
            hand0 = hands[0]
            side = handedness[0] if handedness else "Right"
            state["buf"].append(_normalize_hand(hand0, img_w, img_h, side))
            state["mask"].append(1.0)
        else:
            state["buf"].append([0.0] * 42)
            state["mask"].append(0.0)
        state["tick"] += 1

        if not want_sign:
            return None, None

        # 2) Conditions pour tenter l'inférence
        if state["tick"] % SIGN_EVERY_N != 0:
            return None, None
        real_count = sum(state["mask"])
        if real_count < MIN_REAL_FRAMES:
            return None, None

        # 3) Snapshot. Si on n'a pas encore SEQ_LEN entrées, on pad à gauche
        #    avec des zéros (et le mask à 0) pour matcher la shape attendue.
        sequence = list(state["buf"])
        mask     = list(state["mask"])
        if len(sequence) < SEQ_LEN:
            pad = SEQ_LEN - len(sequence)
            sequence = [[0.0] * 42] * pad + sequence
            mask     = [0.0] * pad + mask

    result = _run_sign(sequence, mask)
    if result is None:
        return None, None
    return result.get("sign"), result.get("confidence")


# ── Pré-traitement V2 ────────────────────────────────────────────────────────
# Doit rester strictement aligné avec
# Modèle_Ocarina/Dataset.py::SignLanguageDataset.normalize_frame :
#   1) coordonnées normalisées par l'image (-> [0, 1])
#   2) handedness canonicalization : si la main est gauche, miroir x
#   3) centrage sur le wrist (landmark 0)
#   4) division par la taille de main (distance wrist <-> middle MCP)
_WRIST_IDX      = 0
_MIDDLE_MCP_IDX = 9
_CANONICAL_HAND = "Right"


def _normalize_hand(hand, img_w, img_h, handedness="Right"):
    """Retourne une liste plate de 42 floats normalisés V2.

    hand        : 21 [x, y] en coordonnées pixel runtime
    img_w/img_h : dimensions de l'image runtime
    handedness  : "Right" ou "Left" (vient de MediaPipe.handedness)
    """
    if not hand or not img_w or not img_h or len(hand) < 21:
        return [0.0] * 42

    # 1) en coords normalisées [0, 1]
    xs = [pt[0] / img_w for pt in hand]
    ys = [pt[1] / img_h for pt in hand]

    # 2) mirror si pas du côté canonique
    if handedness and handedness != _CANONICAL_HAND:
        xs = [1.0 - x for x in xs]

    # 3) centrage sur le wrist
    wx, wy = xs[_WRIST_IDX], ys[_WRIST_IDX]
    xs = [x - wx for x in xs]
    ys = [y - wy for y in ys]

    # 4) scale par la taille de main = |wrist -> middle MCP|
    hs = (xs[_MIDDLE_MCP_IDX] ** 2 + ys[_MIDDLE_MCP_IDX] ** 2) ** 0.5
    if hs > 1e-6:
        xs = [x / hs for x in xs]
        ys = [y / hs for y in ys]

    # Interleave x0,y0,x1,y1,...
    out = [0.0] * 42
    for i in range(21):
        out[2 * i]     = xs[i]
        out[2 * i + 1] = ys[i]
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
