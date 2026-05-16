# HearMyHands

Real-time sign-language helper: a webcam stream is sent over a WebSocket to a
Flask web app, which forwards each frame to a local PyTorch model API. The
model returns body keypoints (custom ResNet-50 + heatmaps) and hand landmarks
(MediaPipe), drawn back on the page as an overlay skeleton.

```
Browser  ──(binary JPEG over Socket.IO)──▶  webapp (Flask)
                                            │
                                            └─(POST /model_predict)─▶ model API ─▶ JSON keypoints
```

## What's optimized

The WebSocket transport is the hot path for this app. Recent changes:

- **Binary frames over Socket.IO** instead of base64 data URLs — removes the
  ~33 % encoding overhead and an extra base64 round-trip on the server.
- **Client-side downscale to 480 px** wide before JPEG encoding — typical
  payload drops from ~80 KB to ~12 KB per frame at q=0.7.
- **Single-roundtrip ack callback** instead of a separate `prediction` event —
  cleaner state, no race on `isWaiting`.
- **Persistent `requests.Session`** between the web app and the model API —
  reuses the TCP connection.
- **Raw-bytes endpoint** at `/model_predict` (Content-Type `application/octet-stream`)
  — no JSON / base64 round-trip on the model side.
- **Frame size cap** (`max_http_buffer_size=2 MB`) on the Socket.IO server to
  prevent runaway payloads.

## Layout

```
.
├── hearmyhands/    Flask web app (UI + Socket.IO transport)
│   ├── app.py
│   ├── static/
│   └── templates/
└── HmH/            Model API + training/inference scripts
    ├── api.py
    └── heatnoks/
        ├── model.py                  (ResNet-50 + heatmap head + spatial softmax)
        ├── inference_video.py
        ├── inference_heatnoks.py
        ├── hand_landmarker.task      (MediaPipe HandLandmarker, included)
        └── checkpoints/best.pt       (download from Releases, see below)
```

## Setup

```bash
pip install -r requirements.txt
```

### Get the model weights

`best.pt` (~390 MB) is published as a GitHub Release asset rather than
committed to the repo. Download it and drop it into either
`HmH/heatnoks/checkpoints/` or `HmH/heatnoks/checkpoints_pretrain/`:

```bash
mkdir -p HmH/heatnoks/checkpoints
# from the GitHub Releases page, save best.pt into that folder
```

## Run

Two processes, two ports.

```bash
# 1) Model API (port 5001)
python HmH/api.py

# 2) Web app (port 5000)
python hearmyhands/app.py
```

Open <http://localhost:5000/translate>, click *Activer Caméra*, then
*Lancer Traduction*.

### Configuration (env vars)

| Variable          | Default                                  | Where      |
| ----------------- | ---------------------------------------- | ---------- |
| `MODEL_API_URL`   | `http://127.0.0.1:5001/model_predict`    | `app.py`   |
| `MODEL_TIMEOUT`   | `5` (seconds)                            | `app.py`   |
| `PORT`            | `5000` (web) / `5001` (model)            | both       |

## Inference scripts (no web app)

```bash
# Video file
python HmH/heatnoks/inference_video.py \
    --ckpt HmH/heatnoks/checkpoints/best.pt \
    --source path/to/video.mp4 --rotate 180 --skip 3

# Webcam
python HmH/heatnoks/inference_video.py \
    --ckpt HmH/heatnoks/checkpoints/best.pt --source 0
```

## License

See LICENSE if present, otherwise all rights reserved by the original authors.
