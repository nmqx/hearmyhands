# HearMyHands

Real-time sign-language helper. A browser sends webcam frames over a WebSocket
to a small Flask app, which forwards them to a PyTorch model that returns body
keypoints (custom ResNet-50 + heatmaps) and hand landmarks (MediaPipe). The
overlay is drawn back on the page.

```
Browser ──Socket.IO──▶ webapp ──HTTP──▶ model API ──▶ keypoints + hands
```

## Layout

```
hearmyhands/   Flask web app (UI + WebSocket transport)
HmH/           Model API + training/inference scripts
```

## Setup

```bash
pip install -r requirements.txt
```

Grab `best.pt` from the [latest release](../../releases/latest) and put it in
`HmH/heatnoks/checkpoints/`.

## Run

Two processes:

```bash
python HmH/api.py            # model API on :5001
python hearmyhands/app.py    # web app on :5000
```

Open <http://localhost:5000/translate>, allow the camera, and hit *Lancer
Traduction*.

### Config (env vars)

| Variable        | Default                               |
| --------------- | ------------------------------------- |
| `MODEL_API_URL` | `http://127.0.0.1:5001/model_predict` |
| `MODEL_TIMEOUT` | `5`                                   |
| `PORT`          | `5000` (web) / `5001` (model)         |

## Standalone inference

```bash
# video file
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source video.mp4

# webcam
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source 0
```
