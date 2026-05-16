"""HTTP wrapper around InferenceEngine — kept for backwards-compatible
service-split deployments. Local single-process runs import the engine
directly from `hearmyhands/app.py` and skip the HTTP entirely.
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request

from inference import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine()


@app.route("/model_predict", methods=["POST"])
def model_predict():
    if engine.model is None:
        return jsonify({"error": "model not loaded"}), 503
    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return jsonify({"error": "empty body"}), 400
    result = engine.predict_frame(image_bytes)
    if result is None:
        return jsonify({"error": "invalid image"}), 400
    return jsonify(result)


@app.route("/sign_predict", methods=["POST"])
def sign_predict():
    if engine.sign_classifier is None:
        return jsonify({"error": "sign classifier not loaded"}), 503
    payload = request.get_json(silent=True) or {}
    seq = payload.get("sequence")
    if not isinstance(seq, list):
        return jsonify({"error": "missing 'sequence' (list of frames)"}), 400
    result = engine.predict_sign(seq)
    if result is None:
        return jsonify({"error": "bad sequence shape (need 60 × 42)"}), 400
    return jsonify(result)


@app.route("/healthz")
def healthz():
    return jsonify(engine.health())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=False, threaded=True)
