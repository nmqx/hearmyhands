"""
Live test interface for the Ocarina GRU sign-language model.

Same pipeline as before, but now:
  - Model loading status is shown in the UI (not just the console).
  - If prediction throws, the error appears in the UI.
  - --debug prints every prediction to the console.
  - Predicts as soon as there are >=5 real-hand frames (don't wait for 45).
  - Resolves weight/class paths against the script's own folder so the
    demo works regardless of the cwd the user launched it from.
"""

import argparse
import collections
import json
import os
import sys
import tkinter as tk
import traceback
from tkinter import ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Dataset import SignLanguageDataset  # noqa: E402
from Ocarina_GRU import SignLanguageGRU   # noqa: E402

import mediapipe as mp


SEQ_LEN = 45
NUM_FEATURES = 42
DEFAULT_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PRED_EVERY_N_FRAMES = 3


# ============================================================
def landmarks_to_tensor(hand_landmarks, handedness_label):
    """MediaPipe 21 normalized landmarks -> 42-vec, then wrist-center + scale."""
    xy = np.empty(NUM_FEATURES, dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        xy[2 * i] = lm.x
        xy[2 * i + 1] = lm.y
    t = torch.from_numpy(xy)
    return SignLanguageDataset.normalize_frame(
        t, handedness=handedness_label, canonical_hand="Right"
    )


# ============================================================
class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        root.title("Ocarina GRU -- live demo")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.classes, self.classes_msg = self._load_classes(args.classes)
        self.model, self.model_msg, self.model_loaded = self._build_model(
            args.weights
        )
        self.model.eval()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # On Windows DSHOW is usually faster + less laggy than the default backend.
        if os.name == "nt":
            self.cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {args.camera}")

        self.buffer = collections.deque(maxlen=SEQ_LEN)
        self.mask = collections.deque(maxlen=SEQ_LEN)
        self.frame_count = 0
        self.last_top3 = []
        self.recording = True
        self.last_error = ""

        self._build_ui()
        self._refresh_status()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update()

    # --------------------------------------------------------
    def _load_classes(self, path):
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    cls = json.load(f)
                return cls, f"classes: {len(cls)} from {os.path.basename(path)}"
            except Exception as e:
                return DEFAULT_CLASSES, f"classes file unreadable ({e}); using A-Z"
        return DEFAULT_CLASSES, f"no classes file at {path!r}; using A-Z"

    def _build_model(self, weights_path):
        model = SignLanguageGRU(
            input_size=NUM_FEATURES, hidden_size=96, num_layers=2,
            num_classes=len(self.classes), bidirectional=True,
        ).to(self.device)

        if not weights_path:
            return model, "no --weights given (random init)", False
        if not os.path.exists(weights_path):
            return model, f"weights not found: {weights_path}", False
        try:
            state = torch.load(weights_path, map_location=self.device,
                               weights_only=True)
            model.load_state_dict(state)
            return model, f"loaded {os.path.basename(weights_path)} on {self.device}", True
        except Exception as e:
            msg = (f"weights mismatch: {e}. Retrain with the new Train.py "
                   "(bidir, 2 layers, hidden=96).")
            return model, msg, False

    # --------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid()

        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=(0, 12))

        ttk.Label(main, text="Top predictions",
                  font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=1, sticky="w")

        self.pred_labels = []
        self.pred_bars = []
        for i in range(3):
            lab = ttk.Label(main, text="—", width=28,
                            font=("TkFixedFont", 14))
            lab.grid(row=1 + i * 2, column=1, sticky="w", pady=(4, 0))
            bar = ttk.Progressbar(main, length=240, maximum=100)
            bar.grid(row=2 + i * 2, column=1, sticky="w")
            self.pred_labels.append(lab)
            self.pred_bars.append(bar)

        self.buf_label = ttk.Label(main, text="Buffer: 0/45")
        self.buf_label.grid(row=7, column=1, sticky="w", pady=(12, 0))

        # persistent status line for model/classes
        self.status_label = ttk.Label(
            main, text="", foreground="gray", wraplength=260, justify="left",
        )
        self.status_label.grid(row=8, column=1, sticky="w", pady=(4, 0))

        # error / hint line, always visible
        self.error_label = ttk.Label(
            main, text="", foreground="#b00", wraplength=260, justify="left",
        )
        self.error_label.grid(row=9, column=1, sticky="w", pady=(4, 0))

        btns = ttk.Frame(main)
        btns.grid(row=10, column=0, columnspan=2, pady=(12, 0), sticky="w")
        ttk.Button(btns, text="Clear buffer",
                   command=self.clear_buffer).grid(row=0, column=0, padx=(0, 6))
        self.rec_btn = ttk.Button(btns, text="Pause", command=self.toggle_record)
        self.rec_btn.grid(row=0, column=1)

    def _refresh_status(self):
        self.status_label.config(text=f"{self.model_msg}\n{self.classes_msg}")
        if self.last_error:
            self.error_label.config(text=self.last_error)

    # --------------------------------------------------------
    def clear_buffer(self):
        self.buffer.clear()
        self.mask.clear()

    def toggle_record(self):
        self.recording = not self.recording
        self.rec_btn.config(text="Pause" if self.recording else "Resume")

    def on_close(self):
        try:
            self.cap.release()
            self.hands.close()
        finally:
            self.root.destroy()

    # --------------------------------------------------------
    def update(self):
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self.update)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)
        hand_seen = False

        if results.multi_hand_landmarks and self.recording:
            hand_seen = True
            lm = results.multi_hand_landmarks[0]
            handed_label = "Right"
            if results.multi_handedness:
                handed_label = results.multi_handedness[0].classification[0].label
                # Image was horizontally flipped above, so MediaPipe's
                # Left/Right labels are reversed vs the user's actual hand.
                handed_label = "Left" if handed_label == "Right" else "Right"

            feat = landmarks_to_tensor(lm, handed_label)
            self.buffer.append(feat)
            self.mask.append(1.0)

            self.mp_draw.draw_landmarks(
                rgb, lm, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        elif self.recording:
            self.buffer.append(torch.zeros(NUM_FEATURES))
            self.mask.append(0.0)

        # -------- predict --------
        self.frame_count += 1
        if (self.frame_count % PRED_EVERY_N_FRAMES == 0
                and self.model_loaded
                and sum(self.mask) >= 5):
            try:
                self._predict()
            except Exception as e:
                self.last_error = f"predict error: {e}"
                self.error_label.config(text=self.last_error)
                if self.args.debug:
                    traceback.print_exc()

        # -------- redraw --------
        self._draw_overlay(rgb, hand_seen)
        img = Image.fromarray(rgb).resize((480, 360))
        self._tkimg = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._tkimg)

        self.buf_label.config(
            text=f"Buffer: {len(self.buffer)}/{SEQ_LEN}   "
                 f"({int(sum(self.mask))} with hand)"
        )

        self.root.after(15, self.update)

    # --------------------------------------------------------
    def _predict(self):
        x = list(self.buffer)
        m = list(self.mask)
        if len(x) < SEQ_LEN:
            pad_n = SEQ_LEN - len(x)
            x = x + [torch.zeros(NUM_FEATURES)] * pad_n
            m = m + [0.0] * pad_n
        x_t = torch.stack(x).unsqueeze(0).to(self.device)
        m_t = torch.tensor(m).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x_t, m_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top3 = probs.argsort()[-3:][::-1]
        self.last_top3 = [(self.classes[i], float(probs[i])) for i in top3]

        for i, (lbl, p) in enumerate(self.last_top3):
            self.pred_labels[i].config(text=f"{lbl}    {p*100:5.1f}%")
            self.pred_bars[i]["value"] = p * 100

        if self.args.debug:
            print(" | ".join(f"{l}:{p*100:.1f}%" for l, p in self.last_top3))


    def _draw_overlay(self, rgb, hand_seen):
        h, w, _ = rgb.shape
        color = (0, 200, 0) if hand_seen else (200, 50, 50)
        cv2.rectangle(rgb, (4, 4), (w - 4, h - 4), color, 2)
        if self.last_top3:
            lbl, p = self.last_top3[0]
            cv2.putText(rgb, f"{lbl}  {p*100:.0f}%", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ocarina_gru_v2.pth")
    ap.add_argument("--classes", default="ocarina_classes.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="print every prediction to the console")
    args = ap.parse_args()

    # resolve relative paths against the script's directory, so the demo
    # works regardless of the cwd the user launched it from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.weights):
        args.weights = os.path.join(script_dir, args.weights)
    if not os.path.isabs(args.classes):
        args.classes = os.path.join(script_dir, args.classes)

    print(f"[info] weights: {args.weights} "
          f"(exists: {os.path.exists(args.weights)})")
    print(f"[info] classes: {args.classes} "
          f"(exists: {os.path.exists(args.classes)})")

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()