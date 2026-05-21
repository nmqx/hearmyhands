"""
Dataset for sign language recognition.

CHANGES vs original:
  1. Coordinates are NORMALIZED to image size first (-> [0, 1]), then
     re-centered on wrist, then scaled by hand size.
     -> Same coordinate space as MediaPipe at inference time.
  2. Handedness canonicalization: every sample is flipped to a single
     reference hand (default: right). The JSON `annotations[i].handedness`
     field tells us the side; if it's 'Left' we mirror x.
  3. Hand-size normalization: divide centered coords by the distance
     wrist -> middle-finger MCP (landmark 9). Removes camera-distance
     variation.
  4. Augmentations: gaussian noise + occlusion + small rotation + scale
     jitter. Rotation/scale are 10x more effective than per-coord noise
     for hand-pose data.
  5. Returns (frames, mask, label). The mask is 1 for real frames, 0 for
     padding -> the model can mean-pool over real frames only.
  6. Reproducible split via list_samples() helper (sorted file list) so
     train/val never leak into each other.
"""

import os
import json
import math
import re

import torch
from torch.utils.data import Dataset


# MediaPipe Hands landmark indices we rely on for normalization
WRIST = 0
MIDDLE_MCP = 9


def _extract_label(filename: str) -> str:
    """Same magic regex as before, isolated for testability."""
    match = re.search(
        r'(?:prise\d+|corrigé|_)([A-Za-z])(?:[0-9_.]|$)',
        filename, re.IGNORECASE
    )
    return match.group(1).upper() if match else "ERREUR"


def list_samples(data_dir):
    """Return a sorted list of (path, label) for the dataset.

    Sorting is critical: the original Train.py created two Dataset instances
    and indexed both with the same numpy permutation, but the underlying
    os.listdir() is not deterministic across instances on all filesystems.
    With list_samples() we build the split *once* on a sorted list and pass
    indices around safely.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    samples = []
    for f in files:
        lbl = _extract_label(f)
        if lbl == "ERREUR":
            print(f"⚠️  Could not parse label from {f}")
            continue
        samples.append((os.path.join(data_dir, f), lbl))
    return samples


class SignLanguageDataset(Dataset):
    def __init__(
        self,
        samples,
        class_to_idx,
        max_frames=45,
        num_features=42,
        augment=False,
        noise_std=0.01,         # in normalized units (image is now [0,1])
        occlusion_prob=0.05,
        rot_deg=15.0,
        scale_jitter=0.10,
        canonical_hand="Right",  # mirror everything to this side
    ):
        """
        samples       : list of (path, label_str) -- use list_samples()
        class_to_idx  : dict label_str -> int
        max_frames    : sequence length (45 to match production)
        """
        self.samples = [(p, class_to_idx[l]) for p, l in samples]
        self.classes = sorted(class_to_idx, key=lambda k: class_to_idx[k])
        self.class_to_idx = class_to_idx
        self.max_frames = max_frames
        self.num_features = num_features
        self.augment = augment
        self.noise_std = noise_std
        self.occlusion_prob = occlusion_prob
        self.rot_deg = rot_deg
        self.scale_jitter = scale_jitter
        self.canonical_hand = canonical_hand

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------------
    # Core feature extraction. Exposed as a static method so the inference
    # script can call EXACTLY the same code on live MediaPipe data.
    # -------------------------------------------------------------------
    @staticmethod
    def normalize_frame(xy, handedness, canonical_hand="Right"):
        """
        xy           : tensor [42] of normalized [0,1] x,y values
                       (x0,y0,x1,y1,...,x20,y20)
        handedness   : "Left" or "Right" -- which physical hand this is
        canonical_hand: target side; mirror x if mismatch

        Returns a tensor [42] of wrist-centered, scale-normalized coords.
        Returns zeros if the frame is empty (all zeros).
        """
        if xy.abs().sum() == 0:
            return xy

        xs = xy[0::2].clone()
        ys = xy[1::2].clone()

        # 1) handedness canonicalization (mirror x around 0.5)
        if handedness and handedness != canonical_hand:
            xs = 1.0 - xs

        # 2) center on wrist
        wx, wy = xs[WRIST].item(), ys[WRIST].item()
        xs = xs - wx
        ys = ys - wy

        # 3) scale by hand size = |wrist -> middle MCP|
        hand_size = math.hypot(xs[MIDDLE_MCP].item(), ys[MIDDLE_MCP].item())
        if hand_size > 1e-6:
            xs = xs / hand_size
            ys = ys / hand_size

        out = torch.empty_like(xy)
        out[0::2] = xs
        out[1::2] = ys
        return out

    # -------------------------------------------------------------------
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "r") as f:
            data = json.load(f)

        W = float(data["info"]["width"])
        H = float(data["info"]["height"])

        # chronological frames
        images = sorted(data.get("images", []), key=lambda x: x["frame_index"])
        ann_by_id = {a["image_id"]: a for a in data.get("annotations", [])}

        frames = []
        mask = []
        for img in images:
            ann = ann_by_id.get(img["id"])
            if ann is None or "keypoints" not in ann:
                frames.append(torch.zeros(self.num_features))
                mask.append(0.0)
                continue

            kp = ann["keypoints"]
            # keep only x,y (drop visibility -- every 3rd value)
            xy = torch.tensor(
                [kp[i] for i in range(len(kp)) if i % 3 != 2],
                dtype=torch.float32,
            )

            # to normalized [0,1] -- same space MediaPipe gives at runtime
            xy[0::2] /= W
            xy[1::2] /= H

            xy = self.normalize_frame(
                xy,
                handedness=ann.get("handedness", self.canonical_hand),
                canonical_hand=self.canonical_hand,
            )
            frames.append(xy)
            mask.append(1.0)

        if not frames:
            frames = [torch.zeros(self.num_features)]
            mask = [0.0]

        x = torch.stack(frames)              # [T, 42]
        m = torch.tensor(mask)               # [T]

        # pad / truncate
        T = x.shape[0]
        if T > self.max_frames:
            # take the LAST max_frames -> end of the sign usually carries
            # the most discriminative pose
            x = x[-self.max_frames :]
            m = m[-self.max_frames :]
        elif T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, self.num_features)
            x = torch.cat([x, pad], dim=0)
            m = torch.cat([m, torch.zeros(self.max_frames - T)], dim=0)

        if self.augment:
            x = self._augment(x, m)

        return x, m, label

    # -------------------------------------------------------------------
    def _augment(self, x, m):
        """Apply augmentations only to non-padding frames."""
        valid = m.unsqueeze(1)  # [T, 1]

        # global rotation (applied per-sequence -> entire sign rotates)
        if self.rot_deg > 0:
            theta = math.radians((torch.rand(1).item() * 2 - 1) * self.rot_deg)
            c, s = math.cos(theta), math.sin(theta)
            xs = x[:, 0::2]
            ys = x[:, 1::2]
            xr = xs * c - ys * s
            yr = xs * s + ys * c
            x = x.clone()
            x[:, 0::2] = xr
            x[:, 1::2] = yr

        # scale jitter (per-sequence)
        if self.scale_jitter > 0:
            s = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale_jitter
            x = x * s

        # gaussian noise (per-coord, only valid frames)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std * valid

        # occlusion
        if self.occlusion_prob > 0:
            mask = (torch.rand_like(x) > self.occlusion_prob).float()
            x = x * mask

        # re-zero padding rows (rotation/scale leaves zeros as zeros, but
        # noise above is already gated by `valid`, so we're fine)
        x = x * valid
        return x