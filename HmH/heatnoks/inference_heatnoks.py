"""
inference_heatnoks.py — Run Heatnoks (9 points) model on an image and visualise results.

Usage:
    python inference_heatnoks.py --ckpt heatnoks/checkpoints/best.pt --img path/to/image.jpg
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

# On importe les composants depuis le dossier heatnoks
import sys
sys.path.append(os.path.join(os.getcwd(), "heatnoks"))

from model import HeatnoksModel

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE     = 256
NUM_KEYPOINTS  = 9
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]

VIS_THRESHOLD  = 0.3

# Noms des points pour l'affichage
NAMES = [
    "Neck", 
    "L_Shoulder", "R_Shoulder", 
    "L_Elbow", "R_Elbow", 
    "L_Wrist", "R_Wrist", 
    "L_Hip", "R_Hip"
]

# Couleurs (R, G, B)
COLOR_NECK    = (255, 255, 255) # Blanc
COLOR_ARMS    = (50, 200, 255)  # Cyan
COLOR_BODY    = (50, 255, 120)  # Vert

def get_color(idx):
    if idx == 0: return COLOR_NECK
    if idx in [1, 3, 5]: return COLOR_ARMS # Gauche
    if idx in [2, 4, 6]: return (255, 80, 120) # Droite (Rose)
    return COLOR_BODY

# Liste des segments du squelette (indices des points)
SKELETON_EDGES = [
    (0, 1), (0, 2), # Cou -> Épaules
    (1, 3), (3, 5), # Bras gauche
    (2, 4), (4, 6), # Bras droit
    (1, 7), (2, 8), # Épaule -> Hanche
    (7, 8)          # Hanches
]

def preprocess(img_path: str):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    
    # On fait un center crop comme à l'entraînement pour la cohérence
    crop_size = min(orig_w, orig_h)
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    img = TF.crop(img, top, left, crop_size, crop_size)
    
    img = TF.resize(img, [INPUT_SIZE, INPUT_SIZE])
    img = TF.to_tensor(img)
    img = TF.normalize(img, mean=MEAN, std=STD)
    return img.unsqueeze(0), crop_size, crop_size, left, top

def draw_predictions(img_path, keypoints, out_path, crop_size, left, top):
    # Charger l'image originale
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Remettre les points à l'échelle du crop puis de l'image originale
    kp_px = keypoints.copy()
    kp_px[:, 0] = kp_px[:, 0] * crop_size + left
    kp_px[:, 1] = kp_px[:, 1] * crop_size + top

    # Dessiner le squelette
    for (i, j) in SKELETON_EDGES:
        xi, yi, vi = kp_px[i]
        xj, yj, vj = kp_px[j]
        if vi < VIS_THRESHOLD or vj < VIS_THRESHOLD:
            continue
        alpha = int(min(vi, vj) * 200)
        color = get_color(i) + (alpha,)
        draw.line([(xi, yi), (xj, yj)], fill=color, width=6)

    # Dessiner les points
    R = 10
    for k in range(NUM_KEYPOINTS):
        x, y, v = kp_px[k]
        if v < VIS_THRESHOLD:
            continue
        alpha = int(v * 255)
        color = get_color(k) + (alpha,)
        draw.ellipse([(x - R, y - R), (x + R, y + R)], fill=color, outline=(0,0,0,alpha))
        # Texte optionnel
        # draw.text((x+R, y-R), NAMES[k], fill=(255,255,255,255))

    out_img = Image.alpha_composite(img, overlay).convert("RGB")
    out_img.save(out_path)
    print(f"Résultat sauvegardé dans : {out_path}")

def main():
    parser = argparse.ArgumentParser()
    # Déterminer le dossier dans lequel se trouve ce script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(script_dir, "checkpoints", "best.pt")
    parser.add_argument("--ckpt", default=default_ckpt)
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = HeatnoksModel(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Modèle chargé (Époque {ckpt.get('epoch', '?')})")

    # Preprocess
    tensor, crop_size, _, left, top = preprocess(args.img)
    tensor = tensor.to(device)

    # Inférence
    with torch.no_grad():
        out = model.predict(tensor) # [1, 9, 3]

    kp = out[0].cpu().numpy()

    # Visualisation
    if args.out is None:
        out_dir = Path(os.path.join(script_dir, "img_pred_ho"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(args.img).stem + "_pred.jpg")
    else:
        out_path = args.out

    draw_predictions(args.img, kp, out_path, crop_size, left, top)

    # Print confidence
    print("\nConfiance des points :")
    for i, name in enumerate(NAMES):
        print(f"  {name:<12} : {kp[i, 2]:.4f}")

if __name__ == "__main__":
    main()
