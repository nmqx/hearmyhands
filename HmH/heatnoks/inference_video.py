import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Permet d'importer model.py depuis n'importe où
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from model import HeatnoksModel

# ── Constantes globales ───────────────────────────────────────────────────────
INPUT_SIZE     = 256
NUM_KEYPOINTS  = 9
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]

VIS_THRESHOLD  = 0.3

# (B, G, R) au lieu de (R, G, B) car OpenCV utilise BGR
COLOR_NECK    = (255, 255, 255) # Blanc
COLOR_ARMS_L  = (255, 200, 50)  # Cyan
COLOR_ARMS_R  = (120, 80, 255)  # Rose
COLOR_BODY    = (120, 255, 50)  # Vert

def get_color(idx):
    if idx == 0: return COLOR_NECK
    if idx in [1, 3, 5]: return COLOR_ARMS_L
    if idx in [2, 4, 6]: return COLOR_ARMS_R
    return COLOR_BODY

SKELETON_EDGES = [
    (0, 1), (0, 2), # Cou -> Épaules
    (1, 3), (3, 5), # Bras gauche
    (2, 4), (4, 6), # Bras droit
    (1, 7), (2, 8), # Épaule -> Hanche
    (7, 8)          # Hanches
]

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]


# ── Fonctions Utilitaires ─────────────────────────────────────────────────────

def preprocess_frame(frame):
    """Prépare la frame OpenCV pour le modèle (padding carré, normalisation, etc)."""
    # Convertir BGR en RGB pour torchvision / PIL
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    orig_w, orig_h = img.size
    
    # Padding pour garder l'aspect ratio (au lieu du center crop qui coupe les bords)
    max_side = max(orig_w, orig_h)
    pad_left = (max_side - orig_w) // 2
    pad_top = (max_side - orig_h) // 2
    pad_right = max_side - orig_w - pad_left
    pad_bottom = max_side - orig_h - pad_top
    
    img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
    
    # Redimensionnement et Tensorisation
    img = TF.resize(img, [INPUT_SIZE, INPUT_SIZE])
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=MEAN, std=STD)
    
    return tensor.unsqueeze(0), max_side, pad_left, pad_top


def draw_predictions_cv2(frame, keypoints, pad_size, pad_left, pad_top):
    """Dessine le squelette et les keypoints directement sur la frame (BGR)."""
    kp_px = keypoints.copy()
    
    # Remettre à l'échelle de l'image de base (enlever le padding)
    kp_px[:, 0] = kp_px[:, 0] * pad_size - pad_left
    kp_px[:, 1] = kp_px[:, 1] * pad_size - pad_top

    # Dessiner les liens (os/segments)
    for (i, j) in SKELETON_EDGES:
        xi, yi, vi = kp_px[i]
        xj, yj, vj = kp_px[j]
        if vi < VIS_THRESHOLD or vj < VIS_THRESHOLD:
            continue
            
        color = get_color(i)
        start_point = (int(xi), int(yi))
        end_point   = (int(xj), int(yj))
        cv2.line(frame, start_point, end_point, color, 4, cv2.LINE_AA)

    # Dessiner les articulations (points)
    for k in range(NUM_KEYPOINTS):
        x, y, v = kp_px[k]
        if v < VIS_THRESHOLD:
            continue
            
        color = get_color(k)
        center = (int(x), int(y))
        cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)
        
        # Effet bordure noire simple autour du cercle
        cv2.circle(frame, center, 6, (0, 0, 0), 1, cv2.LINE_AA)


# ── Boucle Principale ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inférence vidéo / temps réel pour Heatnoks")
    default_ckpt = os.path.join(script_dir, "checkpoints", "best.pt")
    
    parser.add_argument("--ckpt", default=default_ckpt, help="Chemin vers le modèle (best.pt)")
    parser.add_argument("--source", default="0", help="ID de la webcam (0, 1...) ou chemin vers une vidéo mp4")
    parser.add_argument("--output", default=None, help="Chemin pour enregistrer la vidéo de sortie (ex: output.mp4)")
    parser.add_argument("--rotate", type=int, default=0, help="Rotation (90, 180, 270) si la vidéo sort tournée.")
    parser.add_argument("--skip", type=int, default=1, help="Ne calculer l'inférence qu'une frame sur N pour accélérer.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil cible : {device}")

    # Initialisation de MediaPipe Hands (Tasks API)
    task_path = os.path.join(script_dir, "hand_landmarker.task")
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hands_detector = mp_vision.HandLandmarker.create_from_options(options)

    # 1. Charger le modèle
    model = HeatnoksModel(pretrained=False).to(device)
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch_str = ckpt.get('epoch', '?')
        print(f"Modèle chargé (Époque {epoch_str}) depuis {args.ckpt}")
    else:
        print(f"Attention: Aucun modèle trouvé à {args.ckpt}. Assure-toi que le chemin est bon.")
        return
        
    model.eval()

    # 2. Ouvrir le flux
    # Règle d'argument : 0, 1... si numéros (Webcam), sinon c'est un chemin de vidéo statique
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la source '{args.source}'")
        return

    print("--- Lancement du flux vidéo ---")
    print("Appuie sur 'q' pour quitter, et d'autres touches pour interagir si besoin.")

    # 3. Initialiser le VideoWriter si demandé
    out_video = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if fps_out == 0 or fps_out > 100: # Cas webcam ou FPS inconnu
            fps_out = 30.0
        
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.rotate in [90, 270, -90]:
            width, height = height, width
            
        out_video = cv2.VideoWriter(args.output, fourcc, fps_out, (width, height))
        print(f"Enregistrement activé : {args.output} ({width}x{height} @ {fps_out}fps)")

    # Variables FPS
    prev_time = time.time()

    frame_count = 0
    last_kp = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin du flux vidéo ou erreur de lecture.")
            break

        frame_count += 1

        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate in [270, -90]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Mesure prédiction seule (Optionnel si tu veux isoler l'inférence du preprocessing)
        t_inference_start = time.time()
        
        # Preprocessing (uniquement utile pour pad_size etc. mais on peut skip l'inférence)
        tensor, pad_size, pad_left, pad_top = preprocess_frame(frame)
        
        # On calcule seulement 1 frame sur args.skip
        if frame_count % args.skip == 1 or last_kp is None or args.skip == 1:
            tensor = tensor.to(device)
            # Inférence
            with torch.no_grad():
                out = model.predict(tensor) # [1, 9, 3]
            
            # Récupération au format numpy
            last_kp = out[0].cpu().numpy()
            
        kp = last_kp
        
        # FPS Global incluant preprocess + inférence
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Dessin du squelette principal Heatnoks
        draw_predictions_cv2(frame, kp, pad_size, pad_left, pad_top)
        
        # --- Encadré et MediaPipe pour les mains ---
        kp_px = kp.copy()
        kp_px[:, 0] = kp_px[:, 0] * pad_size - pad_left
        kp_px[:, 1] = kp_px[:, 1] * pad_size - pad_top
        
        # Calcul de la distance entre les épaules pour avoir une échelle de référence stable
        # 1=L_Shoulder, 2=R_Shoulder
        s1x, s1y, _ = kp_px[1] 
        s2x, s2y, _ = kp_px[2]
        sh_dist = np.sqrt((s1x - s2x)**2 + (s1y - s2y)**2)
        
        # 3=L_Elbow, 5=L_Wrist, 1=L_Shoulder | 4=R_Elbow, 6=R_Wrist, 2=R_Shoulder
        for elbow_idx, wrist_idx, shoulder_idx in [(3, 5, 1), (4, 6, 2)]:
            wx, wy, wv = kp_px[wrist_idx]
            ex, ey, ev = kp_px[elbow_idx]
            sx, sy, sv = kp_px[shoulder_idx]
            
            # On cherche la main si le poignet est confiant, OU si le coude est visible
            # (Si le modèle a perdu le poignet et l'a forcé à wv=0, on se sert du coude)
            if wv > VIS_THRESHOLD or (ev > VIS_THRESHOLD and sv > VIS_THRESHOLD):
                
                search_x, search_y = wx, wy
                dist_fw = np.sqrt((wx - ex)**2 + (wy - ey)**2) if wv > VIS_THRESHOLD else 0
                
                if ev > VIS_THRESHOLD and sv > VIS_THRESHOLD:
                    dist_ua = np.sqrt((ex - sx)**2 + (ey - sy)**2) # Upper Arm
                    
                    # Si le poignet est introuvable (wv=0), OU si le bras semble anormalement tendu
                    if (wv <= VIS_THRESHOLD) or (dist_fw < 0.4 * dist_ua):
                        unit_dx = (ex - sx) / (dist_ua + 1e-6)
                        unit_dy = (ey - sy) / (dist_ua + 1e-6)
                        # On projette le centre de recherche depuis le coude
                        search_x = ex + unit_dx * dist_ua * 0.9
                        search_y = ey + unit_dy * dist_ua * 0.9
                        # On donne une estimation de l'avant-bras pour la taille de boîte
                        dist_fw = dist_ua * 0.9 
                
                # Taille de boîte plus robuste (min 60% de la largeur d'épaules)
                box_size = int(max(sh_dist * 0.6, dist_fw * 1.6, 160))
                
                half = box_size // 2
                x1 = max(0, int(search_x - half))
                y1 = max(0, int(search_y - half))
                x2 = min(frame.shape[1], int(search_x + half))
                y2 = min(frame.shape[0], int(search_y + half))
                
                if x2 - x1 > 20 and y2 - y1 > 20:
                    # Dessiner le rectangle de la main (en orange)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    
                    # Extraire l'encadré
                    hand_crop = frame[y1:y2, x1:x2]
                    hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                    
                    # Détection des doigts avec MediaPipe Tasks API sur le crop
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_crop_rgb)
                    results = hands_detector.detect(mp_image)
                    
                    if results.hand_landmarks:
                        for hand_landmarks in results.hand_landmarks:
                            h, w, _ = hand_crop.shape
                            # Dessiner les connexions
                            for (i, j) in HAND_CONNECTIONS:
                                pt1 = (int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h))
                                pt2 = (int(hand_landmarks[j].x * w), int(hand_landmarks[j].y * h))
                                cv2.line(hand_crop, pt1, pt2, (255, 255, 255), 2)
                            # Dessiner les articulations
                            for lm in hand_landmarks:
                                pt = (int(lm.x * w), int(lm.y * h))
                                cv2.circle(hand_crop, pt, 3, (0, 0, 255), -1)
        # --- Fin MediaPipe ---
        
        # Texte de FPS sur l'image
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Enregistrement de la frame
        if out_video:
            out_video.write(frame)

        # Affichage
        cv2.imshow("Heatnoks Real-Time Inference", frame)

        # Input clavier : waitKey reçoit la touche pressée
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Arrêt par l'utilisateur.")
            break

    # Libération propre
    if out_video:
        out_video.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
