import json
import numpy as np
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

Alphabet =['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
def simplifier(data):
    simplified_data = []
    for frame in data:
        # reccupere les infos utiles
        keypoints = frame["hands"]["keypoints"]
        if len(keypoints) >= 21:
            # Coordonnées du poignet (Point 0)
            wx, wy = keypoints[0]["x"], keypoints[0]["y"]
            # On soustrait la position du poignet pour centrer la main
            numbers = [coord for point in keypoints for coord in (point["x"] - wx, point["y"] - wy)]
            # créé une liste avec les infos utiles et la renvoie
            simplified_data.append(numbers)
    return simplified_data

def fusion(data) :
    data_finale = []
    print(len(data))
    for i in range(len(data)) :
         liste = simplifier(data[i])
         for frame in liste : 
              frame.append(i)
         # Ajoute un indice d'origine pour retracer les frames lors de la fusion
              if len(frame) == 43 :
                data_finale.append(frame)    
    # Melange 
    random.shuffle(data_finale)

    # Créé une liste qui permet de retracer l'origine des frames
    origin_labels = [frame[-1] for frame in data_finale]
    # Supprime l'indice d'orine de la frame
    data_finale = [frame[:-1] for frame in data_finale]
    return origin_labels, data_finale

dossier_destination = BASE_DIR / "Tests_centered"
dossier_destination.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_EVAL_LABEL = dossier_destination / "eval_label.json"
OUTPUT_JSON_EVAL_KEYPOINTS = dossier_destination / "eval_keypoints.json"
OUTPUT_JSON_TRAINING_LABEL = dossier_destination / "training_label.json"
OUTPUT_JSON_TRAINING_KEYPOINTS = dossier_destination / "training_keypoints.json"

#Lit les fichiers JSON (datasets)
data = []
for i in range(len(Alphabet)) :

    file = BASE_DIR / "Data_sets" / "data" / f"{Alphabet[i]}.json"
    with open(file, "r") as f1:
        data1 = (json.load(f1))
        data.append(data1)

print(len(data))
label, keypoints = fusion(data)
lenght = int(len(keypoints)*0.8)
label, label_eval = label[:lenght], label[lenght:]
keypoints, keypoints_eval = keypoints[:lenght], keypoints[lenght:]

#ecrit un Json permettant de retracer l'origine des frames du dataset fusionné

with open(OUTPUT_JSON_EVAL_LABEL, "w") as f:
        json.dump(label_eval, f, indent=2)    
#Ecrit le datasets fusionné 
with open(OUTPUT_JSON_EVAL_KEYPOINTS, "w") as f:
        json.dump(keypoints_eval, f, indent=2)
with open(OUTPUT_JSON_TRAINING_KEYPOINTS, "w") as f:
        json.dump(keypoints, f, indent=2)
with open(OUTPUT_JSON_TRAINING_LABEL, "w") as f:
        json.dump(label, f, indent=2)