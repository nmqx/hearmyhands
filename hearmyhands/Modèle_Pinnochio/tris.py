import json
import numpy as np
import random

def simplifier(data):
    simplified_data = []
    for frame in data:
        # reccupere les infos utiles
        keypoints = frame["hands"]["keypoints"]
        numbers = [coord for point in keypoints for coord in (point["x"], point["y"])]
        
        # créé une liste avec les infos utiles et la renvoie
        simplified_data.append(numbers)
    return simplified_data

def fusion(data1, data2) :
    data1 = simplifier(data1)
    data2 = simplifier(data2)

    # Ajoute un indice d'origine pour retracer les frames lors de la fusion
    for frame in data1:
        frame = frame.append(0)  # vient du dataset 1
    for frame in data2:
        frame = frame.append(1)  # vient du dataset 2

    data1 = [frame for frame in data1 if len(frame) == 43] #43 parce que 42 nombres representent une frame + 1 pour l'indice d'origne
    data2 = [frame for frame in data2 if len(frame) == 43]

    # Fusion des deux datasets
    merged = data1 + data2

    # Melange 
    random.shuffle(merged)

    # Créé une liste qui permet de retracer l'origine des frames
    origin_labels = [frame[-1] for frame in merged]

    # Supprime l'indice d'orine de la frame
    merged = [frame[:-1] for frame in merged]
    return origin_labels, merged


file1 = "hand_keypoints.json"
file2 = "L.json"

#Lit les deux fichiers JSON (datasets)
with open(file1, "r") as f1:
    data1 = json.load(f1)

with open(file2, "r") as f1:
    data2 = json.load(f1)


data1, data2 = fusion(data1, data2)

#ecrit un Json permettant de retracer l'origine des frames du dataset fusionné
with open('label3.json', "w") as f:
        json.dump(data1, f, indent=2)    
#Ecrit le datasets fusionné 
with open('hand3.json', "w") as f:
        json.dump(data2, f, indent=2)   