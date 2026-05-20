import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import re


class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, max_frames=45, num_features=42, augment=True,
                 noise_std=4.0, occlusion_prob=0.05):
        """
        max_frames    : doit matcher SEQ_LEN côté inférence (HmH/sign_classifier.py).
                        45 = la valeur utilisée en prod.
        noise_std     : écart-type du bruit gaussien d'augmentation, en pixels
                        (les features sont des coords pixels recentrées sur le
                        wrist, range typique +-200, donc ~5px de bruit est une
                        perturbation réaliste).
        occlusion_prob: probabilité qu'une coordonnée individuelle soit "cachée"
                        (mise à 0) pour simuler une landmark manquante.
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.num_features = num_features
        self.augment = augment
        self.noise_std = noise_std
        self.occlusion_prob = occlusion_prob
        self.samples = []
        
        #liste des fichiers
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        extracted_labels = []
        
        #détecteur magique
        for f in all_files:
            # Cette règle cherche une lettre unique située :
            # - Soit juste après "prise" + un chiffre (ex: prise1F)
            # - Soit juste après "corrigé" (ex: corrigéF2)
            # - Soit juste après un tiret du bas (ex: _A2.json)
            match = re.search(r'(?:prise\d+|corrigé|_)([A-Za-z])(?:[0-9_.]|$)', f, re.IGNORECASE)
            
            if match:
                extracted_labels.append(match.group(1).upper())
            else:
                # Si un de tes potes nomme son fichier "video_de_louna_salut.json", 
                # ça n'explosera pas, ça te préviendra juste dans la console.
                print(f"⚠️ ATTENTION : Impossible de trouver la lettre dans {f}")
                extracted_labels.append("ERREUR")
        
        #création des classes
        valid_labels = [l for l in extracted_labels if l != "ERREUR"]
        self.classes = sorted(list(set(valid_labels)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        #ssignation
        for file_name, label in zip(all_files, extracted_labels):
            if label != "ERREUR":
                file_path = os.path.join(data_dir, file_name)
                self.samples.append((file_path, self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        #on récupère et on trie les images par ordre chronologique
        images = sorted(data.get('images', []), key=lambda x: x['frame_index'])
        
        frames_data = []
        
        #sur chaque frame on recupere les points
        for img in images:
            img_id = img['id']
            ann = next((a for a in data.get('annotations', []) if a['image_id'] == img_id), None)
            
            if ann and 'keypoints' in ann:
                kp = ann['keypoints']
                # Le JSON contient [x, y, v, x, y, v...]. On ne garde que X et Y (on ignore les indices multiples de 3)
                xy_only = [kp[i] for i in range(len(kp)) if i % 3 != 2]
                frames_data.append(xy_only)
            else:
                #si frame vide, rempli de 0
                frames_data.append([0.0] * self.num_features)
                
        #conversion tensor
        tensor_frames = torch.tensor(frames_data, dtype=torch.float32)

        # tensor_frames a une forme : [nb_frames, 42]
        #pn sépare les X (indices pairs 0, 2, 4...) et les Y (indices impairs 1, 3, 5...)
        for i in range(tensor_frames.shape[0]):
            wrist_x = tensor_frames[i, 0].item()
            wrist_y = tensor_frames[i, 1].item()
            
            #si le poignet n'est pas à 0
            if wrist_x != 0 and wrist_y != 0:
                #soustrait wrist_x à toutes les colonnes paires (les X)
                tensor_frames[i, 0::2] = tensor_frames[i, 0::2] - wrist_x
                #soustrait wrist_y à toutes les colonnes impaires (les Y)
                tensor_frames[i, 1::2] = tensor_frames[i, 1::2] - wrist_y



        #(Padding / Truncating)
        seq_len = tensor_frames.shape[0]
        
        if seq_len > self.max_frames:
            tensor_frames = tensor_frames[:self.max_frames, :]
        elif seq_len < self.max_frames:
            padding = torch.zeros(self.max_frames - seq_len, self.num_features)
            tensor_frames = torch.cat((tensor_frames, padding), dim=0)
            
        if self.augment:
            # aug 1 : bruit gaussien sur les coordonnées (en pixels)
            # ATTENTION : la version d'origine faisait `torch.rand_like(x)` qui
            # génère un bruit dans [0, 1]. Mais les features sont des pixels
            # (~+-200), donc ce bruit était 200x trop petit pour avoir un effet
            # réel sur l'entraînement. On passe à du gaussien centré, sigma
            # contrôlable via self.noise_std (~4-5 px = perturbation réaliste).
            # On n'ajoute pas de bruit aux frames de padding (vecteurs nuls)
            # pour éviter de polluer ce qui doit rester "no signal".
            non_pad = (tensor_frames.abs().sum(dim=1, keepdim=True) > 0).float()
            noise = torch.randn_like(tensor_frames) * self.noise_std
            tensor_frames = tensor_frames + noise * non_pad

            # aug 2 : occlusion légère — chaque coord a une petite proba d'être
            # masquée à 0 (simule une landmark mal détectée par MediaPipe).
            mask = (torch.rand_like(tensor_frames) > self.occlusion_prob).float()
            tensor_frames = tensor_frames * mask

        return tensor_frames, label

    
# --------- Test ----------
if __name__ == "__main__":
    # Remplacer par le vrai chemin vers les données
    # data_path = "./mon_dataset_lsf/"
    # dataset = SignLanguageDataset(data_dir=data_path, max_frames=50)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # for inputs, labels in dataloader:
    #     print(f"Batch inputs shape: {inputs.shape}") # Devrait être (32, 50, 266)
    #     print(f"Batch labels shape: {labels.shape}") # Devrait être (32,)
    #     break
    pass