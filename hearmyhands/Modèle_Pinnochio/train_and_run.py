import numpy as np
import json
import cv2
import time
import random 
import mod_json as htm # Nécessite mod_json.py pour la détection en temps réel

# --- PARAMÈTRES ET FICHIERS ---
FILE_NEGATIVES = "hand_keypoints.json"  # Exemples de non-L
FILE_POSITIVES = "L.json"               # Exemples de L
W1_FILE = "W1.json"
W2_FILE = "W2.json"

# --- 1. PRÉPARATION DES DONNÉES (Fusion et Simplification) ---
def simplifier(data):
    """Convertit les données brutes de frame en listes de 42 coordonnées (x, y)."""
    simplified_data = []
    for frame in data:
        # Récupère tous les couples (x, y) dans l’ordre
        # S'assure que 'keypoints' est présent et que c'est une liste d'au moins 21 points
        if frame.get("hands") and frame["hands"].get("keypoints"):
             keypoints = frame["hands"]["keypoints"]
             numbers = [coord for point in keypoints for coord in (point["x"], point["y"])]
             if len(numbers) == 42:
                simplified_data.append(numbers)
    return simplified_data

def fusion(file1, file2):
    """Charge, simplifie, fusionne, mélange, et sépare les données d'entrée (X) et les labels (y)."""
    print("--- 1. PRÉPARATION DES DONNÉES ---")
    with open(file1, "r") as f1:
        data_neg = json.load(f1)
    with open(file2, "r") as f2:
        data_pos = json.load(f2)

    data_neg = simplifier(data_neg) # Classe 0 (Non-L)
    data_pos = simplifier(data_pos) # Classe 1 (L)

    # Ajoute les labels (0 ou 1) et fusionne
    labeled_neg = [frame + [0] for frame in data_neg]
    labeled_pos = [frame + [1] for frame in data_pos]
    merged = labeled_neg + labeled_pos

    random.shuffle(merged)

    # Sépare les entrées (X) et les labels (y)
    X = np.array([frame[:-1] for frame in merged], dtype=float)
    y = np.array([frame[-1] for frame in merged], dtype=float).reshape(-1, 1)
    
    # Normalisation robuste (comme dans training.py)
    eps = 1e-8
    max_per_col = np.max(np.abs(X), axis=0) + eps
    X_normalized = X / max_per_col
    
    print(f"Dataset prêt : {X.shape[0]} exemples. L: {len(data_pos)}, Non-L: {len(data_neg)}")
    return X_normalized, y, max_per_col

# --- 2. RÉSEAU NEURONAL ET ENTRAÎNEMENT (Classe NeuralNetwork de training.py) ---
class NeuralNetwork:
    # Utilise la même structure que training.py
    def __init__(self, input_size=42, hidden_size=32, output_size=1, lr=1e-3):
        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.outputSize = output_size
        self.lr = lr

        # Poids init small
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) * 0.01
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) * 0.01
        self.b1 = np.zeros((1, self.hiddenSize))
        self.b2 = np.zeros((1, self.outputSize))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, Xbatch):
        # Propagation avant (identique à training.py)
        z1 = np.dot(Xbatch, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        cache = (Xbatch, z1, a1, z2, a2)
        return a2, cache

    def compute_loss_acc(self, a2, ybatch):
        # Fonction de perte et précision (identique à training.py)
        a2_clipped = np.clip(a2, 1e-9, 1 - 1e-9)
        bce = - (ybatch * np.log(a2_clipped) + (1 - ybatch) * np.log(1 - a2_clipped))
        loss = np.mean(bce)
        preds = (a2 >= 0.5).astype(int)
        acc = np.mean(preds == ybatch)
        return loss, acc

    def backward_and_update(self, cache, ybatch, grad_clip=5.0):
        # Rétropropagation et mise à jour (identique à training.py)
        Xbatch, z1, a1, z2, a2 = cache
        B = Xbatch.shape[0]
        delta2 = (a2 - ybatch) / B
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * self.relu_prime(z1)
        dW1 = np.dot(Xbatch.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2))

        if grad_norm > grad_clip: # Gradient clipping
            scale = grad_clip / (grad_norm + 1e-9)
            dW1 *= scale; dW2 *= scale; db1 *= scale; db2 *= scale

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return grad_norm

    def train(self, X, y, epochs=20000, batch_size=32, print_every=1000):
        # Boucle d'entraînement (identique à training.py)
        N = X.shape[0]
        print(f"--- 2. ENTRAÎNEMENT DU MODÈLE (Epochs: {epochs}, LR: {self.lr}) ---")
        for ep in range(epochs):
            idx = np.random.permutation(N)
            Xs = X[idx]; ys = y[idx]
            grad_norm = 0
            for start in range(0, N, batch_size):
                Xb = Xs[start:start+batch_size]
                yb = ys[start:start+batch_size]
                a2, cache = self.forward(Xb)
                grad_norm = self.backward_and_update(cache, yb)
            
            if ep % print_every == 0 or ep == epochs-1:
                a2_all, _ = self.forward(X)
                loss_all, acc_all = self.compute_loss_acc(a2_all, y)
                print(f"Epoch {ep} | Loss={loss_all:.6f} | Acc={acc_all:.4f}")

    def save_weights(self):
        # Sauvegarde des poids (comme dans training.py)
        with open(W1_FILE, "w") as f:
            json.dump(self.W1.tolist(), f, indent=2)
        with open(W2_FILE, "w") as f:
            json.dump(self.W2.tolist(), f, indent=2)
        print(f"Poids sauvegardés dans {W1_FILE} et {W2_FILE}")

# --- 3. DÉTECTION EN TEMPS RÉEL (L_finder.py simplifié) ---

def run_detector(nn_instance, normalisation_max):
    """Lance la caméra et utilise le réseau entraîné pour la classification en temps réel."""
    print("--- 3. DÉTECTION EN TEMPS RÉEL (Appuyez sur 'q' pour quitter) ---")
    detector = htm.HandDetector(detection_con=0.5, track_con=0.5)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("FATAL ERROR: Impossible d'ouvrir la caméra à l'index 0.")
        return

    previous_time = 0

    def simplify_live_data(frame_data):
        """Simplifie les données d'une seule frame pour l'entrée du NN (42 coords)."""
        if frame_data and frame_data.get('keypoints'):
            return [coord for point in frame_data['keypoints'] for coord in (point["x"], point["y"])]
        return []

    def predict_live(X_input):
        """Prédit la classe et la probabilité à partir d'une liste de 42 coordonnées."""
        if len(X_input) != 42:
            return 'Main Incomplète', 0.0
        
        # 1. Mise en forme et Normalisation (comme dans training.py)
        X_input_array = np.array(X_input, dtype=float).reshape(1, -1)
        X_input_norm = X_input_array / normalisation_max
        
        # 2. Propagation avant pour obtenir la probabilité
        a2, _ = nn_instance.forward(X_input_norm)
        prob = a2[0][0]  # probabilité (valeur entre 0 et 1)

        # 3. Décision finale
        classe = 'L DETECTÉ' if prob >= 0.5 else 'Pas un L'

        return classe, prob

    while True:
        success, img = cap.read()
        if not success:
            print("Échec de la lecture de la frame.")
            break

        img = detector.find_hands(img)
        all_hand_info = detector.find_all_positions(img)

        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if current_time != previous_time else 0
        previous_time = current_time

        # Prédiction si une main est détectée
        prediction_text = "PAS DE MAIN"
        if len(all_hand_info) > 0:
            frame_data = detector.landmarks_to_json(all_hand_info)
            live_data = simplify_live_data(frame_data)
            
            if live_data:
                classe, prob = predict_live(live_data)
                prediction_text = f"{classe} ({prob:.2f})"
                status_color = (0, 255, 0) if classe == 'L DETECTÉ' else (0, 0, 255)
            else:
                 status_color = (255, 255, 0)
                 prediction_text = "Main incomplète (42 points requis)"

            cv2.putText(img, prediction_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)


        # Affichage FPS
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow("Detection en Temps Reel", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- CODE PRINCIPAL (Exécution Séquentielle) ---
if __name__ == "__main__":
    
    # 1. Préparation des données
    X, y, max_per_col = fusion(FILE_NEGATIVES, FILE_POSITIVES)
    
    # 2. Entraînement du modèle
    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=32, lr=1e-3)
    nn.train(X, y, epochs=15000, batch_size=32, print_every=1500)
    nn.save_weights()
    
    # 3. Détection en temps réel
    run_detector(nn, max_per_col)