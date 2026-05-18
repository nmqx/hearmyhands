import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import time
import os

BASE_DIR = Path(__file__).resolve().parent

alphabet = [i * [0] + [1] + (25 - i) * [0] for i in range(26)]
training_label = BASE_DIR / "Tests_centered" / "training_label.json"
training_keypoints = BASE_DIR / "Tests_centered" / "training_keypoints.json"
eval_label_path = BASE_DIR / "Tests_centered" / "eval_label.json"
eval_keypoints_path = BASE_DIR / "Tests_centered" / "eval_keypoints.json"

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

print("Chargement du dataset d'évaluation (Crash-Test) depuis les fichiers séparés...")
with open(eval_label_path, "r") as f:
    y_eval_list = json.load(f)

with open(eval_keypoints_path, "r") as f:
    X_eval_list = json.load(f)

data_set = torch.tensor(X_eval_list, dtype=torch.float32).to(device)
label = torch.tensor(y_eval_list, dtype=torch.long).to(device)
volume_eval = len(data_set)
print(f"-> {volume_eval} frames d'évaluation chargées (données distinctes de l'entraînement).")

# === LECTURE DU DATASET D'ENTRAÎNEMENT ===
with open(training_label, "r") as f1:
    data1 = json.load(f1)

for i in range(len(data1)) :
    data1[i] = alphabet[data1[i]]

with open(training_keypoints, "r") as f2:
    data2 = json.load(f2)

X = torch.tensor(data2, dtype=torch.float32).to(device) 
y = torch.tensor(data1, dtype=torch.float32).reshape(-1, 26).to(device) 
volume_train = len(X)
print(f"-> {volume_train} frames d'entraînement chargées.")


# === NOTRE RÉSEAU DE NEURONES OPTIMISÉ (Architecture Native PyTorch) ===
class PyTorchMLP(nn.Module):
    def __init__(self, input_size=42, hidden_size=30, output_size=26, activation_name="sigmoid"):
        super(PyTorchMLP, self).__init__()
        
        # Couches linéaires
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Fonction d'activation de la couche cachée
        if activation_name == "relu":
            self.activation = nn.ReLU()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
            
        # Activation finale (Softmax pour classification multiclasse)
        self.output_activation = nn.Softmax(dim=1)

        # Initialisation manuelle pour correspondre exactement à ton ancienne version
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(hidden_size, input_size) / (input_size ** 0.5))
            self.fc1.bias.fill_(0)
            self.fc2.weight.copy_(torch.randn(output_size, hidden_size) / (hidden_size ** 0.5))
            self.fc2.bias.fill_(0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x))
        return x


# === EVALUATION DETAILLEE ===
lettres_alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
loss_fn_eval = nn.MSELoss()

def evaluation_detaillee(model, data_eval, label_eval, volume_eval):
    result = 0
    with torch.no_grad():
        outputs = model(data_eval)
        
        y_eval_onehot = torch.zeros(outputs.size(), device=device)
        y_eval_onehot.scatter_(1, label_eval.unsqueeze(1), 1.0)
        val_loss = loss_fn_eval(outputs, y_eval_onehot).item()
        
        dico = {lettre: {
            'volume': 0, 'winrate': 0, 'faux positifs': 0, 'faux negatifs': 0, 'vrai positifs': 0,
            'detail faux negatifs': {l: 0 for l in lettres_alphabet if l != lettre},
            'detail faux positifs': {l: 0 for l in lettres_alphabet if l != lettre}
        } for lettre in lettres_alphabet}

        predictions = torch.argmax(outputs, dim=1)
        for i in range(volume_eval):
            verite = lettres_alphabet[label_eval[i].item()]
            guess = lettres_alphabet[predictions[i].item()]
            
            dico[verite]['volume'] += 1
            if guess == verite:
                dico[verite]['vrai positifs'] += 1
                result += 1
            else:
                dico[verite]['faux positifs'] += 1
                dico[guess]['faux negatifs'] += 1
                dico[guess]['detail faux positifs'][verite] += 1
                dico[verite]['detail faux negatifs'][guess] += 1
                
    for lettre in lettres_alphabet:
        if dico[lettre]['volume'] > 0:
            dico[lettre]['winrate'] = (dico[lettre]['vrai positifs'] - dico[lettre]['faux negatifs']) / dico[lettre]['volume']
            
    dico.update({
        'volume eval': volume_eval,
        'winrate global': result / volume_eval,
        'validation loss': val_loss
    })
    return dico

# === PARAMÈTRES POUR LE GRID SEARCH ===
dossier_destination = BASE_DIR / "Poids_centered"
dossier_destination.mkdir(parents=True, exist_ok=True)

learning_rates = [0.1, 0.5, 0.9]
hidden_sizes = [50]
iterations = [50000]
activations = ["relu"]

results_log = []
best_accuracy = 0
best_base_model = None
best_stats = None
best_config = None

criterion = nn.MSELoss(reduction='sum')

print(f"\n--- Début du Grid Search Accéléré (Mathématiques corrigées) ---")
total_tests = len(learning_rates) * len(hidden_sizes) * len(iterations) * len(activations)
test_actuel = 0

for lr in learning_rates:
    for hs in hidden_sizes:
        for act in activations:
            for it in iterations:
                test_actuel += 1
                print(f"Test {test_actuel}/{total_tests} | LR={lr}, Couche={hs}, Act={act}, Iter={it}", end="")
                
                # Création du modèle et envoi au GPU
                base_model = PyTorchMLP(input_size=42, hidden_size=hs, output_size=26, activation_name=act).to(device)
                
                # Compilation PyTorch 2.0+ (Ignoré si non-supporté sous Windows)
                try:
                    model = torch.compile(base_model) if os.name != 'nt' else base_model
                except:
                    model = base_model
                
                # Optimiseur Gradient Descent
                optimizer = optim.SGD(model.parameters(), lr=lr)
                
                # Boucle d'entraînement ultra-rapide (Natif)
                model.train()
                for epoch in range(it):
                    optimizer.zero_grad()
                    outputs = model(X)
                    # On divise uniquement par le volume d'entraînement, exactement comme ton code manuel
                    loss = criterion(outputs, y) / volume_train
                    loss.backward()
                    optimizer.step()
                
                # Évaluation
                model.eval()
                stats = evaluation_detaillee(model, data_set, label, volume_eval)
                acc = stats['winrate global']
                val_loss = stats['validation loss']
                
                print(f" -> Acc: {acc*100:.2f}% | Loss: {val_loss:.4f}")
                
                # Extraction des matrices pour ce test
                current_W1 = base_model.fc1.weight.t().cpu().tolist()
                current_W2 = base_model.fc2.weight.t().cpu().tolist()
                current_b1 = base_model.fc1.bias.unsqueeze(0).cpu().tolist()
                current_b2 = base_model.fc2.bias.unsqueeze(0).cpu().tolist()
                
                results_log.append({
                    "test_name": f"Test {test_actuel}/{total_tests} | LR={lr}, Couche={hs}, Act={act}, Iter={it}",
                    "learning_rate": lr, "hidden_size": hs, "iterations": it,
                    "activation": act, "accuracy": acc, "val_loss": val_loss,
                    "statistiques_detaillees": stats,
                    "W1": current_W1, "W2": current_W2, "b1": current_b1, "b2": current_b2
                })
                
                if acc > best_accuracy or best_base_model is None:
                    best_accuracy = acc
                    best_base_model = base_model
                    best_stats = stats
                    best_config = {"activation": act, "hiddenSize": hs}

# === SAUVEGARDE FINALE ===
OUTPUT_JSON_RESULTS = dossier_destination / "hyperparameters_results.json"
with open(OUTPUT_JSON_RESULTS, "w") as f:
    json.dump(results_log, f, indent=4)

print(f"\n=== Recherche terminée ===")
print(f"Meilleur winrate global : {best_accuracy*100:.2f}%")

# IMPORTANT : Extraction et transposition des matrices pour L_finder.py
# PyTorch sauvegarde en [Sortie, Entrée], ton code veut [Entrée, Sortie], d'où le .t()
poids1 = best_base_model.fc1.weight.t().cpu().tolist()
poids2 = best_base_model.fc2.weight.t().cpu().tolist()
bias1 = best_base_model.fc1.bias.unsqueeze(0).cpu().tolist()
bias2 = best_base_model.fc2.bias.unsqueeze(0).cpu().tolist()

with open(dossier_destination / "W1.json", "w") as f:
    json.dump(poids1, f, indent=4)
with open(dossier_destination / "W2.json", "w") as f:
    json.dump(poids2, f, indent=4)
with open(dossier_destination / "b1.json", "w") as f:
    json.dump(bias1, f, indent=4)
with open(dossier_destination / "b2.json", "w") as f:
    json.dump(bias2, f, indent=4)
with open(dossier_destination / "config.json", "w") as f:
    json.dump(best_config, f, indent=4)

dossier_evals = BASE_DIR / "Evals"
dossier_evals.mkdir(parents=True, exist_ok=True)
with open(dossier_evals / f"eval_metrics_{int(time.time())}.json", "w") as f:
    json.dump(best_stats, f, indent=4)

# Sauvegarde du rapport global d'évaluation détaillée (Remplace evaluation_complete.py)
all_evaluations = []
for r in results_log:
    all_evaluations.append({
        "test_name": r["test_name"],
        "hyperparameters": {
            "learning_rate": r["learning_rate"],
            "hidden_size": r["hidden_size"],
            "iterations": r["iterations"],
            "activation": r["activation"]
        },
        "validation_loss_entrainement": r["val_loss"],
        "accuracy": r["accuracy"],
        "statistiques_detaillees": r["statistiques_detaillees"]
    })

output_eval_file = dossier_evals / f"evaluations_completes_historique_{int(time.time())}.json"
with open(output_eval_file, "w") as f:
    json.dump(all_evaluations, f, indent=4)

print(f"Rapport global (tous les modèles) sauvegardé dans : {output_eval_file}")
