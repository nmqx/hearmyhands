import torch
import json
from pathlib import Path
import time

poids1 = "Poids/W1.json"
poids2 = "Poids/W2.json" #poids à évaluer
# Lecture des fichiers JSON
biais1 = "Poids/b1.json"
biais2 = "Poids/b2.json"
eval_label = "Tests/eval_label.json"
eval_keypoints = "Tests/eval_keypoints.json"
# Lecture des deux fichiers JSON
with open(eval_label, "r") as f1:
    eval_label = json.load(f1)

with open(eval_keypoints, "r") as f2:
    eval_keypoints = json.load(f2)

with open(poids1, "r") as f:
    poids1 = json.load(f)


with open(poids2, "r") as f:
    poids2 = json.load(f)

with open(biais1, "r") as f:
    biais1 = json.load(f)

with open(biais2, "r") as f:
    biais2 = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

W1 = torch.tensor(poids1, dtype=torch.float32).to(device)
W2 = torch.tensor(poids2, dtype=torch.float32).to(device)
b1 = torch.tensor(biais1, dtype=torch.float32).to(device)
b2 = torch.tensor(biais2, dtype=torch.float32).to(device)

alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','Q','R','S','T','U','V','W','X','Y']
data_set = torch.tensor(eval_keypoints, dtype=torch.float32).to(device)
label = torch.tensor(eval_label, dtype=torch.long).to(device)

# Normalisation des données pour correspondre à l'échelle de l'entraînement (0 à 1)
# On utilise le max du dataset d'évaluation comme approximation du max de l'entraînement
max_values = torch.max(data_set, dim=0).values
data_set = data_set / max_values

volume = len(data_set)


def choix(liste) :
    indice = 0
    for i in range(len(liste)) :
        if liste[i] > liste[indice] :
            indice = i
    return indice

class Neural_Network(object):
  def __init__(self):
        
  #Nos paramètres
    self.inputSize = 42 # Nombre de neurones d'entrer
    self.outputSize = 23 # Nombre de neurones de sortie
    self.hiddenSize = 20 # Nombre de neurones cachés

  #Nos poids
    self.W1 = torch.randn(self.inputSize, self.hiddenSize, device=device) # (42*50) Matrice de poids entre les neurones d'entrer et cachés
    self.W2 = torch.randn(self.hiddenSize, self.outputSize, device=device) # (50x1) Matrice de poids entre les neurones cachés et sortie
    self.b1 = torch.zeros(1, self.hiddenSize, device=device)
    self.b2 = torch.zeros(1, self.outputSize, device=device)


  #Fonction de propagation avant
  def forward(self, X):

    self.z = torch.matmul(X, self.W1) + self.b1 # Multiplication matricielle entre les valeurs d'entrer et les poids W1
    self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
    self.z3 = torch.matmul(self.z2, self.W2) + self.b2 # Multiplication matricielle entre les valeurs cachés et les poids W2
    o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
    return o

  # Fonction d'activation
  def sigmoid(self, s):
    return 1/(1+torch.exp(-s))

  # Dérivée de la fonction d'activation
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #Fonction de prédiction
  def predict(self, X_input):
     # Si X_input est une simple liste de 42 nombres → la transformer
        if isinstance(X_input, list):
            X_input = torch.tensor(X_input, dtype=torch.float32, device=device)
        
        if X_input.dim() == 1:
            X_input = X_input.unsqueeze(0)

        # Calcul de la sortie du réseau
        o = self.forward(X_input)
        classe = []
        for element in o[0] : 
           classe.append(element)

        # Retourne les deux infos
        return choix(classe)


NN = Neural_Network()
NN.W1 = W1
NN.W2 = W2
NN.b1 = b1
NN.b2 = b2

def evaluation(volume) :
    result = 0
    dico = {lettre : {'volume' : 0, 'winrate' : 0, 'faux positifs' : 0, 'faux negatifs' : 0, 'vrai positifs' : 0, 'detail faux negatifs' : {lettre2 : 0 for lettre2 in alphabet if lettre2 != lettre}, 'detail faux positifs' : {lettre2 : 0 for lettre2 in alphabet if lettre2 != lettre}} for lettre in alphabet}
    for i in range(volume):
        
        verite = alphabet[label[i].item()]
        guess = alphabet[NN.predict(data_set[i])]
        dico[verite]['volume'] += 1
        if guess == verite :
            dico[verite]['vrai positifs'] += 1
            result = result  + 1
        else : 
            dico[verite]['faux positifs'] += 1
            dico[guess]['faux negatifs'] += 1
            dico[guess]['detail faux positifs'][verite] += 1
            dico[verite]['detail faux negatifs'][guess] += 1
    for lettre in alphabet :
        dico[lettre]['winrate'] = (dico[lettre]['vrai positifs'] - dico[lettre]['faux negatifs'])/dico[lettre]['volume']
    dico.update({'volume eval' : volume, 'volume training' : volume, 'winrate' : result/volume})
    dico.update({'W1' : NN.W1.cpu().tolist(), 'W2' : NN.W2.cpu().tolist(), 'b1' : NN.b1.cpu().tolist(), 'b2' : NN.b2.cpu().tolist()})
    #dico.update({'hiden size' : NN.hiddenSize, 'learning rate' : NN.learning_rate, 'Nb iteration' : NN.Nb_it})
    return  dico

dossier_destination = Path("Evals")
dossier_destination.mkdir(parents=True, exist_ok=True)
path = Path(f"eval_label_{int(time.time())}.json")
OUTPUT_JSON_EVAL = dossier_destination / path


with open(OUTPUT_JSON_EVAL, "w") as f:
    json.dump(evaluation(volume), f, indent=4)
