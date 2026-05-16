import cv2
import mediapipe as mp
import time
import json
import numpy as np
import mod_json as htm

'''
Prend en parametre 2 json
W1.json : vient de training.py, est la matrice des poids entre l'entrée et la couche cachée.
W2.json : vient de training.py, est la matrice des poids entre la couche cachée et la sortie.
Permet de reconnaitre une position particuliere d'une main
'''
alphabet = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','Q','R','S','T','U','V','W','X','Y']
with open("Poids/W1.json", "r") as f:
    poids1 = json.load(f)
W1 = np.array(poids1)

with open("Poids/W2.json", "r") as f:
    poids2 = json.load(f)
W2 = np.array(poids2)

with open("Poids/b1.json", "r") as f:
    bias1 = json.load(f)
b1 = np.array(bias1)

with open("Poids/b2.json", "r") as f:
    bias2 = json.load(f)
b2 = np.array(bias2)

def choix(liste) :
    indice = 0
    for i in range(len(liste)) :
        if liste[i] > liste[indice] :
            indice = i
    return indice
        
def simplifier(data):
    simplified_data = []
    if type(data) == list :
        for frame in data:
            # Récupère tous les couples (x, y) dans l’ordre
            keypoints = frame["hands"]["keypoints"]
        numbers = [coord for point in keypoints for coord in (point["x"], point["y"])]
        
        # Ajoute la liste à la structure simplifiée
        simplified_data.append(numbers)
    elif type(data) == dict :
        simplified_data = [coord for point in data['keypoints'] for coord in (point["x"], point["y"])]
    simplified_data = [simplified_data[i]/640 if i%2 == 0 else simplified_data[i]/480 for i in range(len(simplified_data))]
    return simplified_data

class Neural_Network(object):
  def __init__(self):
        
  #Nos paramètres
    self.inputSize = 42 # Nombre de neurones d'entrer
    self.outputSize = 23 # Nombre de neurones de sortie
    self.hiddenSize = 30 # Nombre de neurones cachés

  #Fonction de propagation avant
  def forward(self, X):

    self.z = np.dot(X, self.W1) + self.b1 # Multiplication matricielle entre les valeurs d'entrer et les poids W1
    self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
    self.z3 = np.dot(self.z2, self.W2) + self.b2 # Multiplication matricielle entre les valeurs cachés et les poids W2
    o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
    return o

  # Fonction d'activation
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #Fonction de prédiction
  def predict(self, X_input):
     # Si X_input est une simple liste de 42 nombres → la transformer
        if isinstance(X_input, list) or isinstance(X_input, np.ndarray):
            X_input = np.array(X_input, dtype=float).reshape(1, -1)

        # Calcul de la sortie du réseau
        o = self.forward(X_input)
        classe = []
        for element in o[0] : 
           classe.append(element)

        # Retourne les deux infos
        return classe



NN = Neural_Network()
NN.W1 = W1
NN.W2 = W2
NN.b1 = b1
NN.b2 = b2

cap = cv2.VideoCapture(0)  # 0 ou 1 selon ta caméra
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Caméra", frame)
        cv2.waitKey(2000)  # 2 secondes
    cap.release()
    cv2.destroyAllWindows()
# Global list to store data from all frames during a recording session
recorded_frames = []

# Use consistent snake_case variable names
previous_time = 0
current_time = 0

# IMPORTANT: Camera index is now correctly set to 0 (Internal Camera)
cap = cv2.VideoCapture(0)

# Set low resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("FATAL ERROR: Camera failed to open at index 0. Check if the device is connected or in use.")
    exit()

detector = htm.HandDetector()


# Recording state variable, starts stopped
is_recording = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame during runtime. Exiting.")
        break

    img = detector.find_hands(img)
    # Use the new function to get data for ALL hands and their handedness
    all_hand_info = detector.find_all_positions(img)

    # FPS Calculation
    current_time = time.time()
    if current_time != previous_time:
        fps = 1 / (current_time - previous_time)
    else:
        fps = 0
    previous_time = current_time
    # 1. Convert ALL detected hands to the required JSON format for this FRAME
    # Check if any hand is detected
    if len(all_hand_info) > 0:
        frame_data = detector.landmarks_to_json(all_hand_info)
        frame_entry = {
                    "frame": len(recorded_frames),
                    "timestamp": current_time,
                    "hands": frame_data}
        data = simplifier(frame_data)

        if len(data) == 42 :
            text2 = str(alphabet[choix(NN.predict(data))])
        else : 
            text2 = 'flop'
        cv2.putText(img, text2, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    # ----------------------------------------------------
    # Conditional JSON recording (APPENDING to list)
    # ----------------------------------------------------


    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)


    cv2.imshow("image", img)

    # Keypress handling
    key = cv2.waitKey(1) & 0xFF

    # Quit button ('q')
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()