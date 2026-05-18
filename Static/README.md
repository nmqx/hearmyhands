# 🤟 Reconnaissance de la Langue des Signes Française (LSF) en Temps Réel

Ce projet est une solution complète d'Intelligence Artificielle permettant d'apprendre et de reconnaître les lettres de l'alphabet (A-Z) en Langue des Signes Française via une webcam, en temps réel.

Il couvre l'intégralité d'un pipeline de Machine Learning : de la création du dataset personnalisé jusqu'à l'inférence optimisée en direct.

---

## ✨ Fonctionnalités

- **Création de Dataset sur mesure :** Enregistrement de vidéos et extraction automatique des squelettes de la main via **MediaPipe**.
- **Robustesse spatiale :** Les coordonnées 3D sont normalisées et centrées par rapport au poignet pour rendre la détection invariante à la position à l'écran.
- **Entraînement ultra-rapide :** Utilisation de **PyTorch** pour l'entraînement d'un Perceptron Multicouche (MLP) avec optimisation par *CrossEntropyLoss*.
- **Inférence Temps Réel (Légère) :** L'application finale de détection webcam fonctionne uniquement avec **NumPy** et **OpenCV** (pas besoin de charger PyTorch en production !).
- **Lissage temporel :** Utilisation d'un buffer prédictif pour éviter les clignotements et stabiliser la lettre affichée à l'écran.

---

## ⚙️ Installation

1. Clonez ce dépôt sur votre machine.
2. Installez les dépendances requises via `pip` :

```bash
pip install -r requirements.txt
```
*(Note : Pour entraîner un nouveau modèle, vous aurez également besoin d'installer PyTorch. Pour l'inférence seule, les librairies ci-dessus suffisent).*

---

## 🚀 Pipeline d'utilisation (Étape par Étape)

### Étape 1 : Collecte des Données
Générez vos propres données en vous filmant. Deux choix s'offrent à vous :
- **`L_to_json.py` / `squel_to_json.py` :** Enregistrement en direct via la webcam. Appuyez sur `r` pour démarrer/arrêter l'enregistrement et `q` pour quitter.
- **`dataset.py` :** Permet de traiter tout un dossier de vidéos (`.mp4` / `.mov`) pré-enregistrées. Les vidéos sont traitées en multi-threading pour extraire les données MediaPipe.

*Les données brutes seront sauvegardées dans `Data_sets/data/` au format `.json`.*

### Étape 2 : Préparation et Tri
Exécutez le script de tri pour nettoyer les données et préparer le réseau de neurones :
```bash
python tris.py
```
**Ce que fait ce script :**
1. Supprime l'axe Z (inutile de face).
2. Centre les 21 points de la main par rapport au poignet (42 coordonnées relatives).
3. Effectue un **Stratified Split (80/20)** pour répartir équitablement les images d'entraînement et d'évaluation pour chaque lettre.

### Étape 3 : Entraînement du Modèle
Lancez le script d'optimisation PyTorch :
```bash
python training_optimized.py
```
Le script va :
- Charger les données et entraîner un réseau de neurones (MLP).
- Évaluer les performances sur le set de validation (données inconnues du modèle).
- Exporter l'intelligence du réseau (Matrices de Poids et Biais) dans le dossier `Poids_centered/`.
- Sauvegarder un rapport complet dans `Evals/`.

### Étape 4 : Utilisation en Temps Réel (Webcam)
Une fois le modèle entraîné, lancez simplement la prédiction en direct :
```bash
python predict.py
```
Placez votre main devant la caméra ! L'IA affichera la lettre détectée en vert. Si l'IA a un doute entre deux lettres (écart de probabilité très faible), la lettre s'affichera en orange.

---

## 📊 Analyse des Performances

Vous pouvez analyser les erreurs de votre modèle pour savoir quelles lettres ré-enregistrer :
- Le fichier **`import string.py`** lit les rapports d'évaluation et génère une **Matrice de Confusion** visuelle grâce à `seaborn` et `matplotlib`.

---

## 📂 Structure du Projet

```text
📁 Projet2A/
├── dataset.py                # Traitement de vidéos .mp4 en batch
├── L_to_json.py              # Enregistrement webcam interactif
├── mod_json.py               # Wrapper MediaPipe (Détection de main)
├── tris.py                   # Nettoyage et Split 80/20 (Data Prep)
├── training_optimized.py     # Entraînement IA (PyTorch)
├── predict.py                # Inférence IA Temps Réel (Webcam + Numpy)
├── import string.py          # Génération de graphiques d'analyse
├── requirements.txt          # Dépendances Python
│
├── Data_sets/                # [Généré] Données brutes de MediaPipe
├── Tests_centered/           # [Généré] Données prêtes pour l'entraînement
├── Poids_centered/           # [Généré] Matrices W1, W2, b1, b2 de l'IA
└── Evals/                    # [Généré] Rapports de performances JSON
```

---

## 🛠️ Architecture du Réseau (MLP)

- **Entrée :** 42 Neurones (21 points × X, Y centrés).
- **Couche Cachée :** 50 Neurones (Activation *ReLU*).
- **Sortie :** 26 Neurones (1 par lettre de l'alphabet, Activation *Softmax*).

*(Par défaut, l'entraînement s'effectue avec la Loss `CrossEntropyLoss` et l'optimiseur `SGD` sur PyTorch).*