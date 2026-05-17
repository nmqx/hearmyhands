# HearMyHands

### 🌐 Site en ligne — **<https://hearmyhands.asia>**

> Projet **PeiP 2A — Polytech** · Année 2025-2026
> Traducteur de Langue des Signes Française (LSF) en temps réel via webcam.
> Contact équipe : **hearmyhands.polytech@gmail.com**

**Accès direct :**
- [`/translate`](https://hearmyhands.asia/translate) — traduction LSF temps réel via la webcam
- [`/learn`](https://hearmyhands.asia/learn) — mode apprentissage Anki des 26 lettres

## Crédits

**Équipe HearMyHands** (12 membres) :

- **Marius DEMONFAUCON** : Coordinateur / Correspondant
- Dalyan PENISSON · Thibault CAPDEVIELLE · Maxime D'AURIA · Killian SEVALLE
- Titouan DESAILLY · Arthur PIERRE · Nestor CORABOEUF · Louna PEJOT
- François DELAUNAY · Kaelig LERAY · Nathan DIZY

Organisée en trois pôles techniques :

| Pôle | Rôle |
| --- | --- |
| **Vision** (`image_to_Squelette` / *Gepetto*) | Détection du squelette corps + mains |
| **Traduction** (`squelette_to_mot` / *Pinocchio*) | Classification des signes en lettres/mots |
| **Application & UI** | Interface web, intégration, portage |

**Encadrants — Laboratoire LS2N / Polytech Nantes**

- **Matthieu PERREIRA DA SILVA** — Enseignant-chercheur LS2N, supervision IA
- **Paul TERRASSIN** — Doctorant, imagerie / IA
- **Tristan GOMEZ** — Doctorant, vision par ordinateur

**Outils & données**

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Google) — détection des landmarks de la main
- Dataset [How2Sign](https://how2sign.github.io/) + dataset interne tourné par l'équipe
- Ressources de calcul prêtées par le **laboratoire LS2N**

Le projet s'inscrit dans les Objectifs de Développement Durable de l'ONU :
**ODD 10** (Réduction des inégalités) et **ODD 4** (Éducation de qualité).

---

## Le projet

Un navigateur envoie les frames de la webcam via WebSocket à une petite app
Flask, qui les transmet à un modèle PyTorch retournant les keypoints du
corps (ResNet-50 + heatmaps) et les landmarks des mains (MediaPipe). Un MLP
prédit ensuite la lettre LSF à partir des landmarks de la main détectée.

```
Browser ──Socket.IO──▶ webapp ──HTTP──▶ model API ──▶ keypoints + hands + letter
```

## Layout

```
hearmyhands/   App Flask (UI + transport WebSocket)
HmH/           API modèle + scripts d'entraînement / inférence
```

## Setup

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Puis récupérer `best.pt` depuis la [dernière release](../../releases/latest)
et le placer dans `HmH/heatnoks/checkpoints/`.

## Run

Un seul processus (par défaut, modèle chargé en in-process pour la perf) —
**même commande sur les trois OS** :

```bash
python hearmyhands/app.py    # webapp + modèle sur :5000
```

Ouvrir <http://localhost:5000/translate>, autoriser la caméra, et cliquer sur
*Lancer Traduction*.

### Mode service séparé (optionnel)

Si tu veux faire tourner le modèle sur une autre machine, lance les deux
services et active le backend HTTP. La syntaxe des variables d'env diffère
entre shells :

**Linux / macOS (bash, zsh)**

```bash
python HmH/api.py                           # terminal 1 — modèle sur :5001
USE_HTTP_MODEL=1 python hearmyhands/app.py  # terminal 2 — webapp sur :5000
```

**Windows (PowerShell)**

```powershell
python HmH\api.py                                              # terminal 1
$env:USE_HTTP_MODEL = "1"; python hearmyhands\app.py           # terminal 2
```

**Windows (cmd.exe)**

```cmd
python HmH\api.py
set USE_HTTP_MODEL=1 && python hearmyhands\app.py
```

`HmH/inference.py` reste l'unique source de vérité pour la logique
d'inférence — `HmH/api.py` n'est qu'une fine couche Flask par-dessus.

### Config (variables d'env)

| Variable          | Défaut                                |
| ----------------- | ------------------------------------- |
| `USE_HTTP_MODEL`  | `0` (in-process)                      |
| `MODEL_API_URL`   | `http://127.0.0.1:5001/model_predict` |
| `SIGN_API_URL`    | `http://127.0.0.1:5001/sign_predict`  |
| `MODEL_TIMEOUT`   | `5`                                   |
| `PORT`            | `5000` (web) / `5001` (modèle)        |
| `USE_AMP`         | `0` (fp16 opt-in — souvent plus lent) |

## Inférence standalone

```bash
# fichier vidéo
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source video.mp4

# webcam
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source 0
```
