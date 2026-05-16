# HearMyHands

> Projet **PeiP 2A — Polytech** · Année 2025-2026
> Traducteur de Langue des Signes Française (LSF) en temps réel via webcam.
> Contact équipe : **hearmyhands.polytech@gmail.com**

## Crédits

**Équipe HearMyHands** (12 membres) :

- **Marius DEMONFAUCON** — Coordinateur / Correspondant
- Dalyan PENISSON · Thibault CAPEDEVIELLE · Maxime D'AURIA · Killian SEVALLE
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

```bash
pip install -r requirements.txt
```

Récupérer `best.pt` depuis la [dernière release](../../releases/latest) et le
placer dans `HmH/heatnoks/checkpoints/`.

## Run

Deux processus :

```bash
python HmH/api.py            # API modèle sur :5001
python hearmyhands/app.py    # webapp sur :5000
```

Ouvrir <http://localhost:5000/translate>, autoriser la caméra, et cliquer sur
*Lancer Traduction*.

### Config (variables d'env)

| Variable        | Défaut                                |
| --------------- | ------------------------------------- |
| `MODEL_API_URL` | `http://127.0.0.1:5001/model_predict` |
| `MODEL_TIMEOUT` | `5`                                   |
| `PORT`          | `5000` (web) / `5001` (modèle)        |

## Inférence standalone

```bash
# fichier vidéo
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source video.mp4

# webcam
python HmH/heatnoks/inference_video.py --ckpt HmH/heatnoks/checkpoints/best.pt --source 0
```
