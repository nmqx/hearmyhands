# Contexte — HearMyHands

Document de contexte du projet à jour au **22 mai 2026**. À lire en
priorité quand on arrive frais sur le repo (humain ou agent IA).

## En une phrase

HearMyHands est un traducteur de Langue des Signes Française en temps
réel dans le navigateur (webcam → texte). Projet **PeiP 2A — Polytech
Nantes**, année 2025–2026, 12 personnes.

- Live : <https://hearmyhands.asia>
- Repo : <https://github.com/nmqx/hearmyhands>

## L'équipe et qui fait quoi

Coordinateur : **Marius DEMONFAUCON** (chef de projet, planning,
correspondant pédagogique).

| Pôle | Responsable | Membres |
| --- | --- | --- |
| Vision (Modèle 1) | **Titouan DESAILLY** | Maxime D'AURIA, Thibault CAPDEVIELLE, François DELAUNAY |
| Traduction (Modèle 2 — MLP/GRU) | **Killian SEVALLE** | Dalyan PENISSON, Nathan DIZY, Louna PEJOT, Marius DEMONFAUCON |
| Interface & Backend | **Maxime D'AURIA** | Kaelig LERAY, François DELAUNAY |
| Communication & DA | **Louna PEJOT** | François DELAUNAY, Arthur PIERRE (CM), Kaelig LERAY, Nestor CORABOEUF |

Outils transverses : Thibault et Maxime co-responsables Git, Maxime
gérant Drive, François admin Discord.

Encadrants Polytech Nantes / LS2N : Hélène Pérennou (pédago),
Matthieu Perreira da Silva (IA), Paul Terrassin (doctorant IA).

## Architecture globale

Le navigateur envoie ses frames webcam en binaire via Socket.IO à un
serveur Flask qui chaîne trois modèles :

```
Browser ──Socket.IO──▶ Flask ──▶ Vision (CSPDarknet53)
                          │              │
                          │              ▼ keypoints corps
                          │           crop poignets
                          │              │
                          │              ▼
                          │       MediaPipe Hands ──▶ 21 landmarks
                          │                              │
                          ├──▶ MLP statique (42 → 50 → 26)
                          │       lettres par frame
                          │
                          └──▶ GRU Ocarina V2 (bidir, 2 layers, hidden=96,
                                  masked mean-pool)
                                  signes temporels sur 45 frames
```

**Modèle 1 — Vision (`HmH/heatnoks/`)**
- CSPDarknet53 + heatmaps + Soft-Argmax → 9 keypoints du haut du corps
- Poids : `HmH/heatnoks/checkpoints/best.pt` (récupérable depuis la
  dernière release GitHub)

**Modèle 2a — MLP statique (`HmH/letter_classifier.py`)**
- 42 features (21 landmarks × x,y, recentrés sur le poignet) → 50 hidden
  ReLU → 26 classes softmax (alphabet complet A-Z)
- Poids : `HmH/Poids/{W1,W2,b1,b2}.json` (versionnés dans git)
- Pré-traitement : rescale 640×480 + centrage wrist (cf
  `HmH/inference.py::_predict_letter`)

**Modèle 2b — GRU Ocarina V2 (`Modèle_Ocarina/`)**
- Architecture : embedding 42→128→96 (ReLU+Dropout) → GRU bidirectionnel
  2 couches hidden 96 → masked mean-pool → LayerNorm+Dropout+Linear(26)
- Poids : `Modèle_Ocarina/ocarina_gru_v2.pth` (~1.2 MB)
- Classes : `Modèle_Ocarina/ocarina_classes.json` (26 lettres)
- Pré-traitement : coords [0,1] → mirror si main gauche → centrage
  wrist → division par taille de main (distance wrist ↔ middle MCP).
  La fonction `Dataset.normalize_frame` est partagée mot pour mot entre
  training et inférence.

## Layout du code

```
hearmyhands-public/
├── hearmyhands/                   App Flask + UI
│   ├── app.py                     Routes HTTP, handler Socket.IO frame,
│   │                              buffer roulant GRU, leaderboard quiz,
│   │                              hall of fame, healthz, monitor, etc.
│   ├── templates/
│   │   ├── base.html              Layout commun (navbar, thème, logo)
│   │   ├── home.html              Hero + pitch + équipe + ODD
│   │   ├── translate.html         /translate : webcam + inférence live
│   │   ├── learn.html             /learn : 3 cartes (Cartes, Reco, Bib.)
│   │   ├── learn_cards.html       /learn/cards : flow Anki
│   │   ├── learn_library.html     /learn/library : statut des 26 lettres
│   │   ├── learn_quiz.html        /learn/quiz : sélection mode + tops
│   │   ├── learn_quiz_game.html   /learn/quiz/<mode> : jeu
│   │   ├── image.html             /image : hall of fame debug (Basic Auth)
│   │   ├── qr.html                /qr : QR vers hearmyhands.asia
│   │   ├── monitor.html           /monitor : CPU / RAM / load
│   │   └── videotest.html         /videotest : test rapide d'une vidéo
│   └── static/
│       ├── css/style.css
│       ├── js/
│       │   ├── script.js          Logique /translate (webcam, Socket.IO,
│       │   │                      MediaPipe Hands client-side, ping,
│       │   │                      seuil dynamique)
│       │   ├── learn.js           /learn/cards (Anki + validation MLP)
│       │   ├── quiz.js            /learn/quiz (jeu + leaderboard SQLite)
│       │   └── script.js          Reveal + menu mobile + /translate
│       ├── images/
│       │   ├── logo.png           Logo light
│       │   └── logo_dark.png      Logo dark (swap auto via data-theme)
│       ├── learn/                 25 vidéos d'apprentissage (A-Z sauf X
│       │                          plus tard, H.264 baseline + faststart)
│       └── hall_of_fame/          Captures debug par IP (gitignored)
│
├── HmH/                           API modèle + scripts d'inférence
│   ├── inference.py               InferenceEngine (chaîne complète)
│   ├── api.py                     Wrapper Flask HTTP (USE_HTTP_MODEL=1)
│   ├── letter_classifier.py       MLP statique (numpy + JSON weights)
│   ├── sign_classifier.py         GRU Ocarina V2 (torch + .pth weights)
│   ├── heatnoks/                  CSPDarknet53 + checkpoints
│   └── Poids/                     Poids MLP versionnés (W1, W2, b1, b2)
│
├── Modèle_Ocarina/                Training du GRU
│   ├── Dataset.py                 Loader + augmentation + normalize_frame
│   ├── Ocarina_GRU.py             Définition du modèle
│   ├── Train.py                   Boucle d'entraînement + diagnostics
│   ├── demo.py                    Démo tkinter standalone (référence
│   │                              pour le buffer + le mask serveur)
│   ├── ocarina_gru_v2.pth         Poids V2 (current)
│   ├── ocarina_gru_v1.pth         Poids V1 (déprécié, archi incompatible)
│   └── ocarina_classes.json       Ordre des classes (A-Z)
│
└── docs/
    ├── flow_models.tex            Doc compilable (3 pages) du flow
    ├── flow_models.pdf
    ├── nginx-deploy-notes.md      Fix nginx (Connection: upgrade)
    └── README.md
```

## Routes HTTP/Socket.IO

| Route | Méthode | Auth | Description |
| --- | --- | --- | --- |
| `/` | GET | — | Home + pitch + équipe |
| `/translate` | GET | — | Page principale, webcam + inférence live |
| `/learn` | GET | — | Landing apprentissage |
| `/learn/cards`, `/learn/cards/<L>` | GET | — | Flow Anki, URL par lettre |
| `/learn/play/<letter>` | GET | — | Wrapper iframe lecture vidéo (no-store) |
| `/learn/library` | GET | — | Statut des 26 lettres |
| `/learn/quiz` | GET | — | Sélection mode + leaderboards |
| `/learn/quiz/<mode>` | GET | — | Jeu, mode ∈ {hardcore, 10sec, survival} |
| `/api/video/<letter>` | GET | — | CDN local des vidéos (max-age 24h) |
| `/api/quiz/score` | POST | — | Submit score, validation côté serveur |
| `/api/quiz/leaderboard/<mode>` | GET | — | Top 10 par mode (SQLite) |
| `/qr` | GET | — | QR code grand format vers le site |
| `/image` | GET | Basic Auth | Hall of fame debug |
| `/api/debug/hall` | GET | Basic Auth | JSON des captures |
| `/monitor`, `/stats` | GET | — | CPU/RAM/load |
| `/healthz` | GET | — | État des modèles |
| `/videotest` | GET | — | Test vidéo direct |
| **Socket.IO `frame`** | — | — | `(image_bytes, flags, ack)` → prédictions |
| **Socket.IO `ping_test`** | — | — | Echo serveur, RTT mesuré côté client |

Auth Basic : password = `WEBCAM_DEBUG_PASS` (var d'env, défaut `admin`).

## Protocole Socket.IO `frame`

Client émet :
```js
socket.emit('frame', jpegBuffer, { predict_sign: true|false }, ack)
```

- `predict_sign: false` → le serveur remplit le buffer GRU (45 frames
  + mask) mais n'appelle PAS l'inférence GRU. Économie CPU énorme en
  mode statique et en idle.
- `predict_sign: true` → un appel GRU tous les `SIGN_EVERY_N = 3`
  frames (≈ 10 Hz à 30 fps webcam) si au moins `MIN_REAL_FRAMES = 5`
  frames avec main détectée sont dans le buffer.

Server répond (ack) :
```json
{
  "skeleton":  [[x,y,vis], ...9],
  "hands":     [[[x,y], ...21], ...],
  "letter":    "A",
  "confidence": 0.84,
  "sign":       "B" | null,
  "sign_confidence": 0.71 | null
}
```

## Modes côté `/translate`

- **Statique (MLP)** : inférence continue, seuil confidence 0.6,
  10 frames stables pour commit au mot. Pas de seuil géométrique.
  `predict_sign: false` envoyé au serveur (GRU pas inférencé).
- **Dynamique (GRU)** : ligne pointillée horizontale sur le canvas
  (slider 10-90 %, persisté). Quand la main est au-dessus du seuil,
  on stocke `pendingSign`. À la descente (debouncée 500 ms), on commit
  la prédiction de plus haute confidence. **Pas** de stabilité 10
  frames (le GRU travaille déjà sur du temporel cohérent).

## Quiz `/learn/quiz`

Trois modes, leaderboards persistés en SQLite (`quiz_scores.db`,
gitignored) :

- **Hardcore** : 10 questions, vidéo joue une fois, écran noir, 3 s
  pour répondre.
- **10 sec** : 10 questions, vidéo en boucle pendant 10 s, mêmes 10 s
  pour répondre.
- **Survie** : illimité. Temps par lettre `1 + 14·exp(-n/12)` s,
  décroît exponentiellement (15 s → ~1 s asymptote). Quand le temps
  passe sous la durée vidéo, accélération du `playbackRate` jusqu'à
  ×3. Erreur = game over.

Soumission au serveur via `POST /api/quiz/score` avec pseudo +
validation côté serveur (anti-abus naïf : bornes par mode, sanitization
du pseudo). Endpoint protégé par timeout client (8 s).

## Hall of fame `/image`

Pour la démo. À chaque session Socket.IO :
- On compte les **signs différents** détectés (transition vers une
  nouvelle lettre, conf ≥ 0.75).
- Au **2e sign différent**, on sauvegarde la dernière frame JPEG dans
  `static/hall_of_fame/<ip>.jpg`.
- Une seule capture par session (flag `hall_saved`).
- Sur reconnexion de la même IP, le fichier est overwrite.
- Page `/image` : grille des captures triées par date.

Auth Basic, password `admin`. Dossier `hall_of_fame/` gitignored.

## Déploiement

VM GCP `34.22.162.219`, user `hmh`, deploy clé Tibo + Maxime déposées
dans `~/.ssh/authorized_keys`.

Stack :
- nginx (sites-available/hmh) → reverse proxy + TLS Let's Encrypt
- gunicorn 1 worker `geventwebsocket.gunicorn.workers.GeventWebSocketWorker`
- Flask app dans `/home/hmh/hearmyhands/.venv`
- Service systemd `hmh.service` avec restart-on-failure

Commande standard de deploy :
```bash
ssh hmh@34.22.162.219 \
  "cd hearmyhands && git pull && sudo systemctl restart hmh"
```

**Piège nginx important** : le block proxy d'origine avait
`proxy_set_header Connection "upgrade"` pour Socket.IO, ce qui fait
timeout les POST normaux (le serveur attend l'upgrade qui ne vient
jamais). Fix : `map $http_upgrade $hmh_connection_upgrade { default
upgrade; "" close; }` puis `proxy_set_header Connection
$hmh_connection_upgrade;`. Cf `docs/nginx-deploy-notes.md`.

Cloudflare devant : règles standard, on lit la vraie IP client via
l'en-tête `CF-Connecting-IP` (utilisée par le hall of fame).

## Outils debug intégrés

- **Bouton Ping** dans `/translate` (mesure RTT live, dot coloré).
- **`/image`** : hall of fame (password `admin`).
- **`/monitor`** : CPU/RAM/load.
- **`/healthz`** : JSON état des modèles.
- **Toggle squelette** : masque le squelette custom (Modèle 1) sans
  toucher aux mains MediaPipe (gardées toujours visibles pour la démo).

## Pièges connus / décisions

- **Buffer GRU** : on pousse une entrée **à chaque frame** dans la
  deque (zéros si pas de main) + un mask binaire en parallèle. La
  timeline est préservée. Aligné mot pour mot sur le buffer de
  `Modèle_Ocarina/demo.py`.
- **`MAX_FRAMES = 45`** côté training doit rester == `SEQ_LEN = 45`
  côté inférence. Vérifié dans `Train.py`, `Dataset.py`,
  `HmH/sign_classifier.py` et `hearmyhands/app.py`.
- **Pré-traitement V2 partagé train/inférence** : `Dataset.py
  ::normalize_frame` est la source de vérité, et `app.py::_normalize_hand`
  est validé bit-à-bit identique (diff < 1e-6 sur les deux
  handedness).
- **Vidéos d'apprentissage** : doivent être H.264 (`avc1`) Constrained
  Baseline + `+faststart`. Les anciennes en `mp4v` (DivX) ne sont
  lues par aucun navigateur moderne. Re-encoder avec
  `ffmpeg -c:v libx264 -profile:v baseline -level 3.1 -pix_fmt
  yuv420p -movflags +faststart -an`.
- **Lecture vidéo /learn/cards** : on passe par un `<iframe
  src="/learn/play/<L>">` parce qu'un `<video>` embedded dans la page
  principale restait noir (probable conflit GPU avec la webcam). Le
  wrapper est servi en `Cache-Control: no-store` et l'URL vidéo
  carry un fingerprint mtime pour buster le cache navigateur après
  un re-encode.
- **`/learn/cards` URL par lettre** : pushState/replaceState côté JS,
  et `contentWindow.location.replace()` (pas `iframe.src =`) pour ne
  pas accumuler d'entrées d'historique iframe sinon le back désync.
- **localStorage cleanup** : pseudo quiz et état Anki survivent entre
  sessions, pas de cleanup automatique. À nettoyer manuellement si
  besoin.
- **Attribution Claude** : tous les commits Co-Authored-By: Claude
  ont été retirés via `git commit --amend` + `force-push` au début.
  `~/.claude/settings.json` désactive maintenant l'attribution
  automatique (`attribution.commit = ""`).

## Roadmap court terme

- Ré-encoder X.mp4 si encore en mp4v (vérifier après chaque ajout
  vidéo via `ffprobe`).
- Audit perf en démo : vérifier que /translate tient 10+ utilisateurs
  simultanés sur la VM (1 worker gevent, GRU on-demand côté client
  aide beaucoup).
- Discussion à avoir : seuils MLP/GRU à ajuster en fonction des
  retours utilisateurs (actuellement 0.6 / 0.7).

## Liens

- Site : <https://hearmyhands.asia>
- Code : <https://github.com/nmqx/hearmyhands>
- Contact équipe : <hearmyhands.polytech@gmail.com>
