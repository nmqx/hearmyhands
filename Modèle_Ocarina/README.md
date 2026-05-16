# Modèle Ocarina (GRU temporel)

Modèle séquentiel qui classe une séquence de **60 frames × 42 features**
(21 landmarks de main × x, y) en une lettre / un signe. Sert à reconnaître
les signes qui nécessitent du mouvement (J, Z, mots) — là où le MLP
instantané `HmH/letter_classifier.py` ne peut pas.

## Entraîner

```bash
cd Modèle_Ocarina
# Place ton dataset JSON dans ./dataset/ (voir Dataset.py)
python Train.py
# → produit ocarina_gru_v1.pth (gitignoré)
```

## Intégration runtime

Le serveur (`HmH/api.py`) charge automatiquement les poids au démarrage via
`SignClassifier.try_load()`. Aucune action requise s'ils ne sont pas là —
l'endpoint `/sign_predict` répond 503 et la webapp désactive silencieusement
les prédictions temporelles.

Pour activer :

1. Déposer `ocarina_gru_v1.pth` dans ce dossier
2. (Optionnel) Déposer `ocarina_classes.json` — liste JSON des noms de
   classes dans l'ordre des sorties du modèle. Sans ce fichier, le
   classifieur retombe sur A-Z (26 lettres).
3. Redémarrer l'API modèle (`python HmH/api.py`).

`/healthz` indique `sign_classifier: true` quand le modèle est chargé.

## Variables d'environnement

| Variable           | Défaut                                              |
| ------------------ | --------------------------------------------------- |
| `OCARINA_WEIGHTS`  | `Modèle_Ocarina/ocarina_gru_v1.pth`                 |
| `OCARINA_CLASSES`  | `Modèle_Ocarina/ocarina_classes.json`               |
| `SIGN_API_URL`     | `http://127.0.0.1:5001/sign_predict` (côté webapp)  |
| `SIGN_TIMEOUT`     | `2` (secondes)                                      |
