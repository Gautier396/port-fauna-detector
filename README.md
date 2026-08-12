# Détecteur de faune marine du port

Pipeline de vision par ordinateur pour détecter et identifier automatiquement
les poissons (+ oursins) observés dans un port méditerranéen, à partir de
vidéos de plongée GoPro, avec un registre d'individus qui évite qu'un même
animal revu plusieurs fois soit compté en double.

## Statut actuel

- **Données d'entraînement** : pivot vers **SFISHTRACK** (dataset externe,
  vidéos sous-marines annotées COCO — segmentation d'instance + tracking
  multi-objet), remplace l'ancien pipeline iNaturalist (retiré, cf. plus
  bas). **Téléchargement pas encore terminé** (26 Go, bloqué côté Google
  Drive — "too many users have viewed or downloaded this file recently").
- **Classes labellisées par SFISHTRACK** : à confirmer une fois le fichier
  réel accessible — la classe COCO connue à ce jour est une classe unique
  `fish` (pas de distinction par espèce), donc tout mappe vers `poisson`
  (classe 0) pour l'instant. Voir `configs/sfishtrack_species_map.yaml`.
- **Conséquence directe** : les 24 classes d'espèces (1-24) et les 2
  classes d'oursins (25-26) de `configs/species.yaml` n'ont plus AUCUN
  mécanisme d'acquisition de données actif (l'ancien pipeline iNaturalist,
  seule source qui les alimentait, a été retiré). Seule la classe `poisson`
  (0) a une source active tant que SFISHTRACK reste mono-classe.
- **Prétraitement** : `src/preprocess.py` corrige la dominante bleu/vert de
  l'eau (balance des blancs gray-world, proportionnelle à la dominante
  détectée — n'altère pas les images déjà neutres/chaudes). Appliqué en
  place sur le dataset ET à chaque frame vidéo à l'inférence (`track.py`,
  `app.py`), pour que l'entraînement et l'inférence voient la même
  distribution de couleurs.
- **Modèles entraînés** (`models/portfauna_v1.pt`, `portfauna_v2.pt`) :
  issus de l'ancien pipeline iNaturalist, **obsolètes** — un nouvel
  entraînement sur SFISHTRACK est nécessaire avant d'utiliser `app.py` en
  confiance.
- **Pipeline vidéo** (tracking → embeddings → registre → export) :
  jamais exécuté sur une vraie vidéo du port.

## Architecture

```
SFISHTRACK (COCO, par vidéo) ──► convert_sfishtrack.py ──► images + boîtes YOLO (data/sfishtrack/)
                                                                          │
                                                                          ▼
                                                         train.py ──► models/<name>.pt

Vidéo GoPro ──► track.py (détection+tracking) ──┬─► embeddings.py ──► registry.py ──► export.py
                                                  └─► review_queue.py (si confiance basse) ──► resolve_review.py
```

`pipeline.py` enchaîne tracking → embeddings → registre pour une vidéo.
`app.py` : voir Structure.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
src/
  convert_sfishtrack.py     <- convertit les annotations COCO SFISHTRACK (par vidéo) en labels YOLO
  preprocess.py              <- correction couleur sous-marine (dataset en place + frames vidéo à l'inférence)
  train.py                  <- entraînement YOLO -> models/<name>.pt
  track.py                  <- détection + ByteTrack + crops (marque needs_review)
  review_queue.py           <- file de vérification (tracks à confiance basse)
  resolve_review.py         <- referme la file une fois l'espèce confirmée
  embeddings.py             <- embedding ResNet18 par track
  registry.py               <- registre SQLite + anti-doublon (espèce + fenêtre temporelle)
  export.py                 <- CSV, stats
configs/
  species.yaml                    <- 27 classes cibles (PROVISOIRE)
  sfishtrack_species_map.yaml     <- catégorie COCO SFISHTRACK -> classe species.yaml
  data_sfishtrack.yaml            <- généré par convert_sfishtrack.py, format Ultralytics
data/sfishtrack/              <- images + labels YOLO générés par convert_sfishtrack.py
data/external/sfishtrack/     <- zip SFISHTRACK brut (téléchargement)
outputs/                     <- registry.db, tracks/, crops/, embeddings/, review_queue/
pipeline.py                  <- orchestrateur bout-en-bout (une vidéo -> registre)
app.py                       <- interface glisser-déposer, test vidéo avec le dernier modèle (port 7860)
view_labels.py                <- revue en lecture seule des labels générés (port 7862)
```

## Utilisation

```bash
# Une fois le dataset SFISHTRACK téléchargé (data/external/sfishtrack/) :
# 1. Voir les catégories réelles + couverture du mapping (aucune écriture)
python src/convert_sfishtrack.py --dataset-root data/external/sfishtrack/dataset --check-only

# 2. Convertir (une fois configs/sfishtrack_species_map.yaml rempli/vérifié)
python src/convert_sfishtrack.py --dataset-root data/external/sfishtrack/dataset --output data/sfishtrack

# Correction couleur sous-marine (en place, idempotent) :
python src/preprocess.py --images-dir data/sfishtrack

# Entraînement :
python src/train.py --data configs/data_sfishtrack.yaml --name portfauna_v3

# Pipeline vidéo complet :
python pipeline.py --video data/raw/plongee1.mp4 --model models/portfauna_v3.pt

# Test rapide glisser-déposer :
python app.py

# Revue des labels générés :
python view_labels.py
```

## Points ouverts

- **SFISHTRACK pas encore téléchargé** — bloqué côté Google Drive, à
  réessayer. `convert_sfishtrack.py` est écrit et testé sur des données
  synthétiques reproduisant la structure attendue, mais **pas encore
  validé sur le vrai fichier** (noms de catégories COCO exacts à confirmer
  via `--check-only`).
- **24 classes d'espèces + 2 classes d'oursins sans source de données** :
  l'ancien pipeline iNaturalist (seule source qui les alimentait) a été
  retiré (2026-08-12, pivot SFISHTRACK) — seule la classe `poisson` a une
  source active tant que SFISHTRACK reste mono-classe "fish". À rouvrir
  explicitement si ces classes doivent être ré-alimentées un jour.
- **Modèles existants obsolètes** (`portfauna_v1`/`v2`, issus de
  l'ancien pipeline iNaturalist) — à ré-entraîner sur SFISHTRACK
  (`portfauna_v3`) avant tout usage réel.
- **`species.yaml`** toujours provisoire, à valider avec le Parc/les plongeurs.
- **Seuils non calibrés** : anti-doublon registre (`--threshold` 0.75),
  file de vérification (`--review-threshold` 0.5).
- **Pipeline vidéo jamais testé** sur une vraie vidéo du port.
- **iNaturalist et SFISHTRACK ne doivent pas être mélangés** dans un même
  jeu d'entraînement (demande explicite) — pas de mécanisme de fusion
  multi-source actuellement dans le projet.
