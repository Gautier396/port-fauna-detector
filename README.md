# Détecteur de faune marine du port

Pipeline de vision par ordinateur pour détecter et identifier automatiquement
les poissons (+ oursins) observés dans un port méditerranéen, à partir de
vidéos de plongée GoPro, avec un registre d'individus qui évite qu'un même
animal revu plusieurs fois soit compté en double.

## Statut actuel

- **Données d'entraînement** : iNaturalist uniquement (seule source active).
  27 classes dans `configs/species.yaml` : `poisson` (classe parente,
  cf. plus bas) + 24 espèces de poissons + `oursin_violet`/`oursin_noir`
  (exception hors "poissons", gardée explicitement).
- **Boxing** : SAM2 (`sam2.1_t`), automatique, plusieurs boîtes par image
  si plusieurs individus (bancs de poissons). Filtre de plausibilité sous-
  marine (heuristique couleur, best-effort) avant conservation.
- **Prétraitement** : `src/preprocess.py` corrige la dominante bleu/vert de
  l'eau (balance des blancs gray-world, proportionnelle à la dominante
  détectée — n'altère pas les photos déjà neutres/chaudes, ex. macro au
  flash). Appliqué en place sur le dataset ET à chaque frame vidéo à
  l'inférence (`track.py`, `app.py`), pour que l'entraînement et l'inférence
  voient la même distribution de couleurs.
- **Modèles entraînés** (`models/portfauna_v1.pt`, `portfauna_v2.pt`) :
  sur l'ancien schéma de classes, **plus compatibles** avec le
  `species.yaml` actuel — un nouvel entraînement est nécessaire avant
  d'utiliser `app.py` en confiance.
- **Pipeline vidéo** (tracking → embeddings → registre → export) :
  jamais exécuté sur une vraie vidéo du port.

## Classe "poisson" (parente)

Chaque boîte d'une espèce de poisson (classes 1-24) est dupliquée sous la
classe 0 `poisson` (mêmes coordonnées) via `src/add_fish_umbrella.py`, pour
donner plus de volume d'entraînement à une détection poisson générique.
**Compromis assumé** : à l'inférence, ça peut faire apparaître 2 détections
(poisson + espèce précise) au même endroit pour un seul poisson réel — pas
encore traité côté `track.py`/`registry.py` (cf. Points ouverts).

## Architecture

```
iNaturalist ──► fetch_inaturalist_sam.py (utilise fetch_inaturalist.py en bibliothèque) ──► images + boîtes SAM2
                                                                          │
                                                                          ▼
                                              add_fish_umbrella.py (ajoute la classe "poisson")
                                                                          │
                                                                          ▼
                                                         train.py ──► models/<name>.pt

Vidéo GoPro ──► track.py (détection+tracking) ──┬─► embeddings.py ──► registry.py ──► export.py
                                                  └─► review_queue.py (si confiance basse) ──► resolve_review.py
```

`pipeline.py` enchaîne tracking → embeddings → registre pour une vidéo.
`app.py` et `watch_relabel.py`-like outils : voir Structure.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
src/
  fetch_inaturalist.py      <- bibliothèque API iNaturalist (pas de CLI — importé par fetch_inaturalist_sam.py)
  fetch_inaturalist_sam.py  <- CLI dataset iNaturalist (boîtes SAM2 natives, multi-objets)
  preprocess.py              <- correction couleur sous-marine (dataset en place + frames vidéo à l'inférence)
  add_fish_umbrella.py      <- ajoute la classe "poisson" (duplique les boîtes d'espèces)
  merge_datasets.py         <- fusionne plusieurs sources en un jeu YOLO (inutile tant qu'iNaturalist
                                est la seule source active — cf. configs/data_inaturalist.yaml)
  train.py                  <- entraînement YOLO -> models/<name>.pt
  track.py                  <- détection + ByteTrack + crops (marque needs_review)
  review_queue.py           <- file de vérification (tracks à confiance basse)
  resolve_review.py         <- referme la file une fois l'espèce confirmée
  embeddings.py             <- embedding ResNet18 par track
  registry.py               <- registre SQLite + anti-doublon (espèce + fenêtre temporelle)
  export.py                 <- CSV, stats
configs/
  species.yaml              <- 27 classes cibles (PROVISOIRE)
  inaturalist_taxa.yaml     <- classe -> taxon iNaturalist (espèces historiques)
  inaturalist_taxa_new.yaml <- classe -> taxon iNaturalist (nouvelles espèces, fichier séparé exprès)
  data_inaturalist.yaml      <- généré, format Ultralytics (pointe directement sur data/inaturalist/)
data/inaturalist/            <- images + labels YOLO + ATTRIBUTIONS.csv (licences)
outputs/                     <- registry.db, tracks/, crops/, embeddings/, review_queue/
pipeline.py                  <- orchestrateur bout-en-bout (une vidéo -> registre)
app.py                       <- interface glisser-déposer, test vidéo avec le dernier modèle (port 7860)
```

## Utilisation

```bash
# Dataset (nouvelles espèces, boîtes SAM2 natives multi-objets) :
python src/fetch_inaturalist_sam.py --check-only
python src/fetch_inaturalist_sam.py --output data/inaturalist --max-images-per-class 300

# Correction couleur sous-marine (en place, idempotent — incrémental après un nouveau fetch) :
python src/preprocess.py --images-dir data/inaturalist

# Compléter la classe "poisson" (idempotent, à relancer après tout nouveau fetch) :
python src/add_fish_umbrella.py --images-dir data/inaturalist

# Entraînement (directement sur data/inaturalist/, seule source active) :
python src/train.py --data configs/data_inaturalist.yaml --name portfauna_v3

# Pipeline vidéo complet :
python pipeline.py --video data/raw/plongee1.mp4 --model models/portfauna_v3.pt

# Test rapide glisser-déposer :
python app.py
```

## Points ouverts

- **Double détection poisson/espèce à l'inférence** (cf. section dédiée
  plus haut) — pas de dédoublonnage implémenté côté `track.py`/`registry.py`.
- **Modèles existants obsolètes** vis-à-vis du `species.yaml` actuel — à
  ré-entraîner (`portfauna_v3`) avant tout usage réel.
- **12 nouvelles espèces sans données** pour l'instant (castagnole, oblade,
  bogue, sar_museau_noir, sparaillon, serran_chevrette, serran_ecriture,
  atherine, rouget, chinchard, vive, sole) — fetch interrompu volontairement,
  à relancer (`fetch_inaturalist_sam.py`).
- **Filtre "sous-marin"** (`fetch_inaturalist_sam.py`) : heuristique couleur
  best-effort, pas parfaite.
- **`species.yaml`** toujours provisoire, à valider avec le Parc/les plongeurs.
- **Seuils non calibrés** : anti-doublon registre (`--threshold` 0.75),
  file de vérification (`--review-threshold` 0.5).
- **Pipeline vidéo jamais testé** sur une vraie vidéo du port.
