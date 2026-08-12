"""Calcule un vecteur d'embedding visuel pour chaque track (individu détecté).

Utilise un ResNet18 pré-entraîné ImageNet, tronqué avant la couche fully-
connected (features 512-d) : suffisant pour de la similarité perceptuelle
image-à-image sans entraînement supplémentaire, et torch/torchvision est déjà
une dépendance transitive d'ultralytics — pas de poids/lib additionnels à
gérer (ex: CLIP) pour ce prototype.

Lit le CSV `_tracks.csv` produit par track.py (colonne `best_crop_path`),
calcule un embedding par crop, l'enregistre en .npy, et réécrit le CSV avec
une colonne `embedding_path` en plus — prêt pour registry.py.

Usage:
    python embeddings.py --tracks-csv outputs/tracks/plongee1_tracks.csv --output-dir outputs/embeddings
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

_MODEL = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is None:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _MODEL = torch.nn.Sequential(*list(backbone.children())[:-1]).eval()
    return _MODEL


def embed_image(image_path: str) -> np.ndarray:
    model = get_model()
    image = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.reshape(-1).numpy()


def process_tracks(tracks_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_paths = []

    for _, row in tracks_df.iterrows():
        crop_path = row.get("best_crop_path")
        if not crop_path or not Path(crop_path).exists():
            embedding_paths.append(None)
            continue

        embedding = embed_image(crop_path)
        video_stem = Path(row["source_video"]).stem
        emb_path = output_dir / f"{video_stem}_track{int(row['track_id']):04d}.npy"
        np.save(emb_path, embedding)
        embedding_paths.append(str(emb_path))

    tracks_df = tracks_df.copy()
    tracks_df["embedding_path"] = embedding_paths
    return tracks_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks-csv", required=True, help="CSV _tracks.csv produit par track.py")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie pour les .npy")
    args = parser.parse_args()

    tracks_df = pd.read_csv(args.tracks_csv)
    result_df = process_tracks(tracks_df, Path(args.output_dir))

    result_df.to_csv(args.tracks_csv, index=False)

    n_ok = result_df["embedding_path"].notna().sum()
    print(f"{n_ok}/{len(result_df)} embeddings calculés -> colonne embedding_path ajoutée à {args.tracks_csv}")


if __name__ == "__main__":
    main()
