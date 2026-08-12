"""Prétraitement des images sous-marines : correction de la dominante de
couleur (balance des blancs gray-world, proportionnelle à la dominante
détectée).

Sans ça, la dominante bleu-vert de l'eau (variable selon profondeur/
turbidité/luminosité) est une source de variance que le modèle doit
apprendre à ignorer plutôt qu'un signal utile — ce prétraitement la corrige
en amont, une fois pour toutes.

**CLAHE (amélioration de contraste) testée et abandonnée** : même à faible
clipLimit, elle introduit un halo bleu-gris artificiel sur les arrière-plans
flous (bokeh) très fréquents en macro sous-marine — confirmé visuellement
sur plusieurs échantillons avant d'être retirée. Seule la balance des blancs
est conservée.

Appliqué aux deux points d'entrée des pixels dans le pipeline, pour que
l'entraînement et l'inférence voient la même distribution de couleurs :
  - dataset d'entraînement : CLI ci-dessous, en place, idempotent (incrémental
    — ne retraite pas une image déjà marquée faite, sauf --force)
  - vidéo à l'inférence : `enhance_underwater()` importé et appliqué à
    chaque frame dans track.py et app.py avant détection

Usage (bibliothèque) :
    from src.preprocess import enhance_underwater
    frame = enhance_underwater(frame)  # image BGR (cv2), uint8

Usage (CLI, dataset en place) :
    python src/preprocess.py --images-dir data/inaturalist
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

MARKER_FILE = ".preprocessed_files.txt"


def gray_world_white_balance(image: np.ndarray, max_gain: float = 1.5, strength_scale: float = 2.0) -> np.ndarray:
    """Corrige la dominante bleu/vert de l'eau en poussant la moyenne de
    l'image vers le neutre — hypothèse gray-world classique, mais appliquée
    avec une force proportionnelle à la dominante réellement détectée plutôt
    qu'en force fixe.

    Nécessaire en pratique sur ce dataset : beaucoup de photos sont des
    macros au flash (sujet naturellement chaud/sombre, fond noir, pas de
    vraie dominante eau) où une correction gray-world classique sur-corrige
    et ajoute un cast bleu/violet artificiel qui n'existait pas (confirmé
    visuellement avant d'intégrer cette version). `cast_strength` mesure la
    même signature que `looks_underwater()` dans fetch_inaturalist_sam.py
    (B+G contre R) : ~0 sur une image déjà neutre ou à dominante chaude (pas
    de correction), jusqu'à 1 sur une forte dominante eau (correction pleine,
    plafonnée par `max_gain`).
    """
    img = image.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15) & (gray < 240)
    if mask.sum() < 0.05 * gray.size:
        mask = np.ones_like(gray, dtype=bool)

    b, g, r = cv2.split(img)
    b_mean, g_mean, r_mean = b[mask].mean(), g[mask].mean(), r[mask].mean()

    cast_strength = float(np.clip(strength_scale * ((b_mean + g_mean) / (2 * max(r_mean, 1.0)) - 1), 0, 1))
    if cast_strength <= 0:
        return image

    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    gain_b = np.clip(gray_mean / max(b_mean, 1.0), 1 / max_gain, max_gain)
    gain_g = np.clip(gray_mean / max(g_mean, 1.0), 1 / max_gain, max_gain)
    gain_r = np.clip(gray_mean / max(r_mean, 1.0), 1 / max_gain, max_gain)

    corrected = cv2.merge([b * gain_b, g * gain_g, r * gain_r])
    blended = cast_strength * corrected + (1 - cast_strength) * img
    return np.clip(blended, 0, 255).astype(np.uint8)


def enhance_underwater(image: np.ndarray) -> np.ndarray:
    """Point d'entrée du prétraitement. Image BGR uint8 (cv2)."""
    return gray_world_white_balance(image)


def process_dataset(images_dir: Path, force: bool = False) -> int:
    marker_path = images_dir / MARKER_FILE
    done = set() if force else set(
        marker_path.read_text(encoding="utf-8").splitlines() if marker_path.exists() else []
    )

    newly_done = []
    for split in ("train", "val"):
        split_dir = images_dir / "images" / split
        if not split_dir.is_dir():
            continue
        for image_path in sorted(split_dir.iterdir()):
            key = f"{split}/{image_path.name}"
            if key in done:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            cv2.imwrite(str(image_path), enhance_underwater(image))
            newly_done.append(key)

    if newly_done:
        mode = "w" if force else "a"
        with marker_path.open(mode, encoding="utf-8") as f:
            for key in newly_done:
                f.write(key + "\n")

    print(f"{len(newly_done)} image(s) prétraitée(s) en place -> {images_dir} ({len(done)} déjà faites, ignorées)")
    return len(newly_done)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="data/inaturalist", help="Dossier dataset (contient images/train, images/val)")
    parser.add_argument("--force", action="store_true", help="Retraiter aussi les images déjà marquées faites")
    args = parser.parse_args()

    process_dataset(Path(args.images_dir), force=args.force)


if __name__ == "__main__":
    main()
