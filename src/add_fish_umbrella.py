"""Ajoute une ligne "poisson" (classe 0, parente) dupliquant chaque boîte
d'une espèce de poisson (classes 1-24), pour donner plus de volume
d'entraînement à une détection poisson générique en plus de l'espèce précise
(demande explicite 2026-08-12) — cf. commentaire dans configs/species.yaml.

Idempotent : ne duplique pas une box qui a déjà sa ligne "poisson"
correspondante (comparaison textuelle exacte) — peut être relancé sans
risque après un nouveau fetch d'images de poissons.

ATTENTION (cf. species.yaml et README Points ouverts) : duplique
volontairement les boîtes. Le pipeline vidéo (track.py) verra donc 2
détections (poisson + espèce) au même endroit pour un seul poisson réel à
l'inférence, ce qui peut fausser le comptage d'individus du registre si rien
n'est fait pour dédupliquer — pas encore traité.

Usage:
    python src/add_fish_umbrella.py --images-dir data/inaturalist
"""
import argparse
from pathlib import Path

POISSON_CLASS_ID = 0
FISH_CLASS_IDS = set(range(1, 25))  # 1-24, cf. configs/species.yaml


def add_umbrella_to_label(label_path: Path) -> bool:
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    existing = set(lines)
    new_lines = list(lines)
    changed = False

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        if class_id in FISH_CLASS_IDS:
            poisson_line = " ".join([str(POISSON_CLASS_ID), *parts[1:]])
            if poisson_line not in existing:
                new_lines.append(poisson_line)
                existing.add(poisson_line)
                changed = True

    if changed:
        label_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="data/inaturalist", help="Dossier contenant labels/{train,val} (défaut: data/inaturalist)")
    args = parser.parse_args()

    base = Path(args.images_dir)
    n_changed = 0
    n_total = 0
    for split in ("train", "val"):
        labels_dir = base / "labels" / split
        if not labels_dir.is_dir():
            continue
        for label_path in labels_dir.glob("*.txt"):
            n_total += 1
            if add_umbrella_to_label(label_path):
                n_changed += 1

    print(f"{n_changed}/{n_total} label(s) complété(s) avec une ligne 'poisson'.")


if __name__ == "__main__":
    main()
