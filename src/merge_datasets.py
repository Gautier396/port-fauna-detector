"""Fusionne les jeux d'entraînement en un seul jeu YOLO (configs/data_merged.yaml).

**iNaturalist seul depuis le 2026-08-12** (demande explicite) — FathomNet et
Roboflow ont été retirés de `SOURCES`. Raison : l'entraînement `portfauna_v2`
(après ré-annotation SAM + révision manuelle des labels iNaturalist) a régressé
sur plusieurs classes par rapport à `portfauna_v1`, et l'inspection visuelle a
confirmé une cause concrète — mélanger les vraies bounding boxes de FathomNet/
Roboflow avec les boxes SAM/manuelles d'iNaturalist pour une même classe donne
une vérité terrain incohérente (déjà pressenti au moment du choix "union"
2026-08-07, confirmé empiriquement ensuite). `data/fathomnet/` et
`data/roboflow/` restent sur disque (rien de supprimé) au cas où on voudrait
les réintégrer plus tard — juste exclus de la fusion pour l'instant.

Usage:
    python src/merge_datasets.py --output data/merged --data-yaml configs/data_merged.yaml
"""
import argparse
import shutil
from pathlib import Path

import yaml

SOURCES = {
    "inaturalist": "configs/data_inaturalist.yaml",
    # "fathomnet": "configs/data.yaml",       # retiré 2026-08-12, cf. docstring
    # "roboflow": "configs/data_roboflow.yaml",  # retiré 2026-08-12, cf. docstring
}


def merge_source(source_name: str, data_yaml_path: Path, output_dir: Path) -> int:
    if not data_yaml_path.exists():
        print(f"  {source_name}: {data_yaml_path} introuvable, ignoré.")
        return 0

    config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    source_root = Path(config["path"])

    n = 0
    for split in ("train", "val"):
        src_images_dir = source_root / "images" / split
        src_labels_dir = source_root / "labels" / split
        if not src_images_dir.is_dir():
            continue

        dst_images_dir = output_dir / "images" / split
        dst_labels_dir = output_dir / "labels" / split
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in src_images_dir.iterdir():
            label_path = src_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            dst_image = dst_images_dir / image_path.name
            if dst_image.exists():
                print(f"  Avertissement : collision de nom '{image_path.name}' ({source_name}), écrasée.")

            shutil.copy(image_path, dst_image)
            shutil.copy(label_path, dst_labels_dir / label_path.name)
            n += 1

    print(f"  {source_name}: {n} image(s)")
    return n


def write_data_yaml(species_path: Path, output_dir: Path, data_yaml_path: Path) -> None:
    species = yaml.safe_load(species_path.read_text(encoding="utf-8"))["names"]
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(output_dir.resolve()), "train": "images/train", "val": "images/val", "names": species},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species-config", default="configs/species.yaml", help="Classes YOLO (défaut: configs/species.yaml)")
    parser.add_argument("--output", default="data/merged", help="Dossier de sortie du dataset fusionné")
    parser.add_argument("--data-yaml", default="configs/data_merged.yaml", help="Chemin du data.yaml Ultralytics généré")
    args = parser.parse_args()

    output_dir = Path(args.output)
    print("Fusion des sources :")
    total = sum(merge_source(name, Path(path), output_dir) for name, path in SOURCES.items())

    write_data_yaml(Path(args.species_config), output_dir, Path(args.data_yaml))

    print(f"\nTotal : {total} image(s) -> {output_dir}")
    print(f"data.yaml écrit -> {args.data_yaml}")


if __name__ == "__main__":
    main()
