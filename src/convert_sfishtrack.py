"""Convertit les annotations COCO de SFISHTRACK (segmentation d'instance,
tracking multi-objet, une vidéo par fichier) en labels YOLO (bounding box),
pour intégration dans le pipeline d'entraînement existant (configs/species.yaml).

Structure attendue (racine passée via --dataset-root) :
    dataset/
      videos/          <- vidéos sources, pas utilisées ici (on part des frames)
      frames/<video_id>/frame_XXXXXX.jpg
      annotations/<video_id>.json   <- un COCO par vidéo
      masks/masks_XXX/mask_XXXXXX.jpg   <- pas utilisées (segmentation en plus
                                            des boîtes, ce projet n'entraîne
                                            que sur des boîtes englobantes)
      metadata/<video_id>.json      <- pas utilisé pour l'instant

Chaque annotations/<video_id>.json est un COCO standard : "categories"
(class_labels — à la date d'écriture, incertain si multi-espèces ou classe
unique "fish", cf. configs/sfishtrack_species_map.yaml), "images"
(frame_index, file_name...), "annotations" (bbox, category_id, un
identifiant de tracking par instance — pas utilisé ici, ce projet entraîne
sur des images fixes, pas du tracking).

Seul le champ COCO standard "bbox" ([x, y, largeur, hauteur] en pixels,
origine coin haut-gauche) est utilisé — pas besoin des polygones de
segmentation ni des masques séparés pour de la détection par boîte.

Sortie dans data/sfishtrack/, jeu d'entraînement autonome (pas fusionné
avec d'autres sources).

Split train/val PAR VIDÉO (pas par frame) : les frames d'une même vidéo sont
très corrélées (même scène, même poisson qui bouge peu d'une frame à
l'autre) — les répartir aléatoirement entre train et val ferait fuiter
essentiellement la même image des deux côtés.

**NON TESTÉ sur les vraies données** au moment de l'écriture (zip SFISHTRACK
bloqué côté Google Drive, "too many users..."). À valider dès que le
dataset réel est disponible : noms exacts des catégories COCO (cf.
configs/sfishtrack_species_map.yaml, volontairement vide — à remplir depuis
la sortie réelle de --check-only) et le nom exact du champ frame/index dans
"images" (essaie file_name, sinon frame_index).

Usage:
    # 1. Voir les catégories réelles + couverture du mapping (aucune écriture)
    python src/convert_sfishtrack.py --dataset-root <chemin>/dataset --check-only

    # 2. Convertir (une fois configs/sfishtrack_species_map.yaml rempli)
    python src/convert_sfishtrack.py --dataset-root <chemin>/dataset --output data/sfishtrack
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def load_species_map(species_path: Path, map_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    species = yaml.safe_load(species_path.read_text(encoding="utf-8"))["names"]
    name_to_id = {name: cls_id for cls_id, name in species.items()}
    raw_map = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
    coco_name_to_class_id = {}
    for coco_name, species_name in raw_map.items():
        if species_name not in name_to_id:
            print(f"  Avertissement : '{species_name}' (mappé depuis '{coco_name}') absent de species.yaml, ignoré.")
            continue
        coco_name_to_class_id[coco_name] = name_to_id[species_name]
    return species, coco_name_to_class_id


def iter_video_annotations(dataset_root: Path):
    """Un (video_id, coco_dict, frames_dir) par fichier annotations/<video_id>.json."""
    annotations_dir = dataset_root / "annotations"
    frames_dir = dataset_root / "frames"
    for json_path in sorted(annotations_dir.glob("*.json")):
        video_id = json_path.stem
        coco = json.loads(json_path.read_text(encoding="utf-8"))
        yield video_id, coco, frames_dir / video_id


def check_coverage(dataset_root: Path, coco_name_to_class_id: dict[str, int]) -> None:
    ann_count_by_name = defaultdict(int)
    videos_seen = 0
    for video_id, coco, frames_dir in iter_video_annotations(dataset_root):
        videos_seen += 1
        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        for ann in coco.get("annotations", []):
            ann_count_by_name[cat_names.get(ann["category_id"], "?")] += 1
        if videos_seen == 1:
            img0 = coco.get("images", [{}])[0] if coco.get("images") else {}
            print(f"Champs 'images' (vidéo {video_id}) : {sorted(img0.keys())}")
            print(f"Frames sur disque pour {video_id} ({frames_dir}) : {'trouvé' if frames_dir.is_dir() else 'INTROUVABLE'}\n")

    if videos_seen == 0:
        print(f"Aucun fichier .json trouvé dans {dataset_root / 'annotations'}.")
        return

    print(f"{videos_seen} vidéo(s) (fichiers annotations/*.json), {len(ann_count_by_name)} catégorie(s) au total :\n")
    for name, n in sorted(ann_count_by_name.items(), key=lambda kv: -kv[1]):
        mapped = coco_name_to_class_id.get(name)
        status = f"-> {mapped} ({name})" if mapped is not None else "PAS MAPPÉE (à ajouter dans sfishtrack_species_map.yaml)"
        print(f"  [{n:>7} annotation(s)]  {name!r}  {status}")


def convert(dataset_root: Path, coco_name_to_class_id: dict[str, int], output_dir: Path, val_split: float) -> dict[str, int]:
    video_ids = [video_id for video_id, _, _ in iter_video_annotations(dataset_root)]
    shuffled = list(video_ids)
    random.Random(42).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if shuffled else 0
    val_videos = set(shuffled[:n_val])

    counts = {"videos": 0, "images": 0, "boxes": 0, "missing_image_file": 0, "missing_bbox_field": 0}
    for video_id, coco, frames_dir in iter_video_annotations(dataset_root):
        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        images_by_id = {img["id"]: img for img in coco.get("images", [])}

        anns_by_image = defaultdict(list)
        for ann in coco.get("annotations", []):
            class_id = coco_name_to_class_id.get(cat_names.get(ann["category_id"]))
            if class_id is None:
                continue
            bbox = ann.get("bbox")
            if bbox is None:
                counts["missing_bbox_field"] += 1
                continue
            anns_by_image[ann["image_id"]].append((class_id, bbox))

        if not anns_by_image:
            continue

        split = "val" if video_id in val_videos else "train"
        dst_images_dir = output_dir / "images" / split
        dst_labels_dir = output_dir / "labels" / split
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        counts["videos"] += 1
        for img_id, boxes in anns_by_image.items():
            img_meta = images_by_id.get(img_id)
            if not img_meta:
                continue
            file_name, w, h = img_meta.get("file_name"), img_meta.get("width"), img_meta.get("height")
            if not file_name or not w or not h:
                continue

            src_image = frames_dir / file_name
            if not src_image.exists():
                counts["missing_image_file"] += 1
                continue

            stem = Path(file_name).stem
            lines = []
            for class_id, (x, y, bw, bh) in boxes:
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            out_name = f"sft_{video_id}_{stem}"
            shutil.copy(src_image, dst_images_dir / f"{out_name}.jpg")
            (dst_labels_dir / f"{out_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts["images"] += 1
            counts["boxes"] += len(lines)

    return counts


def write_data_yaml(species: dict, output_dir: Path, data_yaml_path: Path) -> None:
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
    parser.add_argument("--dataset-root", required=True, help="Racine du dataset SFISHTRACK (contient annotations/, frames/, ...)")
    parser.add_argument("--species-config", default="configs/species.yaml", help="Classes YOLO (défaut: configs/species.yaml)")
    parser.add_argument("--species-map", default="configs/sfishtrack_species_map.yaml", help="Mapping catégorie COCO -> classe species.yaml")
    parser.add_argument("--output", default="data/sfishtrack", help="Dossier de sortie YOLO (défaut: data/sfishtrack)")
    parser.add_argument("--data-yaml", default="configs/data_sfishtrack.yaml", help="Chemin du data.yaml Ultralytics généré")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction de VIDÉOS en validation (défaut: 0.15)")
    parser.add_argument("--check-only", action="store_true", help="N'affiche que les catégories/couverture, ne convertit rien")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    species, coco_name_to_class_id = load_species_map(Path(args.species_config), Path(args.species_map))

    if args.check_only:
        check_coverage(dataset_root, coco_name_to_class_id)
        return

    output_dir = Path(args.output)
    counts = convert(dataset_root, coco_name_to_class_id, output_dir, args.val_split)
    write_data_yaml(species, output_dir, Path(args.data_yaml))

    print(f"{counts['videos']} vidéo(s), {counts['images']} image(s), {counts['boxes']} boîte(s) -> {output_dir}")
    if counts["missing_image_file"]:
        print(f"Avertissement : {counts['missing_image_file']} frame(s) référencée(s) dans un annotations/*.json mais introuvable(s) sur disque.")
    if counts["missing_bbox_field"]:
        print(f"Avertissement : {counts['missing_bbox_field']} annotation(s) sans champ 'bbox', ignorée(s).")
    print(f"data.yaml écrit -> {args.data_yaml}")


if __name__ == "__main__":
    main()
