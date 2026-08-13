"""Convertit les annotations COCO de SFISHTRACK (segmentation d'instance,
tracking multi-objet, une vidéo par fichier) en labels YOLO (bounding box),
pour intégration dans le pipeline d'entraînement existant (configs/species.yaml).

Structure réelle (racine passée via --dataset-root = dossier "SFISHTRACK/"
extrait du zip) — vérifiée sur le vrai fichier le 2026-08-13 :
    SFISHTRACK/
      Videos/                    <- vidéos sources, pas utilisées ici (on part des frames)
      Frames/video_XXX/YYYYYY.png    <- ex. Frames/video_012/000000.png
      Annotations/videoXXX.json      <- un COCO par vidéo, ex. Annotations/video012.json
                                         (PAS de "_" ici, contrairement à Frames/video_XXX/)
      Masks/masks_XXX/               <- pas utilisées (segmentation en plus des
                                         boîtes, ce projet n'entraîne que sur
                                         des boîtes englobantes)
      metadata/videoXXX.json         <- pas utilisé pour l'instant

Le nom de dossier vidéo (Annotations/video012.json -> "video012") ne
correspond PAS directement au dossier de frames ("Frames/video_012/", avec
un "_") : `frames_dir_for()` fait la conversion (insère le "_" avant le
numéro).

Chaque Annotations/videoXXX.json est un COCO standard : "videos" (nom local
à ce fichier, pas fiable pour identifier la vidéo — on utilise le nom de
fichier), "images" (file_name, width, height, video_id — champ local
toujours =1 puisqu'un fichier = une vidéo), "categories" (une seule
catégorie observée, "object" — pas de distinction par espèce à ce jour,
cf. configs/sfishtrack_species_map.yaml), "annotations" (bbox, category_id,
segmentation RLE + track_id — pas utilisés ici, ce projet entraîne sur des
images fixes, pas du tracking, et seulement des boîtes).

Seul le champ COCO standard "bbox" ([x, y, largeur, hauteur] en pixels,
origine coin haut-gauche) est utilisé.

Sortie dans data/sfishtrack/, jeu d'entraînement autonome (pas fusionné
avec d'autres sources).

Split train/val PAR VIDÉO (pas par frame) : les frames d'une même vidéo sont
très corrélées (même scène, même poisson qui bouge peu d'une frame à
l'autre) — les répartir aléatoirement entre train et val ferait fuiter
essentiellement la même image des deux côtés.

Usage:
    # 1. Voir les catégories réelles + couverture du mapping (aucune écriture)
    python src/convert_sfishtrack.py --dataset-root data/external/sfishtrack/SFISHTRACK --check-only

    # 2. Convertir (une fois configs/sfishtrack_species_map.yaml vérifié)
    python src/convert_sfishtrack.py --dataset-root data/external/sfishtrack/SFISHTRACK --output data/sfishtrack
"""
import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

VIDEO_ID_RE = re.compile(r"^video(\d+)$")


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


def frames_dir_for(frames_root: Path, video_id: str) -> Path:
    """Annotations/video012.json -> video_id="video012" -> dossier Frames/video_012/."""
    m = VIDEO_ID_RE.match(video_id)
    folder_name = f"video_{m.group(1)}" if m else video_id
    return frames_root / folder_name


def iter_video_annotations(dataset_root: Path):
    """Un (video_id, coco_dict, frames_dir) par fichier Annotations/videoXXX.json."""
    annotations_dir = dataset_root / "Annotations"
    frames_root = dataset_root / "Frames"
    for json_path in sorted(annotations_dir.glob("*.json")):
        video_id = json_path.stem
        coco = json.loads(json_path.read_text(encoding="utf-8"))
        yield video_id, coco, frames_dir_for(frames_root, video_id)


def check_coverage(dataset_root: Path, coco_name_to_class_id: dict[str, int]) -> None:
    ann_count_by_name = defaultdict(int)
    videos_seen = 0
    missing_frames_dirs = []
    for video_id, coco, frames_dir in iter_video_annotations(dataset_root):
        videos_seen += 1
        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        for ann in coco.get("annotations", []):
            ann_count_by_name[cat_names.get(ann["category_id"], "?")] += 1
        if not frames_dir.is_dir():
            missing_frames_dirs.append((video_id, frames_dir))
        if videos_seen == 1:
            img0 = coco.get("images", [{}])[0] if coco.get("images") else {}
            print(f"Champs 'images' (vidéo {video_id}) : {sorted(img0.keys())}")
            print(f"Frames attendues pour {video_id} -> {frames_dir} : {'trouvé' if frames_dir.is_dir() else 'INTROUVABLE'}\n")

    if videos_seen == 0:
        print(f"Aucun fichier .json trouvé dans {dataset_root / 'Annotations'}.")
        return

    print(f"{videos_seen} vidéo(s) (fichiers Annotations/*.json), {len(ann_count_by_name)} catégorie(s) au total :\n")
    for name, n in sorted(ann_count_by_name.items(), key=lambda kv: -kv[1]):
        mapped = coco_name_to_class_id.get(name)
        status = f"-> {mapped} ({name})" if mapped is not None else "PAS MAPPÉE (à ajouter dans sfishtrack_species_map.yaml)"
        print(f"  [{n:>7} annotation(s)]  {name!r}  {status}")

    if missing_frames_dirs:
        print(f"\nAvertissement : {len(missing_frames_dirs)} vidéo(s) sans dossier Frames/ correspondant "
              f"(ex. {missing_frames_dirs[0][0]} -> {missing_frames_dirs[0][1]}).")


FRAME_NUM_RE = re.compile(r"(\d+)")


def build_frame_index(frames_dir: Path) -> dict[int, Path]:
    """Numéro de frame (entier, sans le padding) -> chemin réel du fichier.

    Nécessaire car le nommage des frames n'est PAS uniforme sur tout le
    dataset : la plupart des vidéos utilisent "NNNNNN.png", mais un premier
    lot (~vidéos 1-26, pas un intervalle propre) utilise encore
    "frame_NNNNNN.jpg" — un ancien format d'extraction jamais harmonisé,
    alors que les Annotations/*.json, eux, ont tous été régénérés plus tard
    en supposant uniformément "NNNNNN.png" (confirmé par les dates de
    fichiers : Frames de janvier vs Annotations d'avril pour ces vidéos).
    Indexer par numéro de frame (pas par nom exact) contourne ce décalage
    quel que soit le nom réel sur disque.
    """
    index = {}
    if not frames_dir.is_dir():
        return index
    for p in frames_dir.iterdir():
        m = FRAME_NUM_RE.search(p.stem)
        if m:
            index[int(m.group(1))] = p
    return index


def resolve_frame_path(frame_index: dict[int, Path], file_name: str) -> Path | None:
    m = FRAME_NUM_RE.search(Path(file_name).stem)
    return frame_index.get(int(m.group(1))) if m else None


def convert(dataset_root: Path, coco_name_to_class_id: dict[str, int], output_dir: Path, val_split: float) -> tuple[dict[str, int], set[int]]:
    video_ids = [video_id for video_id, _, _ in iter_video_annotations(dataset_root)]
    shuffled = list(video_ids)
    random.Random(42).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if shuffled else 0
    val_videos = set(shuffled[:n_val])

    used_class_ids: set[int] = set()
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

        frame_index = build_frame_index(frames_dir)

        counts["videos"] += 1
        for img_id, boxes in anns_by_image.items():
            img_meta = images_by_id.get(img_id)
            if not img_meta:
                continue
            file_name, w, h = img_meta.get("file_name"), img_meta.get("width"), img_meta.get("height")
            if not file_name or not w or not h:
                continue

            src_image = resolve_frame_path(frame_index, file_name)
            if src_image is None or not src_image.exists():
                counts["missing_image_file"] += 1
                continue

            lines = []
            for class_id, (x, y, bw, bh) in boxes:
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            out_name = f"sft_{video_id}_{src_image.stem}"
            shutil.copy(src_image, dst_images_dir / f"{out_name}{src_image.suffix}")
            (dst_labels_dir / f"{out_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts["images"] += 1
            counts["boxes"] += len(lines)
            used_class_ids.update(class_id for class_id, _ in boxes)

    return counts, used_class_ids


def write_data_yaml(species: dict, used_class_ids: set[int], output_dir: Path, data_yaml_path: Path) -> None:
    """N'écrit dans "names" que les classes réellement présentes dans les
    labels convertis, pas tout species.yaml (27 classes) : SFISHTRACK est
    mono-classe à ce jour (tout mappe vers "poisson", id 0) — entraîner un
    détecteur nominalement 27-classes alors que 26 d'entre elles n'ont
    jamais un seul exemple positif serait un gâchis de capacité et rendrait
    les métriques par classe inutilisables (AP non défini pour les classes
    absentes). Si une future source ajoute des classes d'espèces réelles,
    used_class_ids grandira en conséquence, aucun changement de code requis
    ici (les entrées manquantes de species.yaml sont juste omises, YOLO
    n'exige pas des ids contigus dans "names")."""
    filtered_names = {cls_id: name for cls_id, name in species.items() if cls_id in used_class_ids}
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(output_dir.resolve()), "train": "images/train", "val": "images/val", "names": filtered_names},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, help="Dossier SFISHTRACK/ extrait (contient Annotations/, Frames/, ...)")
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
    counts, used_class_ids = convert(dataset_root, coco_name_to_class_id, output_dir, args.val_split)
    write_data_yaml(species, used_class_ids, output_dir, Path(args.data_yaml))

    print(f"{counts['videos']} vidéo(s), {counts['images']} image(s), {counts['boxes']} boîte(s) -> {output_dir}")
    print(f"Classes présentes dans les labels : {sorted(species[i] for i in used_class_ids)}")
    if counts["missing_image_file"]:
        print(f"Avertissement : {counts['missing_image_file']} frame(s) référencée(s) dans un Annotations/*.json mais introuvable(s) sur disque.")
    if counts["missing_bbox_field"]:
        print(f"Avertissement : {counts['missing_bbox_field']} annotation(s) sans champ 'bbox', ignorée(s).")
    print(f"data.yaml écrit -> {args.data_yaml}")


if __name__ == "__main__":
    main()
