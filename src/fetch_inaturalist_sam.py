"""Télécharge des images iNaturalist pour de nouvelles classes et les annote
directement avec SAM2 (segmentation automatique, plusieurs boîtes par image
si plusieurs individus sont détectés) — pas d'étape intermédiaire "plein
cadre" comme dans fetch_inaturalist.py (cette approximation s'est avérée
coûteuse en précision, cf. README "Entraînement").

Filtre aussi les photos qui ne semblent pas sous-marines, par une
heuristique de dominante bleu/vert (best-effort, PAS parfait — une photo au
flash rapproché peut avoir des couleurs naturelles même sous l'eau, et une
photo hors de l'eau à dominante bleu/vert — ex. bac/étal sur fond bleu —
pourrait passer le filtre à tort). Le seuil est volontairement permissif
(on préfère garder une photo ambiguë plutôt que rejeter à tort une bonne
photo sous-marine) : la révision manuelle via watch_relabel.py reste
recommandée pour un contrôle qualité final.

Utilise SAM en mode automatique (aucun point-prompt = segmente tout ce qui
est saillant dans l'image), puis ne garde que les masques dont l'aire est
dans une plage plausible pour un poisson individuel (`--min-area-ratio` /
`--max-area-ratio`) — élimine les fragments minuscules et les grands blobs
de fond/eau.

Ce script est volontairement séparé de fetch_inaturalist.py et utilise un
fichier taxa séparé (configs/inaturalist_taxa_new.yaml) : réutiliser
inaturalist_taxa.yaml directement avec fetch_inaturalist.py ré-écraserait
les labels des classes déjà ré-annotées (SAM + révision manuelle) avec un
nouveau plein-cadre — cf. commentaire dans inaturalist_taxa_new.yaml.

Usage:
    python src/fetch_inaturalist_sam.py --check-only
    python src/fetch_inaturalist_sam.py --output data/inaturalist --max-images-per-class 300
"""
import argparse
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import yaml
from ultralytics import SAM

from fetch_inaturalist import check_coverage, load_taxa, query_taxon_photos

_sam_model: SAM | None = None


def get_sam_model(weights: str) -> SAM:
    global _sam_model
    if _sam_model is None:
        _sam_model = SAM(weights)
    return _sam_model


def looks_underwater(image_path: Path) -> bool:
    """Heuristique best-effort : lumière naturelle sous-marine = rouge absorbé en
    premier -> teinte bleu-cyan dominante. Seuil volontairement permissif, cf.
    docstring du module.

    Un simple "(bleu+vert) > rouge" est trompé par la végétation terrestre (le
    vert de la chlorophylle a la même signature globale) — confirmé en pratique
    (photo de plante passée à tort). Ajout d'une deuxième condition : le bleu ne
    doit pas être nettement plus faible que le vert (le vert végétal a un bleu
    bas, le cyan aquatique a bleu ≈ vert), ce qui écarte la végétation sans
    resserrer le filtre sur les vraies photos sous-marines."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    b, g, r = (img[:, :, i].astype(np.float32).mean() for i in range(3))
    return (b + g) > r * 1.15 and b > g * 0.85


def box_with_sam(model: SAM, image_path: Path, class_id: int, min_area: float, max_area: float) -> list[str]:
    """Segmente automatiquement l'image (pas de prompt = tout ce qui est saillant),
    retourne une ligne de label YOLO par masque dont l'aire est plausible."""
    result = model.predict(source=str(image_path), verbose=False)[0]
    if result.boxes is None:
        return []

    lines = []
    for cx, cy, w, h in result.boxes.xywhn.tolist():
        area = w * h
        if min_area <= area <= max_area:
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def fetch_class(
    model: SAM,
    class_name: str,
    taxa: list[str],
    class_id: int,
    max_images: int,
    output_dir: Path,
    min_area: float,
    max_area: float,
) -> tuple[list[dict], dict[str, int]]:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    photos_by_id = {}
    for taxon_name in taxa:
        for photo in query_taxon_photos(taxon_name, max_images):
            photos_by_id.setdefault(photo["photo_id"], photo)

    attributions = []
    counts = {"kept": 0, "not_underwater": 0, "no_box": 0, "download_failed": 0}

    for photo in list(photos_by_id.values())[:max_images]:
        image_path = images_dir / f"inat_{photo['photo_id']}.jpg"
        label_path = labels_dir / f"inat_{photo['photo_id']}.txt"

        if not image_path.exists():
            try:
                urlretrieve(photo["url"], image_path)
            except Exception as exc:
                print(f"    {photo['photo_id']}: échec du téléchargement ({exc}), ignoré.")
                counts["download_failed"] += 1
                continue

        if not looks_underwater(image_path):
            image_path.unlink()
            counts["not_underwater"] += 1
            continue

        lines = box_with_sam(model, image_path, class_id, min_area, max_area)
        if not lines:
            image_path.unlink()
            counts["no_box"] += 1
            continue

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts["kept"] += 1
        attributions.append(
            {
                "photo_id": photo["photo_id"],
                "class_name": class_name,
                "license_code": photo["license_code"],
                "attribution": photo["attribution"],
                "observation_url": f"https://www.inaturalist.org/observations/{photo['observation_id']}",
                "n_boxes": len(lines),
            }
        )

    return attributions, counts


def split_new_images(attributions: list[dict], output_dir: Path, val_split: float) -> None:
    """Répartit train/val uniquement les images tout juste ajoutées (par photo_id) —
    ne touche pas aux images déjà splittées d'un run précédent."""
    import random

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    photo_ids = [a["photo_id"] for a in attributions]
    random.Random(42).shuffle(photo_ids)
    n_val = int(len(photo_ids) * val_split)
    val_ids = set(photo_ids[:n_val])

    for photo_id in photo_ids:
        split = "val" if photo_id in val_ids else "train"
        for sub_dir, ext in ((images_dir, ".jpg"), (labels_dir, ".txt")):
            src = sub_dir / f"inat_{photo_id}{ext}"
            if not src.exists():
                continue
            dst_dir = sub_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dst_dir / f"inat_{photo_id}{ext}")


def append_attributions(attributions: list[dict], output_dir: Path) -> None:
    path = output_dir / "ATTRIBUTIONS.csv"
    fieldnames = ["photo_id", "class_name", "license_code", "attribution", "observation_url", "n_boxes"]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(attributions)


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
    parser.add_argument("--taxa-config", default="configs/inaturalist_taxa_new.yaml", help="Mapping classe -> taxon(s) iNaturalist (fichier séparé, cf. docstring)")
    parser.add_argument("--output", default="data/inaturalist", help="Dossier de sortie (défaut: data/inaturalist, partagé avec fetch_inaturalist.py)")
    parser.add_argument("--data-yaml", default="configs/data_inaturalist.yaml", help="Chemin du data.yaml Ultralytics régénéré")
    parser.add_argument("--sam-model", default="sam2.1_t.pt", help="Poids SAM (défaut: sam2.1_t.pt)")
    parser.add_argument("--max-images-per-class", type=int, default=300, help="Plafond d'images par classe (défaut: 300)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction en validation (défaut: 0.15)")
    parser.add_argument("--min-area-ratio", type=float, default=0.004, help="Aire minimale d'une box (fraction de l'image, défaut: 0.004)")
    parser.add_argument("--max-area-ratio", type=float, default=0.5, help="Aire maximale d'une box (défaut: 0.5)")
    parser.add_argument("--check-only", action="store_true", help="N'affiche que la couverture iNaturalist, ne télécharge rien")
    args = parser.parse_args()

    species, taxa_by_class = load_taxa(Path(args.species_config), Path(args.taxa_config))

    if args.check_only:
        check_coverage(taxa_by_class)
        return

    name_to_id = {name: cls_id for cls_id, name in species.items()}
    output_dir = Path(args.output)
    model = get_sam_model(args.sam_model)

    all_attributions = []
    total_counts = {"kept": 0, "not_underwater": 0, "no_box": 0, "download_failed": 0}

    for class_name, taxa in taxa_by_class.items():
        if not taxa:
            continue
        if class_name not in name_to_id:
            print(f"Avertissement : classe '{class_name}' absente de {args.species_config}, ignorée.")
            continue

        print(f"{class_name} ({', '.join(taxa)}) :")
        attributions, counts = fetch_class(
            model, class_name, taxa, name_to_id[class_name], args.max_images_per_class,
            output_dir, args.min_area_ratio, args.max_area_ratio,
        )
        print(
            f"    {counts['kept']} gardée(s), {counts['not_underwater']} rejetée(s) "
            f"(pas sous-marine), {counts['no_box']} rejetée(s) (pas de box SAM plausible), "
            f"{counts['download_failed']} échec(s) de téléchargement."
        )
        all_attributions.extend(attributions)
        for k in total_counts:
            total_counts[k] += counts[k]

    if not all_attributions:
        print("\nAucune image conservée. Vérifie avec --check-only.")
        return

    split_new_images(all_attributions, output_dir, args.val_split)
    append_attributions(all_attributions, output_dir)
    write_data_yaml(Path(args.species_config), output_dir, Path(args.data_yaml))

    total_boxes = sum(a["n_boxes"] for a in all_attributions)
    print(f"\nTotal : {len(all_attributions)} image(s), {total_boxes} box(es) -> {output_dir}")
    print(f"data.yaml régénéré -> {args.data_yaml}")
    print(f"Attributions ajoutées -> {output_dir / 'ATTRIBUTIONS.csv'}")


if __name__ == "__main__":
    main()
