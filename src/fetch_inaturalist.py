"""Bibliothèque d'accès à l'API iNaturalist (recherche/comptage/photos par
taxon), utilisée par fetch_inaturalist_sam.py — seul point d'entrée CLI
actif pour construire le dataset (boîtes SAM2 natives).

Ne contient volontairement plus ni CLI ni téléchargement à plein cadre :
la version précédente (`download_class`/`main` avec labels YOLO plein-cadre
"0.5 0.5 1.0 1.0") a été retirée le 2026-08-12 — cette approximation s'est
avérée coûteuse en précision (cf. README "Entraînement") et le fetch réel
passe maintenant exclusivement par fetch_inaturalist_sam.py, qui boxe avec
SAM2 dès le téléchargement. Garder ce vieux chemin ici aurait été un piège :
le relancer par erreur aurait réintroduit exactement le problème que le
projet a corrigé.

Pour vérifier la couverture d'un fichier taxa (y compris
configs/inaturalist_taxa.yaml, les espèces "historiques") :
    python src/fetch_inaturalist_sam.py --check-only --taxa-config configs/inaturalist_taxa.yaml

Seules les photos sous licence réutilisable (cc0/cc-by/cc-by-sa/cc-by-nc/
cc-by-nc-sa, filtre serveur `photo_license`) et `quality_grade=research`
(identification vérifiée par la communauté) sont retournées.
"""
import time
from pathlib import Path

import requests
import yaml

API_URL = "https://api.inaturalist.org/v1/observations"
ALLOWED_LICENSES = ["cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"]
PHOTO_SIZE = "large"
PER_PAGE = 200
REQUEST_TIMEOUT_S = 30

# "research" (identification confirmée par 2/3 de la communauté) reste quasi
# toujours vide pour un taxon de rang famille/ordre (ex. Blenniidae,
# Nudibranchia) : la communauté affine presque toujours l'identification
# jusqu'à l'espèce, donc un consensus "research grade" au niveau famille est
# rarissime. Repli sur "research,needs_id" (exclut seulement le grade
# "casual" — animal captif/cultivé, localisation/date non vérifiable) quand
# "research" seul ne renvoie rien. Vérifié le 2026-08-07 : Blenniidae/
# Gobiidae/Paguridae/Nudibranchia à 0 en "research" seul, des centaines en
# "research,needs_id".
QUALITY_GRADES_STRICT = "research"
QUALITY_GRADES_FALLBACK = "research,needs_id"


def load_taxa(species_path: Path, taxa_path: Path) -> tuple[dict[int, str], dict[str, list[str]]]:
    species = yaml.safe_load(species_path.read_text(encoding="utf-8"))["names"]
    taxa_by_class = yaml.safe_load(taxa_path.read_text(encoding="utf-8")) or {}
    return species, taxa_by_class


def _count(taxon_name: str, quality_grade: str) -> int:
    resp = requests.get(
        API_URL,
        params={
            "taxon_name": taxon_name,
            "photos": "true",
            "quality_grade": quality_grade,
            "photo_license": ",".join(ALLOWED_LICENSES),
            "per_page": 0,
        },
        timeout=REQUEST_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["total_results"]


def resolve_quality_grade(taxon_name: str) -> str:
    """"research" d'abord ; repli sur "research,needs_id" si vide (rangs famille/ordre, cf. commentaire plus haut)."""
    return QUALITY_GRADES_STRICT if _count(taxon_name, QUALITY_GRADES_STRICT) > 0 else QUALITY_GRADES_FALLBACK


def count_taxon(taxon_name: str) -> int:
    """Nombre d'observations (research, ou research+needs_id en repli) avec photo sous licence réutilisable."""
    return _count(taxon_name, resolve_quality_grade(taxon_name))


def check_coverage(taxa_by_class: dict[str, list[str]]) -> None:
    for class_name, taxa in taxa_by_class.items():
        if not taxa:
            print(f"  [pas de taxon]  {class_name}")
            continue
        for taxon_name in taxa:
            try:
                n = count_taxon(taxon_name)
            except requests.RequestException as exc:
                print(f"  [erreur] {taxon_name} ({class_name}) : {exc}")
                continue
            print(f"  [{n:>6} obs.]  {taxon_name} ({class_name})")


def query_taxon_photos(taxon_name: str, max_images: int) -> list[dict]:
    """Retourne jusqu'à max_images photos (url grand format, licence, attribution, id) pour un taxon."""
    quality_grade = resolve_quality_grade(taxon_name)
    photos = []
    page = 1
    while len(photos) < max_images:
        resp = requests.get(
            API_URL,
            params={
                "taxon_name": taxon_name,
                "photos": "true",
                "quality_grade": quality_grade,
                "photo_license": ",".join(ALLOWED_LICENSES),
                "per_page": PER_PAGE,
                "page": page,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break

        for obs in results:
            for obs_photo in obs.get("observation_photos", []):
                photo = obs_photo.get("photo", {})
                license_code = photo.get("license_code")
                if license_code not in ALLOWED_LICENSES or not photo.get("url"):
                    continue
                photos.append(
                    {
                        "photo_id": photo["id"],
                        "url": photo["url"].replace("square.", f"{PHOTO_SIZE}."),
                        "license_code": license_code,
                        "attribution": photo.get("attribution", ""),
                        "observation_id": obs.get("id"),
                    }
                )

        if len(results) < PER_PAGE:
            break
        page += 1
        time.sleep(1)  # politesse envers l'API, pas de clé/quota documenté mais éviter le flood

    return photos[:max_images]
