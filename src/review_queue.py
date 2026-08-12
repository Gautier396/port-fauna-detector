"""File d'attente de vérification pour les tracks à espèce incertaine.

track.py marque `needs_review=True` sur un track dont la confiance moyenne
reste sous --review-threshold : le modèle a détecté quelque chose mais n'est
pas assez sûr de l'espèce pour l'enregistrer directement dans le registre
(registry.py, qui ignore ces tracks). Ce script les ajoute plutôt à une file
CSV (crop, vidéo, horodatage, meilleure estimation du modèle) en attente
d'identification manuelle ou par un agent externe — cf. resolve_review.py
pour la remise en registre une fois l'espèce confirmée.

Usage (bibliothèque, appelé depuis pipeline.py) :
    from review_queue import enqueue

Usage (CLI, à partir d'un CSV de tracks avec colonne needs_review) :
    python review_queue.py --tracks-csv outputs/tracks/plongee1_tracks.csv \\
        --queue-csv outputs/review_queue/queue.csv
"""
import argparse
from pathlib import Path

import pandas as pd

QUEUE_COLUMNS = [
    "source_video", "track_id", "best_crop_path", "predicted_species",
    "confidence", "first_seen_at",
    "status", "resolved_species",
]


def enqueue(tracks_df: pd.DataFrame, queue_csv: Path) -> pd.DataFrame:
    """Ajoute à la file (créée si besoin, sinon complétée) les tracks needs_review. Retourne les lignes ajoutées."""
    if "needs_review" not in tracks_df.columns:
        return tracks_df.iloc[0:0]

    to_review = tracks_df[tracks_df["needs_review"] == True]  # noqa: E712 (comparaison explicite, colonne peut être object après relecture CSV)
    if to_review.empty:
        return to_review

    rows = pd.DataFrame(
        {
            "source_video": to_review["source_video"],
            "track_id": to_review["track_id"],
            "best_crop_path": to_review["best_crop_path"],
            "predicted_species": to_review["species"],
            "confidence": to_review["mean_confidence"],
            "first_seen_at": to_review["first_seen_at"],
            "status": "pending",
            "resolved_species": "",
        }
    )

    queue_csv.parent.mkdir(parents=True, exist_ok=True)
    if queue_csv.exists():
        rows = pd.concat([pd.read_csv(queue_csv), rows], ignore_index=True)
    rows.to_csv(queue_csv, index=False)
    return rows.tail(len(to_review))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks-csv", required=True, help="CSV _tracks.csv produit par track.py (colonne needs_review)")
    parser.add_argument("--queue-csv", required=True, help="CSV de la file de vérification (créé/complété)")
    args = parser.parse_args()

    tracks_df = pd.read_csv(args.tracks_csv)
    added = enqueue(tracks_df, Path(args.queue_csv))

    print(f"{len(added)} track(s) ajouté(s) à la file de vérification -> {args.queue_csv}")


if __name__ == "__main__":
    main()
