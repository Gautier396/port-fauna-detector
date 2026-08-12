"""Referme la boucle de la file de vérification (review_queue.py).

Une fois la colonne `resolved_species` remplie pour une ligne `status=pending`
du CSV de la file — à la main, ou par un agent externe qui écrit dans ce même
fichier — ce script calcule l'embedding du crop et enregistre l'individu dans
le registre (registry.py), comme registry.py l'aurait fait directement si le
modèle avait été assez confiant. La ligne passe alors à `status=resolved`.

Usage:
    # 1. Remplir resolved_species sur les lignes status=pending du CSV.
    # 2. python resolve_review.py --queue-csv outputs/review_queue/queue.csv --db outputs/registry.db
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import registry
from embeddings import embed_image


def resolve(queue_df: pd.DataFrame, conn, embeddings_dir: Path, **registry_kwargs) -> pd.DataFrame:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    queue_df = queue_df.copy()
    pending = queue_df["status"].eq("pending") & queue_df["resolved_species"].fillna("").ne("")

    for idx in queue_df[pending].index:
        row = queue_df.loc[idx]

        embedding, embedding_path = None, None
        crop_path = row["best_crop_path"]
        if isinstance(crop_path, str) and Path(crop_path).exists():
            embedding = embed_image(crop_path)
            embedding_path = str(embeddings_dir / f"review_{Path(row['source_video']).stem}_track{int(row['track_id']):04d}.npy")
            np.save(embedding_path, embedding)

        timestamp = pd.Timestamp(row["first_seen_at"]) if pd.notna(row["first_seen_at"]) else None

        registry.register_or_update(
            conn,
            species=row["resolved_species"],
            embedding=embedding,
            confidence=row.get("confidence", 0.0),
            embedding_path=embedding_path,
            thumbnail_path=crop_path,
            source_video=row["source_video"],
            track_id=int(row["track_id"]),
            timestamp=timestamp,
            **registry_kwargs,
        )
        queue_df.loc[idx, "status"] = "resolved"

    return queue_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue-csv", required=True, help="CSV de la file de vérification (review_queue.py)")
    parser.add_argument("--db", required=True, help="Chemin de la base SQLite du registre")
    parser.add_argument("--embeddings-dir", default="outputs/embeddings", help="Dossier de sortie pour les .npy (défaut: outputs/embeddings)")
    parser.add_argument("--time-window-hours", type=float, default=registry.DEFAULT_TIME_WINDOW_HOURS, help="Fenêtre temporelle anti-doublon (h)")
    parser.add_argument("--threshold", type=float, default=registry.DEFAULT_SIMILARITY_THRESHOLD, help="Seuil de similarité cosinus anti-doublon")
    args = parser.parse_args()

    queue_df = pd.read_csv(args.queue_csv)
    conn = registry.init_db(args.db)
    result_df = resolve(
        queue_df,
        conn,
        Path(args.embeddings_dir),
        time_window_hours=args.time_window_hours,
        threshold=args.threshold,
    )
    conn.close()

    result_df.to_csv(args.queue_csv, index=False)

    n_resolved = (result_df["status"] == "resolved").sum()
    n_pending = (result_df["status"] == "pending").sum()
    print(f"{n_resolved} track(s) résolu(s) -> enregistrés dans {args.db}")
    print(f"{n_pending} track(s) encore en attente (resolved_species non rempli)")


if __name__ == "__main__":
    main()
