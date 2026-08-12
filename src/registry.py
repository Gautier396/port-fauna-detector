"""Registre SQLite des individus observés + logique anti-doublon.

Un individu est une ligne de la table `individuals`. Un nouveau track
(intra-vidéo, ID stable garanti par ByteTrack dans track.py) n'est enregistré
comme *nouvel* individu que s'il ne correspond à aucun individu déjà connu :
la mise en correspondance se fait par similarité cosinus des embeddings
visuels (embeddings.py), restreinte aux individus de la même espèce situés
dans une fenêtre temporelle donnée (candidats plausibles seulement —
comparer les embeddings de tout le registre serait à la fois plus lent et
plus sujet aux faux positifs entre sorties/plongées différentes).

Pas de GPS (retiré du projet, expérimental) : l'anti-doublon repose
uniquement sur l'espèce, une fenêtre temporelle et la similarité
d'embedding — pas de contrainte spatiale.

Usage (bibliothèque, dans un pipeline) :
    from registry import init_db, register_or_update

Usage (CLI, à partir d'un CSV de tracks avec colonne embedding_path) :
    python registry.py --tracks-csv outputs/tracks/plongee1_tracks.csv --db outputs/registry.db
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS individuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species TEXT NOT NULL,
    first_seen_at TEXT,
    last_seen_at TEXT,
    confidence REAL,
    embedding_path TEXT,
    thumbnail_path TEXT,
    source_video TEXT,
    track_id INTEGER
);
"""

DEFAULT_TIME_WINDOW_HOURS = 24.0
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def init_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_candidates(
    conn: sqlite3.Connection,
    species: str,
    timestamp: "pd.Timestamp | None",
    time_window_hours: float,
) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM individuals WHERE species = ?", (species,)).fetchall()

    candidates = []
    for row in rows:
        if timestamp is not None and row["last_seen_at"]:
            delta_h = abs((timestamp - pd.Timestamp(row["last_seen_at"])).total_seconds()) / 3600
            if delta_h > time_window_hours:
                continue
        candidates.append(row)
    return candidates


def register_or_update(
    conn: sqlite3.Connection,
    species: str,
    embedding: np.ndarray | None,
    confidence: float,
    embedding_path: str | None,
    thumbnail_path: str | None,
    source_video: str,
    track_id: int,
    timestamp: "pd.Timestamp | None",
    time_window_hours: float = DEFAULT_TIME_WINDOW_HOURS,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[int, str, float | None]:
    """Retourne (individual_id, "created"|"updated", meilleure_similarité_trouvée)."""
    timestamp_str = timestamp.isoformat() if timestamp is not None else None

    candidates = find_candidates(conn, species, timestamp, time_window_hours)

    best_id, best_sim = None, -1.0
    if embedding is not None:
        for row in candidates:
            if not row["embedding_path"] or not Path(row["embedding_path"]).exists():
                continue
            cand_embedding = np.load(row["embedding_path"])
            sim = cosine_similarity(embedding, cand_embedding)
            if sim > best_sim:
                best_sim, best_id = sim, row["id"]

    if best_id is not None and best_sim >= threshold:
        conn.execute(
            """UPDATE individuals SET last_seen_at = ?, confidence = MAX(confidence, ?)
               WHERE id = ?""",
            (timestamp_str, confidence, best_id),
        )
        conn.commit()
        return best_id, "updated", best_sim

    cursor = conn.execute(
        """INSERT INTO individuals
           (species, first_seen_at, last_seen_at, confidence,
            embedding_path, thumbnail_path, source_video, track_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (species, timestamp_str, timestamp_str, confidence,
         embedding_path, thumbnail_path, source_video, track_id),
    )
    conn.commit()
    return cursor.lastrowid, "created", (best_sim if candidates else None)


def process_tracks_csv(conn: sqlite3.Connection, tracks_csv: str, **kwargs) -> pd.DataFrame:
    """Enregistre les tracks dans le registre, à l'exception de ceux marqués needs_review
    (espèce jugée trop incertaine par track.py) — ceux-là sont mis en file d'attente par
    review_queue.py et ne doivent pas polluer le registre avec une espèce non fiable."""
    tracks_df = pd.read_csv(tracks_csv, parse_dates=["first_seen_at", "last_seen_at"])
    results = []

    for _, row in tracks_df.iterrows():
        if row.get("needs_review", False):
            continue

        embedding = None
        emb_path = row.get("embedding_path")
        if isinstance(emb_path, str) and Path(emb_path).exists():
            embedding = np.load(emb_path)

        timestamp = row.get("first_seen_at")
        if pd.isna(timestamp):
            timestamp = None

        individual_id, action, similarity = register_or_update(
            conn,
            species=row["species"],
            embedding=embedding,
            confidence=row.get("mean_confidence", 0.0),
            embedding_path=emb_path if isinstance(emb_path, str) else None,
            thumbnail_path=row.get("best_crop_path"),
            source_video=row["source_video"],
            track_id=int(row["track_id"]),
            timestamp=timestamp,
            **kwargs,
        )
        results.append({"track_id": row["track_id"], "individual_id": individual_id, "action": action, "similarity": similarity})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks-csv", required=True, help="CSV _tracks.csv (avec embedding_path) issu de embeddings.py")
    parser.add_argument("--db", required=True, help="Chemin de la base SQLite du registre")
    parser.add_argument("--time-window-hours", type=float, default=DEFAULT_TIME_WINDOW_HOURS, help="Fenêtre temporelle de recherche de doublons, en heures")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Seuil de similarité cosinus pour fusionner avec un individu existant")
    args = parser.parse_args()

    conn = init_db(args.db)
    result_df = process_tracks_csv(
        conn,
        args.tracks_csv,
        time_window_hours=args.time_window_hours,
        threshold=args.threshold,
    )
    conn.close()

    n_created = (result_df["action"] == "created").sum()
    n_updated = (result_df["action"] == "updated").sum()
    print(f"{len(result_df)} tracks traités : {n_created} nouveaux individus, {n_updated} rattachés à un individu existant")
    print(f"Registre : {args.db}")


if __name__ == "__main__":
    main()
