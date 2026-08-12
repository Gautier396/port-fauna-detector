"""Orchestrateur bout-en-bout : une vidéo GoPro -> registre d'individus mis à jour.

Enchaîne les étapes [1], [4], [5], [6]/[7] pour une vidéo donnée. L'étape [8]
export.py se lance séparément une fois le registre à jour, pour l'ensemble du
registre plutôt qu'à chaque vidéo.

Usage:
    python pipeline.py --video data/raw/plongee1.mp4 --model models/best.pt
"""
import argparse
from pathlib import Path

from src import embeddings, registry, review_queue, track


def run_pipeline(
    video_path: Path,
    model_path: str,
    project_root: Path,
    conf: float,
    tracker: str,
    review_threshold: float,
    time_window_hours: float,
    threshold: float,
) -> None:
    video_stem = video_path.stem

    tracks_dir = project_root / "outputs" / "tracks"
    crops_dir = project_root / "outputs" / "crops" / video_stem
    tracks_dir.mkdir(parents=True, exist_ok=True)

    frames_df, tracks_df = track.run_tracking(model_path, video_path, conf, tracker, crops_dir, review_threshold)

    frames_path = tracks_dir / f"{video_stem}_frames.csv"
    tracks_path = tracks_dir / f"{video_stem}_tracks.csv"
    frames_df.to_csv(frames_path, index=False)
    tracks_df.to_csv(tracks_path, index=False)
    n_review = int(tracks_df["needs_review"].sum()) if not tracks_df.empty else 0
    print(f"[1/4] Tracking : {len(tracks_df)} tracks ({n_review} à vérifier) -> {tracks_path}")

    queue_path = project_root / "outputs" / "review_queue" / "queue.csv"
    added = review_queue.enqueue(tracks_df, queue_path)
    print(f"[2/4] File de vérification : {len(added)} track(s) mis en attente -> {queue_path}")

    embeddings_dir = project_root / "outputs" / "embeddings"
    tracks_df = embeddings.process_tracks(tracks_df, embeddings_dir)
    tracks_df.to_csv(tracks_path, index=False)
    n_emb = tracks_df["embedding_path"].notna().sum()
    print(f"[3/4] Embeddings : {n_emb}/{len(tracks_df)} calculés")

    db_path = project_root / "outputs" / "registry.db"
    conn = registry.init_db(str(db_path))
    result_df = registry.process_tracks_csv(
        conn, str(tracks_path), time_window_hours=time_window_hours, threshold=threshold
    )
    conn.close()

    n_created = (result_df["action"] == "created").sum()
    n_updated = (result_df["action"] == "updated").sum()
    print(f"[4/4] Registre : {n_created} nouveaux individus, {n_updated} rattachés à un individu existant -> {db_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Vidéo GoPro à traiter")
    parser.add_argument("--model", required=True, help="Poids YOLO (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance de détection")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Config tracker Ultralytics")
    parser.add_argument("--review-threshold", type=float, default=0.5, help="Confiance moyenne de track en-dessous de laquelle l'espèce est incertaine -> mis en file de vérification au lieu du registre (défaut: 0.5)")
    parser.add_argument("--time-window-hours", type=float, default=registry.DEFAULT_TIME_WINDOW_HOURS, help="Fenêtre temporelle anti-doublon (h)")
    parser.add_argument("--threshold", type=float, default=registry.DEFAULT_SIMILARITY_THRESHOLD, help="Seuil de similarité cosinus anti-doublon")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    run_pipeline(
        video_path=Path(args.video),
        model_path=args.model,
        project_root=project_root,
        conf=args.conf,
        tracker=args.tracker,
        review_threshold=args.review_threshold,
        time_window_hours=args.time_window_hours,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
