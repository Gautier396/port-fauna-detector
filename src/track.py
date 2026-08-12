"""Détection + tracking intra-vidéo (ByteTrack).

Lit la vidéo frame par frame (cv2.VideoCapture) plutôt que de laisser
ultralytics décoder la source lui-même : chaque frame passe par
`enhance_underwater()` (src/preprocess.py, même correction que sur le
dataset d'entraînement) avant d'être donnée à model.track(frame,
persist=True, ...) — persist=True maintient l'état ByteTrack (IDs stables)
d'un appel à l'autre, comme documenté par ultralytics pour ce pattern.

Pas de géolocalisation (GPS retiré du projet, expérimental) : le timestamp
attribué à chaque track est la date de modification du fichier vidéo (proxy
approximatif de la date d'enregistrement pour un fichier GoPro), identique
pour toute la vidéo — pas un horodatage par frame.

Pour chaque track_id, on conserve le crop de la frame où la confiance de
détection est la plus haute ("meilleur frame du track") — c'est ce crop qui
sera utilisé par embeddings.py pour représenter l'individu.

Si la confiance moyenne d'un track reste sous --review-threshold, le modèle
n'est pas assez sûr de l'espèce pour l'enregistrer directement dans le
registre : le track est marqué `needs_review=True` (colonne dans
<output>_tracks.csv) et sera mis en file d'attente par review_queue.py au
lieu d'être traité par registry.py — cf. review_queue.py / resolve_review.py.

Sorties :
  - <output>_frames.csv   : une ligne par détection (frame, track_id, bbox)
  - <output>_tracks.csv   : une ligne par track_id (agrégat, pour registry.py),
                            colonne needs_review incluse
  - <crops-dir>/track_<id>.jpg : meilleur crop par track

Usage:
    python track.py --model models/best.pt --video data/raw/plongee1.mp4 \\
        --output outputs/tracks/plongee1 --crops-dir outputs/crops/plongee1
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.preprocess import enhance_underwater


def run_tracking(
    model_path: str,
    video_path: Path,
    conf: float,
    tracker: str,
    crops_dir: Path,
    review_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    crops_dir.mkdir(parents=True, exist_ok=True)
    video_timestamp = pd.Timestamp(video_path.stat().st_mtime, unit="s")

    frame_rows = []
    best_conf_by_track: dict[int, float] = {}
    track_meta: dict[int, dict] = {}

    # Boucle frame-à-frame manuelle (plutôt que source=video_path, stream=True)
    # pour pouvoir prétraiter (enhance_underwater) chaque frame avant détection —
    # persist=True maintient l'état du tracker ByteTrack entre les appels, comme
    # documenté par ultralytics pour ce pattern.
    frame_idx = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = enhance_underwater(raw_frame)
        elapsed_s = frame_idx / fps
        result = model.track(frame, conf=conf, tracker=tracker, persist=True, verbose=False)[0]
        frame_idx += 1

        if result.boxes is None or result.boxes.id is None:
            continue

        for box in result.boxes:
            track_id = int(box.id.item())
            cls_id = int(box.cls.item())
            class_name = result.names[cls_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            frame_rows.append(
                {
                    "source_video": video_path.name,
                    "frame_idx": frame_idx,
                    "elapsed_s": elapsed_s,
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            )

            if track_id not in track_meta:
                track_meta[track_id] = {
                    "source_video": video_path.name,
                    "track_id": track_id,
                    "species": class_name,
                    "first_seen_elapsed_s": elapsed_s,
                    "last_seen_elapsed_s": elapsed_s,
                    "first_seen_at": video_timestamp,
                    "last_seen_at": video_timestamp,
                    "n_detections": 0,
                    "mean_confidence": 0.0,
                    "best_crop_path": None,
                }

            meta = track_meta[track_id]
            meta["last_seen_elapsed_s"] = elapsed_s
            meta["n_detections"] += 1
            meta["mean_confidence"] += confidence

            if confidence > best_conf_by_track.get(track_id, -1.0):
                best_conf_by_track[track_id] = confidence
                crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
                if crop.size > 0:
                    crop_path = crops_dir / f"track_{track_id:04d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    meta["best_crop_path"] = str(crop_path)

    cap.release()
    frames_df = pd.DataFrame(frame_rows)

    tracks_rows = []
    for meta in track_meta.values():
        meta["mean_confidence"] /= meta["n_detections"]
        meta["needs_review"] = meta["mean_confidence"] < review_threshold
        tracks_rows.append(meta)
    tracks_df = pd.DataFrame(tracks_rows)

    return frames_df, tracks_df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Poids YOLO (.pt)")
    parser.add_argument("--video", required=True, help="Fichier vidéo source")
    parser.add_argument("--output", required=True, help="Préfixe de sortie (sans extension) pour les CSV")
    parser.add_argument("--crops-dir", required=True, help="Dossier où sauvegarder le meilleur crop par track")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance de détection (défaut: 0.25)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Config tracker Ultralytics (défaut: bytetrack.yaml)")
    parser.add_argument("--review-threshold", type=float, default=0.5, help="Confiance moyenne de track en-dessous de laquelle l'espèce est incertaine -> needs_review=True (défaut: 0.5)")
    args = parser.parse_args()

    video_path = Path(args.video)

    frames_df, tracks_df = run_tracking(
        args.model, video_path, args.conf, args.tracker, Path(args.crops_dir), args.review_threshold
    )

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frames_path = output_prefix.with_name(output_prefix.name + "_frames.csv")
    tracks_path = output_prefix.with_name(output_prefix.name + "_tracks.csv")
    frames_df.to_csv(frames_path, index=False)
    tracks_df.to_csv(tracks_path, index=False)

    n_review = int(tracks_df["needs_review"].sum()) if not tracks_df.empty else 0
    print(f"{video_path.name}: {len(tracks_df)} tracks ({n_review} à vérifier, confiance < {args.review_threshold}), {len(frames_df)} détections")
    print(f"-> {frames_path}")
    print(f"-> {tracks_path}")


if __name__ == "__main__":
    main()
