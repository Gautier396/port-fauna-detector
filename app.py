"""Interface de test rapide : glisser une vidéo, détection avec le dernier
modèle entraîné (le `.pt` le plus récent dans `models/`), boxes + noms
d'espèces affichés au fil du traitement (pas seulement à la fin).

Usage:
    python app.py
Puis ouvrir l'URL locale affichée (http://127.0.0.1:7860 par défaut).
"""
import os
import sys
import types

# Conflit OpenMP connu sur cette machine Windows/Anaconda (cf. src/train.py) —
# à définir avant l'import de torch/ultralytics.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Un paquet PyPI sans rapport ("nn", une lib TensorFlow orpheline sur cette
# machine, cassée par une incompatibilité Keras -- rien dans ce projet n'en
# dépend, absent de requirements.txt) se fait importer de façon intermittente
# pendant le dépicklage d'un modèle .pt, uniquement quand ça arrive dans un
# thread worker de Gradio (reproduit et confirmé : le chargement réussit hors
# thread, ou quand "nn" est désinstallé). Réapparaît parfois tout seul sur
# cette machine (mécanisme non identifié, probablement un auto-install
# d'ultralytics) donc le désinstaller ne suffit pas durablement -- on
# neutralise l'import lui-même : si quelque chose fait `import nn` plus tard,
# Python trouvera ce module vide déjà en cache et n'ira jamais chercher le
# vrai paquet cassé sur le disque. Confirmé que le chargement du modèle n'a
# jamais réellement besoin de "nn" (chargement OK quand il est absent).
sys.modules.setdefault("nn", types.ModuleType("nn"))

from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

from src.preprocess import enhance_underwater

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo"


def latest_model_path() -> Path:
    candidates = sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Aucun modèle .pt dans {MODELS_DIR} — entraîne d'abord un modèle (src/train.py).")
    return candidates[0]


def run_detection(video_path: str, conf: float, confirm_frames: int, confirm_conf: float):
    """Générateur : une frame annotée à la fois pendant le traitement, puis la vidéo
    complète une fois terminé (gr.skip() laisse les sorties non concernées inchangées).

    Utilise model.track() (ByteTrack, comme src/track.py) plutôt que
    model.predict() : suivre chaque poisson d'une frame à l'autre est ce qui
    permet à la fois le compteur (compter les IDs distincts, pas les boîtes
    par frame — sinon un même poisson vu 50 frames compterait 50 fois) et la
    confirmation temporelle ci-dessous.

    Confirmation temporelle : une piste n'est dessinée qu'après avoir été
    vue au moins `confirm_frames` fois DE SUITE avec une confiance >=
    `confirm_conf` — évite qu'une fausse détection isolée (un seul mauvais
    frame) fasse clignoter une boîte à l'écran. Dès qu'une frame retombe
    sous `confirm_conf`, le compteur de la piste se réinitialise (pas de
    moyenne glissante qui laisserait passer une suite de détections
    fluctuantes)."""
    if not video_path:
        yield gr.skip(), gr.skip(), "Dépose une vidéo pour commencer."
        return

    try:
        model_path = latest_model_path()
    except FileNotFoundError as exc:
        yield gr.skip(), gr.skip(), str(exc)
        return

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    streak: dict[int, int] = {}       # track_id -> frames consécutives >= confirm_conf
    confirmed: set[int] = set()       # track_ids actuellement affichées
    seen_confirmed: set[int] = set()  # tous les track_ids confirmés depuis le début (compteur)

    n = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame = enhance_underwater(raw_frame)  # même correction que le dataset d'entraînement (src/preprocess.py)
        result = model.track(frame, conf=conf, persist=True, verbose=False)[0]
        n += 1

        annotated = frame.copy()
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id.item())
                confidence = float(box.conf.item())

                if confidence >= confirm_conf:
                    streak[track_id] = streak.get(track_id, 0) + 1
                else:
                    streak[track_id] = 0

                if streak[track_id] >= confirm_frames:
                    confirmed.add(track_id)
                    seen_confirmed.add(track_id)

                if track_id not in confirmed:
                    continue

                cls_id = int(box.cls.item())
                class_name = result.names[cls_id]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"{class_name} {confidence:.2f} #{track_id}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        cv2.putText(
            annotated, f"Poissons : {len(seen_confirmed)}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        )

        writer.write(annotated)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        total_str = f"/{total_frames}" if total_frames else ""
        yield annotated_rgb, gr.skip(), f"Modèle : {model_path.name} — frame {n}{total_str} — {len(seen_confirmed)} poisson(s) confirmé(s)"

    cap.release()
    writer.release()
    yield gr.skip(), output_path, f"Modèle : {model_path.name} — {n} frames traitées, {len(seen_confirmed)} poisson(s) au total -> {output_path}"


def build_interface() -> gr.Blocks:
    try:
        default_model = latest_model_path().name
    except FileNotFoundError as exc:
        default_model = str(exc)

    with gr.Blocks(title="Détecteur de faune marine du port") as demo:
        gr.Markdown(
            f"# Détecteur de faune marine du port — test rapide\n"
            f"Modèle utilisé (le plus récent dans `models/`) : **{default_model}**"
        )
        with gr.Row():
            video_in = gr.Video(label="Glisser une vidéo ici", sources=["upload"])
            with gr.Column():
                live_frame = gr.Image(label="Détection en direct", type="numpy")
                video_out = gr.Video(label="Vidéo annotée complète")
        conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Seuil de confiance")
        with gr.Row():
            confirm_frames_slider = gr.Slider(
                1, 15, value=3, step=1, label="Frames de confirmation",
                info="Nombre de frames consécutives requises avant d'afficher une boîte",
            )
            confirm_conf_slider = gr.Slider(
                0.05, 0.95, value=0.3, step=0.05, label="Confiance de confirmation",
                info="Confiance minimale à maintenir pendant ces frames (≥ seuil de confiance)",
            )
        status = gr.Markdown("")
        run_btn = gr.Button("Lancer la détection", variant="primary")
        run_btn.click(
            fn=run_detection,
            inputs=[video_in, conf_slider, confirm_frames_slider, confirm_conf_slider],
            outputs=[live_frame, video_out, status],
        )

    return demo


# Force le thème sombre au chargement, indépendamment de la préférence système
# du navigateur (mécanisme ?__theme=dark de Gradio) — sinon le thème sombre
# choisi ci-dessous ne s'applique que si le système est déjà en mode sombre.
FORCE_DARK_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.toString();
    }
}
"""

if __name__ == "__main__":
    build_interface().launch(theme=gr.themes.Default(), js=FORCE_DARK_JS)
