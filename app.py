"""Interface de test rapide : glisser une vidéo, détection avec le modèle
YOLOv8 (benchmark, cf. README), boxes + noms d'espèces affichés au fil du
traitement (pas seulement à la fin).

Usage:
    python app.py
Puis ouvrir l'URL locale affichée (http://127.0.0.1:7860 par défaut).
"""
import os
from pathlib import Path

# Conflit OpenMP connu sur cette machine Windows/Anaconda (cf. src/train.py) —
# à définir avant l'import de torch/ultralytics.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import src.nn.register  # noqa: E402,F401 -- effet de bord, enregistre EMC/GLSA/LSHDetect
# pour les checkpoints BGLE-YOLO (cf. src/nn/) ; sans effet sur les modèles standard.

import cv2
import gradio as gr
import torch
from ultralytics import YOLO

from src.preprocess import enhance_underwater

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo"
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# Modèle YOLOv8 (benchmark, cf. README) -- pas de sélection automatique du
# .pt le plus récent dans models/ : ce dossier contient aussi des
# checkpoints BGLE-YOLO (architecture expérimentale, moins performante pour
# l'instant), qu'un choix basé sur la date de modification finirait par
# préférer par accident.
MODEL_NAME = "portfauna_v3.pt"


def demo_model_path() -> Path:
    path = MODELS_DIR / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — entraîne d'abord ce modèle (src/train.py).")
    return path


def run_detection(video_path: str, conf: float):
    """Générateur : une frame annotée à la fois pendant le traitement, puis la vidéo
    complète une fois terminé (gr.skip() laisse les sorties non concernées inchangées).

    Utilise model.track() (ByteTrack) plutôt que model.predict() : IDs
    stables d'une frame à l'autre pour repérer visuellement un même poisson
    pendant le suivi."""
    if not video_path:
        yield gr.skip(), gr.skip(), "Dépose une vidéo pour commencer."
        return

    try:
        model_path = demo_model_path()
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

    n = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame = enhance_underwater(raw_frame)  # même correction que le dataset d'entraînement (src/preprocess.py)
        result = model.track(
            frame, conf=conf, persist=True, verbose=False, device=DEVICE,
            quantize=16 if DEVICE != "cpu" else None,  # FP16 : ~1.4x plus rapide sur GPU, non supporté sur CPU
        )[0]
        n += 1

        annotated = frame.copy()
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id.item())
                confidence = float(box.conf.item())

                cls_id = int(box.cls.item())
                class_name = result.names[cls_id]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"{class_name} {confidence:.2f} #{track_id}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        writer.write(annotated)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        total_str = f"/{total_frames}" if total_frames else ""
        yield annotated_rgb, gr.skip(), f"Modèle : {model_path.name} ({DEVICE}) — frame {n}{total_str}"

    cap.release()
    writer.release()
    yield gr.skip(), output_path, f"Modèle : {model_path.name} ({DEVICE}) — {n} frames traitées -> {output_path}"


def build_interface() -> gr.Blocks:
    try:
        default_model = demo_model_path().name
    except FileNotFoundError as exc:
        default_model = str(exc)

    with gr.Blocks(title="Détecteur de faune marine du port") as demo:
        gr.Markdown(
            f"# Détecteur de faune marine du port — test rapide\n"
            f"Modèle utilisé : **{default_model}**"
        )
        with gr.Row():
            video_in = gr.Video(label="Glisser une vidéo ici", sources=["upload"])
            with gr.Column():
                live_frame = gr.Image(label="Détection en direct", type="numpy")
                video_out = gr.Video(label="Vidéo annotée complète")
        conf_slider = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Seuil de confiance")
        status = gr.Markdown("")
        run_btn = gr.Button("Lancer la détection", variant="primary")
        run_btn.click(
            fn=run_detection,
            inputs=[video_in, conf_slider],
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
