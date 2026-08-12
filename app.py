"""Interface de test rapide : glisser une vidéo, détection avec le dernier
modèle entraîné (le `.pt` le plus récent dans `models/`), boxes + noms
d'espèces affichés au fil du traitement (pas seulement à la fin).

Usage:
    python app.py
Puis ouvrir l'URL locale affichée (http://127.0.0.1:7860 par défaut).
"""
import os

# Conflit OpenMP connu sur cette machine Windows/Anaconda (cf. src/train.py) —
# à définir avant l'import de torch/ultralytics.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


def run_detection(video_path: str, conf: float):
    """Générateur : une frame annotée à la fois pendant le traitement, puis la vidéo
    complète une fois terminé (gr.skip() laisse les sorties non concernées inchangées)."""
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
    cap.release()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    n = 0
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame = enhance_underwater(raw_frame)  # même correction que le dataset d'entraînement (src/preprocess.py)
        result = model.predict(frame, conf=conf, verbose=False)[0]
        annotated_bgr = result.plot()  # boxes + noms + confiance déjà dessinés par ultralytics
        writer.write(annotated_bgr)
        n += 1
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        total_str = f"/{total_frames}" if total_frames else ""
        yield annotated_rgb, gr.skip(), f"Modèle : {model_path.name} — frame {n}{total_str}"

    cap.release()
    writer.release()
    yield gr.skip(), output_path, f"Modèle : {model_path.name} — {n} frames traitées, terminé -> {output_path}"


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
        status = gr.Markdown("")
        run_btn = gr.Button("Lancer la détection", variant="primary")
        run_btn.click(fn=run_detection, inputs=[video_in, conf_slider], outputs=[live_frame, video_out, status])

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
