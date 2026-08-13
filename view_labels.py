"""Interface de consultation en lecture seule des labels YOLO — pas
d'édition, juste visualiser les boîtes pour contrôle qualité. Scanne aussi
les images pas encore réparties train/val (utile pendant un traitement en
cours).

Usage:
    python view_labels.py
Puis ouvrir http://127.0.0.1:7862
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from pathlib import Path

import cv2
import gradio as gr
import yaml

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "sfishtrack"
DATA_YAML_PATH = PROJECT_ROOT / "configs" / "data_sfishtrack.yaml"

# Lu depuis data_sfishtrack.yaml (classes réellement présentes dans les
# labels), pas configs/species.yaml (27 classes cibles) -- sinon le menu de
# filtre proposerait 26 espèces qui ne matcheraient jamais rien.
NAMES = yaml.safe_load(DATA_YAML_PATH.read_text(encoding="utf-8"))["names"]
NAME_TO_ID = {v: k for k, v in NAMES.items()}


def label_path_for(image_path: Path) -> Path:
    images_root = DATA_DIR / "images"
    labels_root = DATA_DIR / "labels"
    return labels_root / image_path.relative_to(images_root).with_suffix(".txt")


def list_labeled_images() -> list[Path]:
    """Images ayant un label correspondant, dans train/val ET à la racine
    (images pas encore réparties par un fetch en cours)."""
    images_root = DATA_DIR / "images"
    found = []
    for images_dir in (images_root, images_root / "train", images_root / "val"):
        if not images_dir.is_dir():
            continue
        for image_path in images_dir.glob("*.jpg"):
            if label_path_for(image_path).exists():
                found.append(image_path)
    return found


def image_classes(image_path: Path) -> set[int]:
    lines = label_path_for(image_path).read_text(encoding="utf-8").strip().splitlines()
    return {int(line.split()[0]) for line in lines if line.split()}


def draw_boxes(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, f"{image_path.name} — image illisible."
    h, w = img.shape[:2]
    lines = label_path_for(image_path).read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        name = NAMES.get(cls_id, str(cls_id))
        cv2.putText(img, name, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, f"{image_path.name} — {len(lines)} boîte(s)"


def refresh_pool(species_filter: str) -> list[Path]:
    images = list_labeled_images()
    if species_filter and species_filter != "toutes":
        target_id = NAME_TO_ID.get(species_filter)
        images = [p for p in images if target_id in image_classes(p)]
    random.shuffle(images)
    return images


def show_next(species_filter, pool, idx):
    pool = pool or []
    idx = idx or 0
    if not pool or idx >= len(pool):
        pool = refresh_pool(species_filter)
        idx = 0
    if not pool:
        return None, "Aucune image labellisée trouvée pour ce filtre.", pool, 0
    img, caption = draw_boxes(pool[idx])
    return img, caption, pool, idx + 1


def on_filter_change(species_filter):
    pool = refresh_pool(species_filter)
    return show_next(species_filter, pool, 0)


def build_interface() -> gr.Blocks:
    species_choices = ["toutes"] + [NAMES[k] for k in sorted(NAMES)]
    with gr.Blocks(title="Revue des labels") as demo:
        gr.Markdown(
            "# Revue des labels — `data/sfishtrack`\n"
            "Lecture seule (pas de suppression). Actualisé en direct."
        )
        with gr.Row():
            species_dd = gr.Dropdown(choices=species_choices, value="toutes", label="Filtrer par espèce")
            next_btn = gr.Button("Suivante ▶", variant="primary")
        image_out = gr.Image(label="Image + boîtes")
        caption_out = gr.Markdown("")
        pool_state = gr.State([])
        idx_state = gr.State(0)

        species_dd.change(fn=on_filter_change, inputs=[species_dd], outputs=[image_out, caption_out, pool_state, idx_state])
        next_btn.click(fn=show_next, inputs=[species_dd, pool_state, idx_state], outputs=[image_out, caption_out, pool_state, idx_state])
        demo.load(fn=on_filter_change, inputs=[species_dd], outputs=[image_out, caption_out, pool_state, idx_state])

    return demo


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
    build_interface().launch(server_port=7862, theme=gr.themes.Default(), js=FORCE_DARK_JS)
