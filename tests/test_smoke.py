"""Tests de fumée : pas de dataset ni de GPU nécessaires (CI-friendly),
juste de quoi prouver que le code s'exécute réellement plutôt que de
sembler correct en lecture. Racine du repo attendue comme cwd (pytest
depuis la racine, comme les autres commandes du projet).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml

import src.nn.register  # noqa: F401 -- effet de bord, requis avant YOLO(...)
from src.nn.bgle_modules import LSHDetect
from src.preprocess import gray_world_white_balance


def test_configs_parse():
    for path in ["configs/data_sfishtrack.yaml", "configs/species.yaml", "configs/bgle_yolo.yaml"]:
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) is not None


def test_bgle_yolo_builds_and_runs():
    from ultralytics import YOLO

    model = YOLO("configs/bgle_yolo.yaml").model
    model.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    assert torch.isfinite(pred).all()


def test_lshdetect_box_branch_is_actually_shared():
    """Régression : la branche box doit réutiliser UNE seule instance de
    poids à travers les 3 échelles P3/P4/P5 (cf. historique du dépôt --
    une version antérieure ne la partageait que par coïncidence sur une
    seule échelle)."""
    from ultralytics import YOLO

    model = YOLO("configs/bgle_yolo.yaml").model
    head = next(m for m in model.modules() if isinstance(m, LSHDetect))
    shared = [branch.shared for branch in head.cv2]
    assert all(s is shared[0] for s in shared)


def test_white_balance_preserves_shape_and_dtype():
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    corrected = gray_world_white_balance(image)
    assert corrected.shape == image.shape
    assert corrected.dtype == image.dtype
