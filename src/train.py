"""Entraîne un modèle YOLO de détection sur le jeu SFISHTRACK (configs/data_sfishtrack.yaml).

Choix par défaut, documentés plutôt qu'arbitraires :
  - `yolov8s.pt` (small) : SFISHTRACK est un jeu conséquent pour une seule
    classe (23 233 images, 147 582 boîtes, cf. README) — largement de quoi
    justifier `yolov8s` sans risque excessif de surapprentissage ; passer à
    `yolov8m`/`l` est raisonnable si la capacité s'avère limitante en
    pratique (mAP qui plafonne bien avant que la validation ne stagne).
  - `--epochs 200` / `--patience 100` : un premier run réel (2026-08-13,
    `portfauna_v3`, patience=20) s'est arrêté à l'époque 25 (meilleur
    résultat époque 5, mAP50-95=0.130) — mais le taux d'apprentissage
    d'ultralytics décroît sur la totalité de `--epochs` (ici 200), donc à
    l'époque 25 il n'avait décru que d'environ 12% : patience=20 a
    probablement coupé l'entraînement avant que le LR n'ait assez baissé
    pour permettre une convergence plus fine, pas parce que le modèle avait
    réellement atteint son plafond. `--patience 100` (la moitié de
    `--epochs`) garantit qu'un plateau précoce ne coupe plus avant que le
    LR ait significativement décru, tout en gardant un filet de sécurité si
    le modèle plafonne réellement après l'époque 100.
  - `--batch -1` (auto) : ultralytics choisit la taille de batch occupant
    ~60% de la VRAM disponible plutôt qu'une valeur arbitraire.
  - seed fixé (`--seed 42`) : reproductibilité.
  - Pas de split test séparé pour l'instant (seulement train/val) — à
    ajouter avant de publier des métriques comme définitives (cf. README,
    Points ouverts).

Sorties :
  - outputs/training_runs/<name>/ : logs, courbes, matrice de confusion,
    poids (best.pt/last.pt) — générés nativement par ultralytics.
  - models/<name>.pt : copie du meilleur poids, sous un nom versionné (pas
    d'écrasement silencieux d'un run précédent).

Usage:
    python src/train.py --data configs/data_sfishtrack.yaml --name portfauna_v3
"""
import argparse
import os
import shutil
from pathlib import Path

# Conflit connu OpenMP (numpy/MKL vs libiomp5md.dll) sur cette machine
# Windows/Anaconda — sans ce contournement, `import torch` plante avec
# "OMP: Error #15". Workaround standard, documenté par le message d'erreur
# lui-même ; à définir avant l'import de torch/ultralytics.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import nn.register  # noqa: E402,F401 -- effet de bord (enregistre EMC/GLSA/LSHDetect pour
# configs/bgle_yolo.yaml) ; sans effet sur les modèles standard (yolov8s.pt etc.). Import
# relatif au dossier de train.py (src/), pas "src.nn" -- ce script est toujours lancé via
# `python src/train.py` depuis la racine du repo, qui met src/ (pas la racine) sur sys.path.
from ultralytics import YOLO  # noqa: E402


def run_training(
    data_yaml: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    seed: int,
    device: str,
    run_name: str,
    project_dir: Path,
) -> tuple[YOLO, Path]:
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        device=device,
        # ultralytics only honors `project` as given if it's absolute — a relative
        # path gets silently prefixed with the global runs_dir setting instead
        # (ultralytics/cfg/__init__.py:get_save_dir), landing outside this repo.
        project=str(project_dir.resolve()),
        name=run_name,
        exist_ok=False,
    )
    run_dir = project_dir / run_name
    return model, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="configs/data_sfishtrack.yaml", help="data.yaml Ultralytics (défaut: configs/data_sfishtrack.yaml)")
    parser.add_argument("--model", default="yolov8s.pt", help="Poids de départ (défaut: yolov8s.pt, pré-entraîné COCO)")
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'époques max (défaut: 200, coupé plus tôt par --patience si la validation plafonne avant)")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image (défaut: 640)")
    parser.add_argument("--batch", type=int, default=-1, help="Taille de batch, -1 = auto ~60%% VRAM (défaut: -1)")
    parser.add_argument("--patience", type=int, default=100, help="Arrêt anticipé si la validation stagne N époques (défaut: 100 -- cf. docstring, un patience trop bas coupe avant que le LR schedule ait assez décru)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire (défaut: 42)")
    parser.add_argument("--device", default=None, help="Device (défaut: 0 si CUDA disponible, sinon cpu)")
    parser.add_argument("--name", required=True, help="Nom du run (dossier de sortie + nom du modèle final)")
    parser.add_argument("--project-dir", default="outputs/training_runs", help="Dossier racine des runs (défaut: outputs/training_runs)")
    parser.add_argument("--models-dir", default="models", help="Dossier de sortie du poids final versionné (défaut: models)")
    args = parser.parse_args()

    import torch

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else " (CPU — ce sera lent)"))

    model, run_dir = run_training(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=device,
        run_name=args.name,
        project_dir=Path(args.project_dir),
    )

    best_pt = run_dir / "weights" / "best.pt"
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / f"{args.name}.pt"
    shutil.copy(best_pt, final_model_path)

    print(f"\nModèle final -> {final_model_path}")
    print(f"Logs/courbes/matrice de confusion -> {run_dir}")

    print("\nÉvaluation finale sur le split val :")
    metrics = YOLO(str(best_pt)).val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"  mAP50    : {metrics.box.map50:.3f}")
    print(f"  mAP50-95 : {metrics.box.map:.3f}")
    print(f"  Précision (moyenne) : {metrics.box.mp:.3f}")
    print(f"  Rappel (moyen)      : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
