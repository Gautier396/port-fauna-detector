"""Entraîne un modèle YOLO de détection sur le jeu SFISHTRACK (configs/data_sfishtrack.yaml).

Choix par défaut, documentés plutôt qu'arbitraires :
  - `yolov8s.pt` (small) : SFISHTRACK est un jeu conséquent pour une seule
    classe (23 233 images, 147 582 boîtes, cf. README) — largement de quoi
    justifier `yolov8s` sans risque excessif de surapprentissage ; passer à
    `yolov8m`/`l` est raisonnable si la capacité s'avère limitante en
    pratique (mAP qui plafonne bien avant que la validation ne stagne).
  - `--epochs 200` / `--patience 100` : le taux d'apprentissage d'ultralytics
    décroît sur la totalité de `--epochs`, donc une patience trop basse
    relativement à `--epochs` coupe l'entraînement avant que le LR ait
    assez baissé pour permettre une convergence fine, avant même que le
    modèle ait atteint son plafond réel. `--patience 100` (la moitié de
    `--epochs`) laisse le LR décroître significativement avant qu'un
    plateau précoce ne déclenche l'arrêt anticipé.
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

Usage (depuis la racine du repo, pour que "src" résolve comme un package) :
    python -m src.train --data configs/data_sfishtrack.yaml --name portfauna_v3
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

import src.nn.register  # noqa: E402,F401 -- effet de bord (enregistre EMC/GLSA/LSHDetect
# pour configs/bgle_yolo.yaml) ; sans effet sur les modèles standard (yolov8s.pt etc.).
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
    fraction: float = 1.0,
    workers: int = 8,
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
        fraction=fraction,
        workers=workers,
        # ultralytics only honors `project` as given if it's absolute — a relative
        # path gets silently prefixed with the global runs_dir setting instead
        # (ultralytics/cfg/__init__.py:get_save_dir), landing outside this repo.
        project=str(project_dir.resolve()),
        name=run_name,
        exist_ok=False,
    )
    run_dir = project_dir / run_name
    return model, run_dir


def resume_training(last_pt: Path) -> tuple[YOLO, Path]:
    """Reprend un entraînement interrompu depuis son `last.pt`. Contrairement à
    `run_training()`, `resume=True` fait relire à ultralytics tous les
    paramètres d'origine (data/epochs/imgsz/patience/...) depuis le
    `args.yaml` du run, et restaure aussi l'état de l'optimiseur + le
    schedule LR exactement là où il s'était arrêté -- reprendre "à froid"
    avec un nouveau run (au lieu de resume=True) repartirait du LR de pic,
    annulant toute la décroissance déjà faite. Écrit dans le MÊME dossier de
    run (results.csv continue, pas de nouveau dossier)."""
    model = YOLO(str(last_pt))
    model.train(resume=True)
    return model, last_pt.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="configs/data_sfishtrack.yaml", help="data.yaml Ultralytics (défaut: configs/data_sfishtrack.yaml)")
    parser.add_argument("--model", default="yolov8s.pt", help="Poids de départ (défaut: yolov8s.pt, pré-entraîné COCO)")
    parser.add_argument("--epochs", type=int, default=200, help="Nombre d'époques max (défaut: 200, coupé plus tôt par --patience si la validation plafonne avant)")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image (défaut: 640)")
    parser.add_argument("--batch", type=int, default=-1, help="Taille de batch, -1 = auto ~60%% VRAM (défaut: -1)")
    parser.add_argument("--patience", type=int, default=100, help="Arrêt anticipé si la validation stagne N époques (défaut: 100 -- cf. docstring, un patience trop bas coupe avant que le LR schedule ait assez décru)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire (défaut: 42)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction du train set à utiliser, pour un test rapide (défaut: 1.0 = tout)")
    parser.add_argument("--workers", type=int, default=8, help="Workers de chargement de données en parallèle (défaut: 8 -- baisser si RAM système limitée, cf. Points ouverts)")
    parser.add_argument("--device", default=None, help="Device (défaut: 0 si CUDA disponible, sinon cpu)")
    parser.add_argument("--name", required=True, help="Nom du run (dossier de sortie + nom du modèle final)")
    parser.add_argument("--project-dir", default="outputs/training_runs", help="Dossier racine des runs (défaut: outputs/training_runs)")
    parser.add_argument("--models-dir", default="models", help="Dossier de sortie du poids final versionné (défaut: models)")
    parser.add_argument("--resume", action="store_true", help="Reprend le run --name interrompu depuis son weights/last.pt (ignore --data/--model/--epochs/etc., repris depuis l'args.yaml du run d'origine)")
    args = parser.parse_args()

    import torch

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else " (CPU — ce sera lent)"))

    if args.resume:
        last_pt = Path(args.project_dir) / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(f"--resume : {last_pt} introuvable -- rien à reprendre pour --name {args.name!r}.")
        print(f"Reprise depuis {last_pt}")
        model, run_dir = resume_training(last_pt)
    else:
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
            fraction=args.fraction,
            workers=args.workers,
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
