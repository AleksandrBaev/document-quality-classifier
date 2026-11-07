"""Task 3 - YOLOv8 classification on data/quality_split (Task 2 split, no re-splitting).

Trains yolov8n-cls and yolov8s-cls with the fixed parameters:
    epochs=50, patience=5, imgsz=640, batch=32, seed=2473057 (student ID), device=cpu.

Evaluation is done on the TEST split only. Metrics (accuracy, precision, recall,
F1 per class and macro, confusion matrix) are computed from the model's own
probabilities with sklearn. Latency follows the course protocol: 20 warm-up
predictions, then average over 100 timed predictions.

Exact commands (reproducibility, R8):
    python scripts/train_yolo_quality.py

One results.json per variant is written to results/week7_yolov8{n,s}_cls/.
Ultralytics training plots (results.png, confusion_matrix*.png) and misclassified
test images (errors/) are kept as well.
"""

import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from ultralytics import YOLO

SEED = 2473057  # student ID
EPOCHS = 50
PATIENCE = 5
IMGSZ = 640
BATCH = 32
DEVICE = "cpu"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "quality_split"
RESULTS = ROOT / "results"

torch.manual_seed(SEED)
np.random.seed(SEED)


def evaluate_and_save(model_path, variant, run_dir):
    """Evaluate best.pt on the test split and write results.json."""
    results_dir = RESULTS / f"week7_{variant}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    # ---- collect predictions on the test split only ----
    classes = ["bad", "good"]  # folder names = class names, alphabetical
    y_true, y_prob = [], []
    test_files = []
    for ci, cls in enumerate(classes):
        for f in sorted((DATA / "test" / cls).glob("*")):
            test_files.append((f, ci))

    for f, ci in test_files:
        r = model.predict(str(f), imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        y_prob.append(probs)
        y_true.append(ci)

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_pc, r_pc, f1_pc, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    per_class = {
        cls: {
            "precision": float(p_pc[i]),
            "recall": float(r_pc[i]),
            "f1": float(f1_pc[i]),
            "support": int(sup[i]),
        }
        for i, cls in enumerate(classes)
    }

    # ---- confusion matrix image ----
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{variant} - test split confusion matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---- copy misclassified test images ----
    errors_dir = results_dir / "errors"
    errors_dir.mkdir(exist_ok=True)
    n_errors = 0
    for (f, ci), pi in zip(test_files, y_pred):
        if pi != ci:
            dst = errors_dir / f"true_{classes[ci]}__pred_{classes[pi]}__{f.name}"
            shutil.copy2(f, dst)
            n_errors += 1

    # ---- latency: 20 warm-up, 100 timed ----
    sample = str(test_files[0][0])
    for _ in range(20):
        model.predict(sample, imgsz=IMGSZ, device=DEVICE, verbose=False)
    t0 = time.perf_counter()
    N_TIMED = 100
    for _ in range(N_TIMED):
        model.predict(sample, imgsz=IMGSZ, device=DEVICE, verbose=False)
    latency_ms = (time.perf_counter() - t0) / N_TIMED * 1000.0

    # ---- cost metrics ----
    size_mb = model_path.stat().st_size / 1024 / 1024
    params = sum(p.numel() for p in model.model.parameters())

    # ---- training history + actual epochs from ultralytics results.csv ----
    results_csv = run_dir / "results.csv"
    history = {}
    epochs_run = EPOCHS
    if results_csv.exists():
        lines = results_csv.read_text().strip().splitlines()
        epochs_run = len(lines) - 1
        header = [h.strip() for h in lines[0].split(",")]
        cols = {h: [] for h in header}
        for ln in lines[1:]:
            for h, v in zip(header, ln.split(",")):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    pass
        history = {
            "train_loss": cols.get("train/loss", []),
            "val_loss": cols.get("val/loss", []),
            "top1_acc": cols.get("metrics/accuracy_top1", []),
            "top5_acc": cols.get("metrics/accuracy_top5", []),
        }

    payload = {
        "experiment": f"week7_{variant}",
        "week": 7,
        "model": variant.replace("_", "-"),
        "regime": "full_finetune",
        "input_size": IMGSZ,
        "seed": SEED,
        "device": DEVICE,
        "epochs_planned": EPOCHS,
        "epochs_run": int(epochs_run),
        "train_time_s": float(getattr(evaluate_and_save, "train_time_s", 0.0)),
        "latency_ms_per_image": float(latency_ms),
        "model_size_mb": float(size_mb),
        "params": int(params),
        "test_samples": int(len(test_files)),
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "per_class": per_class,
        },
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ---- keep ultralytics plots ----
    for extra in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = run_dir / extra
        if src.exists():
            shutil.copy2(src, results_dir / f"ultralytics_{extra}")

    print(
        f"[{variant}] test acc={acc:.4f} f1_macro={f1_macro:.4f} "
        f"latency={latency_ms:.1f}ms size={size_mb:.2f}MB params={params} "
        f"errors={n_errors} epochs_run={epochs_run}"
    )
    return payload


def train_variant(variant, weights):
    t0 = time.perf_counter()
    model = YOLO(weights)
    results_dir = RESULTS / f"week7_{variant}"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_results = model.train(
        data=str(DATA),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        device=DEVICE,
        project=str(results_dir),
        name="train",
        exist_ok=False,
    )
    train_time_s = time.perf_counter() - t0
    run_dir = Path(train_results.save_dir)
    evaluate_and_save.train_time_s = train_time_s
    payload = evaluate_and_save(run_dir / "weights" / "best.pt", variant, run_dir)
    return payload


if __name__ == "__main__":
    for variant, weights in [
        ("yolov8n_cls", "yolov8n-cls.pt"),
        ("yolov8s_cls", "yolov8s-cls.pt"),
    ]:
        print(f"===== training {variant} =====", flush=True)
        train_variant(variant, weights)
    print("ALL DONE", flush=True)
