"""Task 4 training and final evaluation.

From the project root: python -u scripts/train_efficientnet_b0.py
"""

import json
import math
import random
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


SEED = 2473057
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32
IMAGE_SIZE = 224
WARM_UP = 20
TIMED = 100
DEVICE = torch.device("cpu")
CLASSES = ("bad", "good")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "quality_split"
OUTPUT = ROOT / "results" / "week8_efficientnet_b0"
WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1


def make_transform():
    preset = WEIGHTS.transforms()
    # Resize the full document to 224 x 224 so the edges remain visible.
    return Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=preset.interpolation, antialias=True),
        ToTensor(),
        Normalize(mean=preset.mean, std=preset.std),
    ])


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.eval()
    if training:
        # Keep backbone BatchNorm statistics and stochastic depth frozen.
        model.classifier.train()

    loss_sum = 0.0
    correct = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def save_curves(history, path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(epochs, history[metric], label="Train")
        ax.plot(epochs, history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", title=metric.capitalize())
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate_test(model, dataset, output):
    model.eval()
    labels, predictions = [], []
    for images, batch_labels in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
        labels.extend(batch_labels.tolist())
        predictions.extend(model(images).argmax(dim=1).tolist())

    class_ids = list(range(len(CLASSES)))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0
    )
    metrics = {
        "accuracy": float(matrix.trace() / matrix.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }

    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(colorbar=False, cmap="Blues")
    display.ax_.set_title("EfficientNet-B0 test confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)

    errors_dir = output / "errors"
    errors_dir.mkdir()
    for (filename, label), predicted in zip(dataset.samples, predictions):
        if label != predicted:
            source = Path(filename)
            target = errors_dir / f"true_{CLASSES[label]}__pred_{CLASSES[predicted]}__{source.name}"
            shutil.copy2(source, target)
    return metrics, matrix.tolist()


@torch.inference_mode()
def measure_latency(model, dataset):
    # Match the Week 7 repeat check: sorted JPEG paths, same seed and timing boundary.
    jpeg_paths = sorted(Path(path) for path, _ in dataset.samples if Path(path).suffix == ".jpg")
    paths = random.Random(SEED).sample(jpeg_paths, WARM_UP + TIMED)
    model.eval()

    def predict(path):
        image = dataset.transform(dataset.loader(str(path))).unsqueeze(0)
        return model(image).argmax(dim=1).item()

    for path in paths[:WARM_UP]:
        predict(path)
    start = perf_counter()
    for path in paths[WARM_UP:]:
        predict(path)
    return (perf_counter() - start) * 1000 / TIMED


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists. Archive that run before retrying.")
    OUTPUT.mkdir(parents=True)
    checkpoint = OUTPUT / "best.pt"
    history = {key: [] for key in ("loss", "accuracy", "val_loss", "val_accuracy")}
    results = {
        "experiment": OUTPUT.name,
        "week": 8,
        "model": "EfficientNet-B0",
        "regime": "frozen_base",
        "input_size": IMAGE_SIZE,
        "seed": SEED,
        "device": str(DEVICE),
        "epochs_planned": EPOCHS,
        "epochs_run": 0,
        "train_time_s": None,
        "latency_ms_per_image": None,
        "model_size_mb": None,
        "params": None,
        "test_samples": None,
        "metrics": None,
        "confusion_matrix": None,
        "history": history,
    }
    try:
        transform = make_transform()
        datasets = {}
        for split in ("train", "validation", "test"):
            dataset = ImageFolder(DATA / split, transform=transform)
            if tuple(dataset.classes) != CLASSES:
                raise ValueError(f"{split}: expected classes {CLASSES}, found {dataset.classes}")
            print(f"{split}: {len(dataset)} images", flush=True)
            datasets[split] = dataset
        train_loader = DataLoader(
            datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        val_loader = DataLoader(datasets["validation"], batch_size=BATCH_SIZE)

        model = efficientnet_b0(weights=WEIGHTS).to(DEVICE)
        model.requires_grad_(False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES)).to(DEVICE)
        results["params"] = sum(param.numel() for param in model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
        best_loss = float("inf")
        best_epoch = 0
        start = perf_counter()
        try:
            for epoch in range(1, EPOCHS + 1):
                loss, accuracy = run_epoch(model, train_loader, loss_fn, optimizer)
                val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn)
                if not (math.isfinite(loss) and math.isfinite(val_loss)):
                    raise ValueError(f"Non-finite loss at epoch {epoch}")
                for key, value in zip(history, (loss, accuracy, val_loss, val_accuracy)):
                    history[key].append(value)
                results["epochs_run"] = epoch
                print(f"Epoch {epoch}/{EPOCHS} - loss {loss:.4f}, acc {accuracy:.4f}, "
                      f"val_loss {val_loss:.4f}, val_acc {val_accuracy:.4f}", flush=True)
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch
                    pending = checkpoint.with_suffix(".tmp")
                    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, pending)
                    pending.replace(checkpoint)
                elif epoch - best_epoch >= PATIENCE:
                    break
        finally:
            results["train_time_s"] = perf_counter() - start

        saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        print(f"Best validation-loss checkpoint: epoch {saved['epoch']}", flush=True)
        results["model_size_mb"] = checkpoint.stat().st_size / (1024 ** 2)
        save_curves(history, OUTPUT / "curves.png")
        test = datasets["test"]
        results["test_samples"] = len(test)
        results["metrics"], results["confusion_matrix"] = evaluate_test(model, test, OUTPUT)
        results["latency_ms_per_image"] = measure_latency(model, test)
    except (Exception, KeyboardInterrupt) as error:
        results["status"] = "failed"
        results["note"] = " ".join(f"{type(error).__name__}: {error}".split())
        raise
    finally:
        (OUTPUT / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(f"Saved Task 4 results to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
