import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

SEED = 2473057
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

def find_split_dir():
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for p in [start, *start.parents]:
            if (p / "data" / "quality_split").is_dir():
                return p / "data" / "quality_split"
    return None


if len(sys.argv) > 1:
    DATA_DIR = Path(sys.argv[1])
else:
    DATA_DIR = find_split_dir()
    if DATA_DIR is None:
        raise SystemExit("data/quality_split not found; run make_split_folders.py first")
print("data dir", DATA_DIR.resolve())
OUT_DIR = DATA_DIR.parent.parent / "results" / "mobilenetv2_frozen"
ERR_DIR = OUT_DIR / "errors"
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 50
PATIENCE = 5
LR = 0.001
WEEK = 6
EXPERIMENT = "week6_mobilenetv2_frozen"
THRESHOLD = 0.5

for split in ("train", "validation", "test"):
    for cls in ("good", "bad"):
        if not (DATA_DIR / split / cls).is_dir():
            raise SystemExit(f"missing {DATA_DIR / split / cls}; run make_split_folders.py")

print("image counts")
for split in ("train", "validation", "test"):
    row = {c: len(list((DATA_DIR / split / c).glob("*"))) for c in ("bad", "good")}
    print(split, row, "total", sum(row.values()))

OUT_DIR.mkdir(parents=True, exist_ok=True)
if ERR_DIR.exists():
    shutil.rmtree(ERR_DIR)
ERR_DIR.mkdir(parents=True)

gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = gen.flow_from_directory(
    DATA_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="binary", shuffle=True, seed=SEED,
)
val_gen = gen.flow_from_directory(
    DATA_DIR / "validation", target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="binary", shuffle=False,
)
test_gen = gen.flow_from_directory(
    DATA_DIR / "test", target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="binary", shuffle=False,
)
print("class indices", train_gen.class_indices)
class_names = [c for c, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

best_path = OUT_DIR / "best_model.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True),
]

start = time.perf_counter()
history = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=2,
)
train_time_s = time.perf_counter() - start
epochs_run = len(history.history["loss"])

test_gen.reset()
probs = model.predict(test_gen, verbose=0).ravel()
y_pred = (probs >= THRESHOLD).astype(int)
y_true = test_gen.classes

acc = float(accuracy_score(y_true, y_pred))
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
p_m, r_m, f_m, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1], average="macro", zero_division=0
)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

per_class = {
    name: {
        "precision": float(prec[i]),
        "recall": float(rec[i]),
        "f1": float(f1[i]),
        "support": int(sup[i]),
    }
    for i, name in enumerate(class_names)
}

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False)
ax.set_title("MobileNetV2 frozen: test confusion matrix")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

h = history.history
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(h["loss"], label="train")
axes[0].plot(h["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(h["accuracy"], label="train")
axes[1].plot(h["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "curves.png", dpi=150)
plt.close(fig)

wrong = np.where(y_pred != y_true)[0]
for i in wrong:
    src = Path(test_gen.directory) / test_gen.filenames[i]
    true_name = class_names[int(y_true[i])]
    pred_name = class_names[int(y_pred[i])]
    shutil.copy(src, ERR_DIR / f"true_{true_name}__pred_{pred_name}__{src.name}")
print("misclassified", len(wrong))

def load_one(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    return preprocess_input(arr)

paths = [Path(test_gen.directory) / f for f in test_gen.filenames]
samples = [load_one(paths[i % len(paths)]) for i in range(120)]
for x in samples[:20]:
    model.predict(x, verbose=0)
times = []
for x in samples[20:120]:
    t0 = time.perf_counter()
    model.predict(x, verbose=0)
    times.append(time.perf_counter() - t0)
latency_ms = float(np.mean(times) * 1000)

model_size_mb = best_path.stat().st_size / (1024 * 1024)
device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"

results = {
    "experiment": EXPERIMENT,
    "week": WEEK,
    "model": "MobileNetV2",
    "regime": "frozen_base",
    "input_size": 224,
    "seed": SEED,
    "device": device,
    "epochs_planned": EPOCHS,
    "epochs_run": epochs_run,
    "train_time_s": round(train_time_s, 2),
    "latency_ms_per_image": round(latency_ms, 3),
    "model_size_mb": round(model_size_mb, 3),
    "params": int(model.count_params()),
    "test_samples": int(len(y_true)),
    "metrics": {
        "accuracy": acc,
        "f1_macro": float(f_m),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "per_class": per_class,
    },
    "confusion_matrix": cm.tolist(),
    "history": {k: [float(v) for v in vals] for k, vals in h.items()},
}
with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2, ensure_ascii=False))