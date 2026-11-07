"""Build Task 4 reports from existing results. No training is performed."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


MODELS = (
    ("MobileNetV2", "mobilenetv2_frozen", "MobileNetV2 frozen"),
    ("YOLOv8n-cls", "week7_yolov8n_cls", "YOLOv8n-cls"),
    ("YOLOv8s-cls", "week7_yolov8s_cls", "YOLOv8s-cls"),
    ("EfficientNet-B0", "week8_efficientnet_b0", None),
)
CLASSES = ("bad", "good")
COLORS = ("#315e85", "#007f71", "#ad572d", "#80518d")
ERROR_NAME = re.compile(r"true_(bad|good)__pred_(bad|good)__(.+)")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, content):
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def load_runs(root):
    runs = []
    for model, folder, week7_label in MODELS:
        path = root / "results" / folder / "results.json"
        result = load_json(path)
        matrix = result["confusion_matrix"]
        if (len(matrix) != 2 or any(len(row) != 2 for row in matrix)
                or sum(map(sum, matrix)) != result["test_samples"]):
            raise ValueError(f"Invalid confusion matrix in {path}")
        runs.append((model, folder, week7_label, result))
    if len({run[3]["test_samples"] for run in runs}) != 1:
        raise ValueError("The models have different test sample counts")
    return runs


def collect_cpu_latencies(root, runs, reports):
    source = root / "reports" / "week7_comparison.md"
    match = re.search(r"^## CPU latency check\s*\n(.*?)(?=^## |\Z)",
                      source.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"No CPU repeat section in {source}")
    rows = {}
    for line in match.group(1).splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"\d+\.\d+", cells[1]):
            rows[cells[0]] = float(cells[1])
    values = {}
    for name, folder, label, result in runs:
        if label:
            if label not in rows:
                raise ValueError(f"Missing repeat latency for {label} in {source}")
            path = "reports/week7_comparison.md"
            latency = rows[label]
        else:
            path = f"results/{folder}/results.json"
            latency = result["latency_ms_per_image"]
        if latency <= 0:
            raise ValueError(f"Invalid latency for {name}")
        values[name] = {"ms_per_image": latency, "source": path}
    dest = reports / "task4_cpu_latency.json"
    write(dest, json.dumps({
        "device": "Intel Core Ultra 5 225H CPU", "seed": 2473057,
        "warmup_predictions": 20, "timed_predictions": 100,
        "boundary": "JPEG file path to predicted class, including image preparation",
        "measurements": values,
    }, indent=2))
    return load_json(dest)["measurements"]


def make_comparison(runs, cpu, reports):
    lines = [
        "# Task 4 Model Comparison", "",
        f"All models were evaluated on the same {runs[0][3]['test_samples']} test images.",
        "The first table uses the original results.json files for every value.",
        "Model size is in MiB because the original scripts divide file size by 1,048,576.", "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Original latency (ms/image) | Size (MiB) | Parameters | Training time (s) | Epochs run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _, _, r in runs:
        m = r["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
            f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} | "
            f"{r['latency_ms_per_image']:.2f} | {r['model_size_mb']:.2f} | "
            f"{r['params']:,} | {r['train_time_s']:,.2f} | {r['epochs_run']} |"
        )
    lines.extend([
        "", "## CPU latency under the same protocol", "",
        "The original latency values use different timing methods.",
        "The chart uses the same protocol for all four models. On the same CPU, each model used",
        "the same 120 test JPEGs in the same order. The first 20 images were for warm-up.",
        "The next 100 were timed one at a time, from reading the file to getting a predicted class.",
        "Image reading and preparation were included. Model loading was outside the timing.",
        "The first three timings are from the Week 7 repeat check. EfficientNet-B0 used this",
        "protocol in its original run. These measurements were collected in separate runs.", "",
        "| Model | Comparable latency (ms/image) | Source |",
        "|---|---:|---|",
    ])
    for name, values in cpu.items():
        src = values["source"]
        link = "week7_comparison.md" if src.startswith("reports/") else f"../{src}"
        lines.append(f"| {name} | {values['ms_per_image']:.2f} | [{Path(src).name}]({link}) |")
    lines.extend([
        "", "![Accuracy versus comparable CPU latency](figures/accuracy_vs_latency.png)", "",
        "![Four test confusion matrices](figures/confusion_matrices.png)", "",
        "See the [error analysis and failure gallery](error_analysis.md).",
        "The latency sources are also recorded in [task4_cpu_latency.json](task4_cpu_latency.json).",
    ])
    write(reports / "comparison.md", "\n".join(lines))


def plot_comparison(runs, cpu, figures):
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for (name, _, _, result), color in zip(runs, COLORS):
        x, y = cpu[name]["ms_per_image"], 100 * result["metrics"]["accuracy"]
        ax.scatter(x, y, color=color, s=62, zorder=3, label=name)
        ax.annotate(f"{x:.2f} ms, {y:.2f}%", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set(title="Test accuracy and comparable CPU latency",
           xlabel="Latency (ms per image)", ylabel="Test accuracy (%)")
    ax.margins(x=0.15, y=0.24)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(figures / "accuracy_vs_latency.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2), layout="constrained")
    fig.suptitle("Test confusion matrices")
    limit = max(max(row) for _, _, _, result in runs for row in result["confusion_matrix"])
    for ax, (name, _, _, result) in zip(axes.flat, runs):
        matrix = result["confusion_matrix"]
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=limit)
        ax.set(title=name, xlabel="Predicted label", ylabel="True label",
               xticks=range(2), yticks=range(2), xticklabels=CLASSES, yticklabels=CLASSES)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > limit / 2 else "#11263d")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Images")
    fig.savefig(figures / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def collect_failures(root, runs, gallery):
    metadata = {}
    with (root / "data" / "quality_dataset" / "metadata.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["split"] == "test":
                    metadata[Path(row["image_path"]).name] = row
    if Counter(row["quality_label"] for row in metadata.values()) != Counter(
        dict(zip(CLASSES, map(sum, runs[0][3]["confusion_matrix"])))
    ):
        raise ValueError("Test metadata does not match test class counts")

    all_errors = {}
    for name, folder, _, result in runs:
        errors = []
        errors_dir = root / "results" / folder / "errors"
        dest = gallery / folder
        dest.mkdir(exist_ok=True)
        seen = set()
        for path in sorted(errors_dir.iterdir()):
            if not path.is_file():
                continue
            match = ERROR_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Invalid failure image name: {path}")
            true, pred, filename = match.groups()
            row = metadata[filename]
            if filename in seen or true != row["quality_label"] or true == pred:
                raise ValueError(f"Invalid failure labels: {path}")
            seen.add(filename)
            image = dest / path.name
            original = root / "data" / "quality_split" / "test" / true / filename
            content = path.read_bytes()
            if content != original.read_bytes():
                raise ValueError(f"Failure image does not match the original test image: {path}")
            image.write_bytes(content)
            errors.append((image, true, pred, row["defect_type"], row["class_name"]))
        matrix = result["confusion_matrix"]
        directions = Counter((true, pred) for _, true, pred, _, _ in errors)
        if (directions["bad", "good"], directions["good", "bad"]) != (matrix[0][1], matrix[1][0]):
            raise ValueError(f"Failure image counts do not match the matrix for {name}")
        all_errors[name] = errors
    return all_errors, Counter(row["defect_type"] for row in metadata.values()
                               if row["quality_label"] == "bad")


def make_gallery(all_errors, counts, reports, gallery):
    lines = [
        "# Task 4 Error Analysis and Failure Gallery", "",
        "The gallery shows all saved errors from the synthetic test set.",
        "Each example gives the true and predicted labels.",
        "No extra examples were added. Three models made fewer than five errors on this test set,",
        "so their galleries contain all available errors.", "",
        "| Model | Test errors | Dominant defect among errors | Gallery limit |",
        "|---|---:|---|---|",
    ]
    total = Counter()
    for model, errors in all_errors.items():
        defects = Counter(row[3] for row in errors)
        total.update(defects)
        dominant = ", ".join(f"{defect} ({count})" for defect, count in defects.most_common())
        limit = "Five examples available" if len(errors) >= 5 else "Fewer than five test errors"
        lines.append(f"| {model} | {len(errors)} | {dominant or 'None'} | {limit} |")
    lines.extend([
        "", "## Defects on the test set", "",
        "The same photo can be missed by more than one model. Each model's prediction counts separately.",
        "The pooled miss rate divides the total misses by the number of test images with the defect",
        "times the number of models.", "",
        "| Defect | Bad test images | Model misses | Pooled miss rate |",
        "|---|---:|---:|---:|",
    ])
    for defect, count in sorted(counts.items()):
        lines.append(f"| {defect} | {count} | {total[defect]} | {total[defect] / (count * len(all_errors)):.2%} |")
    hardest = max(counts, key=lambda defect: total[defect] / counts[defect])
    lines.extend([
        "", f"**{hardest.capitalize()} was the hardest defect on this test set by pooled miss rate.**",
        "To reduce glare errors, add training photos with bright reflections in different places.",
        "Vary the size, brightness, and shape of the reflections.",
        "Include real photos taken under lamps and in sunlight, especially for passports and INN documents.",
        "Check the labels with human reviewers and use a separate set of real photos to measure improvement.",
        "", "## Failure gallery", "",
        "![All saved test mistakes, grouped by model](failure_gallery/gallery.png)", "",
        "| Model | Test image | Document | True label | Predicted label | Defect |",
        "|---|---|---|---|---|---|",
    ])
    columns = max(5, max(map(len, all_errors.values())))
    fig, axes = plt.subplots(len(all_errors), columns, figsize=(13, 10), layout="constrained")
    fig.suptitle("Model mistakes on the synthetic test set")
    for (model, errors), row_axes in zip(all_errors.items(), axes):
        for ax in row_axes:
            ax.set_axis_off()
        for i, (image, true, pred, defect, doc) in enumerate(errors):
            path = image.relative_to(reports).as_posix()
            filename = image.name.split("__", 2)[2]
            lines.append(f"| {model} | [{filename}]({path}) | {doc} | {true} | {pred} | {defect} |")
            ax = row_axes[i]
            with Image.open(image) as photo:
                ax.imshow(photo.convert("RGB"))
            ax.set_title(f"{model}\n{doc} / {defect}\nTrue: {true} / Pred: {pred}", fontsize=8)
    fig.savefig(gallery / "gallery.png", dpi=160)
    plt.close(fig)
    write(reports / "error_analysis.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-root", type=Path, help="Write reports under a different root")
    args = parser.parse_args()
    root = args.project_root.resolve()
    reports = (args.output_root or root).resolve() / "reports"
    figures = reports / "figures"
    gallery = reports / "failure_gallery"
    figures.mkdir(parents=True, exist_ok=True)
    gallery.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    cpu = collect_cpu_latencies(root, runs, reports)
    errors, counts = collect_failures(root, runs, gallery)
    make_comparison(runs, cpu, reports)
    plot_comparison(runs, cpu, figures)
    make_gallery(errors, counts, reports, gallery)
    print(f"Built Task 4 reports in {reports}")


if __name__ == "__main__":
    main()
