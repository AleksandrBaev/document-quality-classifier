import json
import shutil
import sys
from pathlib import Path

def find_root():
    starts = [Path.cwd(), Path(__file__).resolve().parent]
    for start in starts:
        for p in [start, *start.parents]:
            if (p / "data" / "quality_dataset" / "metadata.jsonl").exists():
                return p
    return None


if len(sys.argv) > 1:
    SRC = Path(sys.argv[1])
    DST = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.parent / "quality_split"
else:
    root = find_root()
    if root is None:
        raise SystemExit("metadata.jsonl not found; pass the path to quality_dataset as the first argument")
    SRC = root / "data" / "quality_dataset"
    DST = root / "data" / "quality_split"

print("source", SRC.resolve())
print("target", DST.resolve())

records = []
with open(SRC / "metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if DST.exists():
    shutil.rmtree(DST)

counts = {}
missing = 0
for r in records:
    split = r["split"]
    label = r["quality_label"]
    src = SRC / r["image_path"]
    if not src.exists():
        missing += 1
        continue
    out = DST / split / label
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / src.name)
    counts[(split, label)] = counts.get((split, label), 0) + 1

for split in ("train", "validation", "test"):
    print(split, {l: counts.get((split, l), 0) for l in ("good", "bad")})
print("records", len(records), "missing", missing)