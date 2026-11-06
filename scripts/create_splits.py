import csv
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/splits")

CLASSES = {
    "RealArt": 0,
    "AiArtData": 1
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

splits = {
    "train": [],
    "val": [],
    "test": []
}

for class_name, label in CLASSES.items():

    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")

    images = [
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]

    if not images:
        raise ValueError(f"No images found in {class_dir}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits["train"].extend(
        [(str(p).replace("\\", "/"), label) for p in images[:train_end]]
    )

    splits["val"].extend(
        [(str(p).replace("\\", "/"), label) for p in images[train_end:val_end]]
    )

    splits["test"].extend(
        [(str(p).replace("\\", "/"), label) for p in images[val_end:]]
    )

    print(f"\n{class_name}: {total} images")
    print(f"Training: {train_end}")
    print(f"Validation: {val_end - train_end}")
    print(f"Testing: {total - val_end}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for split_name, records in splits.items():

    random.shuffle(records)

    output_file = OUTPUT_DIR / f"{split_name}.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)
        writer.writerow(["filepath", "label"])
        writer.writerows(records)

    print(f"Saved: {output_file} ({len(records)} images)")

print("\n===== SPLITTING COMPLETED =====")
print("Total images:", sum(len(v) for v in splits.values()))

for name, records in splits.items():
    real_count = sum(1 for _, label in records if label == 0)
    ai_count = sum(1 for _, label in records if label == 1)

    print(f"{name}: Real={real_count}, AI={ai_count}, Total={len(records)}")
