import csv
import random
import hashlib
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SEED = 42
IMAGE_SIZE = 224

random.seed(SEED)

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/tensorflow_images")
SPLIT_DIR = Path("data/splits")

CLASSES = {
    "RealArt": 0,
    "AiArtData": 1
}

# Remove only previously generated TensorFlow-ready images.
if OUTPUT_DIR.exists():
    import shutil
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

splits = {
    "train": [],
    "val": [],
    "test": []
}

invalid_files = []

for class_name, label in CLASSES.items():

    source_dir = RAW_DIR / class_name
    destination_dir = OUTPUT_DIR / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    valid_images = []

    print(f"\nProcessing {class_name}...")

    for path in source_dir.rglob("*"):

        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:

                img.load()
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")

                img = ImageOps.pad(
                    img,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    method=Image.Resampling.LANCZOS,
                    color=(0, 0, 0)
                )

                file_hash = hashlib.sha1(
                    str(path.relative_to(source_dir)).encode("utf-8")
                ).hexdigest()[:12]

                output_path = destination_dir / f"{file_hash}.jpg"

                img.save(
                    output_path,
                    format="JPEG",
                    quality=95
                )

                valid_images.append(output_path)

        except Exception as error:
            invalid_files.append((str(path), str(error)))
            print("SKIPPED:", path)
            print("Reason:", error)

    random.shuffle(valid_images)

    total = len(valid_images)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits["train"].extend(
        [(str(p).replace("\\", "/"), label) for p in valid_images[:train_end]]
    )

    splits["val"].extend(
        [(str(p).replace("\\", "/"), label) for p in valid_images[train_end:val_end]]
    )

    splits["test"].extend(
        [(str(p).replace("\\", "/"), label) for p in valid_images[val_end:]]
    )

    print("Valid images:", total)
    print("Training:", train_end)
    print("Validation:", val_end - train_end)
    print("Testing:", total - val_end)

for split_name, records in splits.items():

    random.shuffle(records)

    output_file = SPLIT_DIR / f"{split_name}.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filepath", "label"])
        writer.writerows(records)

    print(f"Saved {output_file}: {len(records)} images")

print("\n========== FINAL SUMMARY ==========")

for name, records in splits.items():

    real_count = sum(1 for _, label in records if label == 0)
    ai_count = sum(1 for _, label in records if label == 1)

    print(
        f"{name}: Real={real_count}, "
        f"AI={ai_count}, Total={len(records)}"
    )

print("Total processed images:", sum(len(v) for v in splits.values()))
print("Skipped original files:", len(invalid_files))

if invalid_files:
    with open("outputs/audit/preprocessing_errors.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filepath", "error"])
        writer.writerows(invalid_files)

    print("Skipped-file report: outputs/audit/preprocessing_errors.csv")

print("\nPreprocessing and splitting completed.")
