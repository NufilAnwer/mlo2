import os
import csv
import json
import hashlib
from collections import Counter, defaultdict
from PIL import Image

DATASET_DIR = "data/raw"
OUTPUT_DIR = "outputs/audit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

records = []
corrupt_records = []
hash_groups = defaultdict(list)
class_counts = Counter()
format_counts = Counter()

total_size = 0

print("Starting VisionGuard Dataset Audit...")
print("=" * 55)

for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        extension = os.path.splitext(filename)[1].lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        file_path = os.path.join(root, filename)
        relative_path = os.path.relpath(file_path, DATASET_DIR)

        parts = relative_path.split(os.sep)
        image_class = parts[0] if len(parts) > 1 else "Unknown"

        file_size = os.path.getsize(file_path)
        total_size += file_size
        class_counts[image_class] += 1

        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    sha256_hash.update(chunk)

            image_hash = sha256_hash.hexdigest()

            with Image.open(file_path) as image:
                image.verify()

            with Image.open(file_path) as image:
                width, height = image.size
                image_format = image.format

            format_counts[image_format] += 1

            record = {
                "file_path": relative_path,
                "class": image_class,
                "extension": extension,
                "size_bytes": file_size,
                "sha256": image_hash,
                "status": "VALID",
                "width": width,
                "height": height,
                "format": image_format,
                "error": ""
            }

            hash_groups[image_hash].append(relative_path)

        except Exception as error:

            record = {
                "file_path": relative_path,
                "class": image_class,
                "extension": extension,
                "size_bytes": file_size,
                "sha256": "",
                "status": "CORRUPT",
                "width": "",
                "height": "",
                "format": "",
                "error": str(error)
            }

            corrupt_records.append(record)

        records.append(record)

        print(record["status"], relative_path)

# Identify exact duplicate groups
duplicate_records = []

for image_hash, paths in hash_groups.items():

    if len(paths) > 1:

        for path in paths:
            duplicate_records.append({
                "sha256": image_hash,
                "file_path": path,
                "duplicate_group_size": len(paths)
            })

# Save complete audit CSV
audit_file = os.path.join(OUTPUT_DIR, "dataset_audit.csv")

with open(audit_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys() if records else [])
    writer.writeheader()
    writer.writerows(records)

# Save corrupt image report
corrupt_file = os.path.join(OUTPUT_DIR, "corrupt_images.csv")

with open(corrupt_file, "w", newline="", encoding="utf-8") as file:
    fields = [
        "file_path", "class", "extension",
        "size_bytes", "sha256", "status",
        "width", "height", "format", "error"
    ]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(corrupt_records)

# Save duplicate report
duplicate_file = os.path.join(OUTPUT_DIR, "duplicate_report.csv")

with open(duplicate_file, "w", newline="", encoding="utf-8") as file:
    fields = ["sha256", "file_path", "duplicate_group_size"]

    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(duplicate_records)

valid_count = sum(
    1 for record in records if record["status"] == "VALID"
)

duplicate_groups = sum(
    1 for paths in hash_groups.values() if len(paths) > 1
)

duplicate_files = len(duplicate_records)

summary = {
    "project": "23L-0532_A2_Ops",
    "dataset_directory": DATASET_DIR,
    "total_images": len(records),
    "valid_images": valid_count,
    "corrupt_images": len(corrupt_records),
    "unique_valid_hashes": len(hash_groups),
    "duplicate_groups": duplicate_groups,
    "duplicate_files_in_groups": duplicate_files,
    "class_counts": dict(class_counts),
    "format_counts": dict(format_counts),
    "total_size_bytes": total_size,
    "total_size_mb": round(total_size / (1024 * 1024), 2)
}

summary_file = os.path.join(OUTPUT_DIR, "audit_summary.json")

with open(summary_file, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 55)
print("VISIONGUARD DATASET AUDIT COMPLETED")
print("=" * 55)

print(json.dumps(summary, indent=4))

print("\nGenerated files:")
print(audit_file)
print(summary_file)
print(corrupt_file)
print(duplicate_file)
