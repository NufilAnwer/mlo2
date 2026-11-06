
import os
import json
import random
import gc

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------
# CONFIGURATION
# --------------------------------------

AUDIT_FILE = "outputs/audit/dataset_audit.csv"
DATASET_DIR = "data/raw"
OUTPUT_DIR = "images"
REPORT_DIR = "outputs/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------
# LOAD DATASET AUDIT
# --------------------------------------

if not os.path.exists(AUDIT_FILE):
    raise FileNotFoundError(
        "Run Task 1.2 dataset audit before running EDA."
    )

df = pd.read_csv(AUDIT_FILE)

# Only analyze valid images
valid_df = df[df["status"] == "VALID"].copy()

if valid_df.empty:
    raise ValueError("No valid images found in the audit report.")

print("=" * 55)
print("VISIONGUARD EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# --------------------------------------
# 1. CLASS DISTRIBUTION
# --------------------------------------

class_counts = valid_df["class"].value_counts()

print("\nCLASS DISTRIBUTION")
print(class_counts)

plt.figure(figsize=(9, 6))

class_counts.plot(kind="bar")

plt.title("VisionGuard Dataset Class Distribution")
plt.xlabel("Image Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/class_distribution.png",
    dpi=200
)

plt.close()
gc.collect()

# --------------------------------------
# 2. CLASS DISTRIBUTION PIE CHART
# --------------------------------------

plt.figure(figsize=(8, 8))

class_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage Distribution of Image Classes")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/class_distribution_pie.png",
    dpi=200
)

plt.close()
gc.collect()

# --------------------------------------
# 3. IMAGE FORMAT DISTRIBUTION
# --------------------------------------

format_counts = valid_df["format"].value_counts()

print("\nIMAGE FORMATS")
print(format_counts)

plt.figure(figsize=(9, 6))

format_counts.plot(kind="bar")

plt.title("Image Format Distribution")
plt.xlabel("Image Format")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/image_formats.png",
    dpi=200
)

plt.close()
gc.collect()

# --------------------------------------
# 4. IMAGE WIDTH DISTRIBUTION
# --------------------------------------

valid_df["width"] = pd.to_numeric(
    valid_df["width"],
    errors="coerce"
)

plt.figure(figsize=(9, 6))

plt.hist(
    valid_df["width"].dropna(),
    bins=20
)

plt.title("Image Width Distribution")
plt.xlabel("Width in Pixels")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/image_width_distribution.png",
    dpi=200
)

plt.close()
gc.collect()

# --------------------------------------
# 5. IMAGE HEIGHT DISTRIBUTION
# --------------------------------------

valid_df["height"] = pd.to_numeric(
    valid_df["height"],
    errors="coerce"
)

plt.figure(figsize=(9, 6))

plt.hist(
    valid_df["height"].dropna(),
    bins=20
)

plt.title("Image Height Distribution")
plt.xlabel("Height in Pixels")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/image_height_distribution.png",
    dpi=200
)

plt.close()
gc.collect()

# --------------------------------------
# 6. IMAGE DIMENSION DISTRIBUTION
# --------------------------------------

dimension_counts = (
    valid_df.groupby(["width", "height"])
    .size()
    .reset_index(name="count")
)

dimension_counts = dimension_counts.sort_values(
    "count",
    ascending=False
)

print("\nMOST COMMON IMAGE DIMENSIONS")

print(
    dimension_counts.head(15).to_string(index=False)
)

dimension_counts.to_csv(
    f"{REPORT_DIR}/image_dimensions.csv",
    index=False
)

# --------------------------------------
# 7. IMAGE SIZE STATISTICS
# --------------------------------------

valid_df["size_bytes"] = pd.to_numeric(
    valid_df["size_bytes"],
    errors="coerce"
)

valid_df["size_mb"] = (
    valid_df["size_bytes"] / (1024 * 1024)
)

print("\nIMAGE SIZE STATISTICS (MB)")
print(valid_df["size_mb"].describe())

# --------------------------------------
# 8. CLASS-BASED DIMENSION STATISTICS
# --------------------------------------

dimension_stats = valid_df.groupby("class").agg(
    image_count=("file_path", "count"),
    average_width=("width", "mean"),
    average_height=("height", "mean"),
    average_size_mb=("size_mb", "mean")
).round(2)

print("\nCLASS-BASED STATISTICS")
print(dimension_stats)

dimension_stats.to_csv(
    f"{REPORT_DIR}/class_statistics.csv"
)

# --------------------------------------
# 9. VISUAL SAMPLE GRID
# --------------------------------------

print("\nGENERATING VISUAL SAMPLE GRID...")

random.seed(42)

classes = sorted(valid_df["class"].unique())

fig, axes = plt.subplots(
    len(classes),
    5,
    figsize=(12, 5 * len(classes)),
    squeeze=False
)

for row, class_name in enumerate(classes):

    class_data = valid_df[
        valid_df["class"] == class_name
    ]

    sample_count = min(5, len(class_data))

    samples = class_data.sample(
        n=sample_count,
        random_state=42
    )

    for col in range(5):

        ax = axes[row][col]
        ax.axis("off")

        if col >= sample_count:
            continue

        relative_path = samples.iloc[col]["file_path"]

        image_path = os.path.join(
            DATASET_DIR,
            relative_path
        )

        try:
            with Image.open(image_path) as image:

                # Resize before loading into Matplotlib
                image.thumbnail((320, 320))

                small_image = image.convert("RGB")

                ax.imshow(small_image)

            ax.set_title(
                class_name,
                fontsize=9
            )

        except Exception as error:

            ax.set_title(
                "Image unavailable",
                fontsize=8
            )

            print(
                "Could not display:",
                image_path,
                error
            )

plt.suptitle(
    "VisionGuard: Visual Samples by Class",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/visual_sample_grid.png",
    dpi=120
)

plt.close(fig)
gc.collect()

# --------------------------------------
# 10. GENERATE EDA SUMMARY
# --------------------------------------

summary = {
    "total_valid_images": int(len(valid_df)),

    "class_distribution": {
        str(key): int(value)
        for key, value in class_counts.items()
    },

    "format_distribution": {
        str(key): int(value)
        for key, value in format_counts.items()
    },

    "average_width": round(
        float(valid_df["width"].mean()),
        2
    ),

    "average_height": round(
        float(valid_df["height"].mean()),
        2
    ),

    "average_file_size_mb": round(
        float(valid_df["size_mb"].mean()),
        3
    )
}

with open(
    f"{REPORT_DIR}/eda_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print("\nEDA SUMMARY")

print(
    json.dumps(
        summary,
        indent=4
    )
)

# --------------------------------------
# 11. LIST GENERATED FIGURES
# --------------------------------------

print("\nGENERATED FIGURES")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.lower().endswith(".png"):
        print(filename)

print("\nGENERATED REPORTS")

for filename in sorted(os.listdir(REPORT_DIR)):
    print(filename)

print("\nEDA COMPLETED SUCCESSFULLY.")
