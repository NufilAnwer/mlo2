
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageDraw

from scripts.input_pipeline import (
    PROJECT_DIR,
    TEST_CSV,
    load_split_csv,
    create_dataset
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = PROJECT_DIR / "models" / "baseline_best.keras"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "error_analysis"

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

PREDICTION_CSV = REPORT_DIR / "baseline_error_analysis.csv"

SUMMARY_JSON = REPORT_DIR / "baseline_error_summary.json"

IMAGE_SIZE = (180, 180)

THUMBNAIL_SIZE = (180, 180)

COLUMNS = 4

IMAGES_PER_PAGE = 16


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_NAMES = {
    "correct_real": "Correctly Classified Real",
    "incorrect_real": "Real Misclassified as AI",
    "correct_ai": "Correctly Classified AI",
    "incorrect_ai": "AI Misclassified as Real"
}


# ============================================================
# PREPARE CATEGORY DIRECTORIES
# ============================================================

def prepare_directories():

    for category in CATEGORY_NAMES:

        category_dir = OUTPUT_DIR / category

        if category_dir.exists():
            shutil.rmtree(category_dir)

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    contact_dir = OUTPUT_DIR / "contact_sheets"

    contact_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return contact_dir


# ============================================================
# LOAD MODEL
# ============================================================

def load_baseline_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Baseline model not found: {MODEL_PATH}"
        )

    print("\n========== LOADING BASELINE MODEL ==========")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded:", MODEL_PATH)

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model):

    print("\n========== LOADING TEST DATA ==========")

    test_df = load_split_csv(TEST_CSV)

    test_dataset = create_dataset(
        test_df,
        training=False
    )

    print("Test images:", len(test_df))

    print("\nGenerating predictions...")

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    if len(probabilities) != len(test_df):

        raise ValueError(
            "Prediction count does not match test dataset."
        )

    actual_labels = test_df["label"].to_numpy(dtype=int)

    predicted_labels = (
        probabilities >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    correct = (
        actual_labels == predicted_labels
    )

    results = pd.DataFrame({

        "filepath": test_df["filepath"].values,

        "actual_label": actual_labels,

        "actual_class": [
            "AI Generated" if x == 1 else "Real"
            for x in actual_labels
        ],

        "predicted_label": predicted_labels,

        "predicted_class": [
            "AI Generated" if x == 1 else "Real"
            for x in predicted_labels
        ],

        "ai_probability": probabilities,

        "confidence": confidence,

        "correct": correct

    })

    return results


# ============================================================
# ORGANIZE MISCLASSIFIED IMAGES
# ============================================================

def organize_images(results):

    print("\n========== ORGANIZING TEST IMAGES ==========")

    category_counts = {}

    for index, row in results.iterrows():

        actual = int(row["actual_label"])

        predicted = int(row["predicted_label"])

        if actual == 0 and predicted == 0:

            category = "correct_real"

        elif actual == 0 and predicted == 1:

            category = "incorrect_real"

        elif actual == 1 and predicted == 1:

            category = "correct_ai"

        else:

            category = "incorrect_ai"

        source_path = Path(row["filepath"])

        destination_dir = OUTPUT_DIR / category

        filename = (
            f"{index:04d}_"
            f"{source_path.name}"
        )

        destination_path = (
            destination_dir / filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    for category, count in category_counts.items():

        print(
            f"{CATEGORY_NAMES[category]}: {count}"
        )

    return category_counts


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

def create_contact_sheets(results, contact_dir):

    print("\n========== GENERATING CONTACT SHEETS ==========")

    for category in CATEGORY_NAMES:

        category_results = results[
            (
                (results["actual_label"] == 0) &
                (results["predicted_label"] == 0)
            ) if category == "correct_real" else

            (
                (results["actual_label"] == 0) &
                (results["predicted_label"] == 1)
            ) if category == "incorrect_real" else

            (
                (results["actual_label"] == 1) &
                (results["predicted_label"] == 1)
            ) if category == "correct_ai" else

            (
                (results["actual_label"] == 1) &
                (results["predicted_label"] == 0)
            )
        ]

        image_records = category_results.to_dict(
            orient="records"
        )

        if not image_records:

            print(f"No images in category: {category}")

            continue

        total_pages = (
            len(image_records) + IMAGES_PER_PAGE - 1
        ) // IMAGES_PER_PAGE

        for page_number in range(total_pages):

            start = page_number * IMAGES_PER_PAGE

            end = start + IMAGES_PER_PAGE

            page_records = image_records[start:end]

            rows = (
                len(page_records) + COLUMNS - 1
            ) // COLUMNS

            cell_width = 220

            cell_height = 250

            sheet = Image.new(
                "RGB",
                (
                    COLUMNS * cell_width,
                    rows * cell_height
                ),
                "white"
            )

            draw = ImageDraw.Draw(sheet)

            for position, record in enumerate(page_records):

                image_path = Path(record["filepath"])

                try:

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    image.thumbnail(
                        THUMBNAIL_SIZE
                    )

                except Exception as error:

                    print(
                        f"Skipping unreadable image {image_path}: {error}"
                    )

                    continue

                column = position % COLUMNS

                row = position // COLUMNS

                x = column * cell_width

                y = row * cell_height

                image_x = (
                    x + (cell_width - image.width) // 2
                )

                sheet.paste(
                    image,
                    (image_x, y)
                )

                actual = record["actual_class"]

                predicted = record["predicted_class"]

                confidence = (
                    float(record["confidence"]) * 100
                )

                text = [
                    f"Actual: {actual}",
                    f"Predicted: {predicted}",
                    f"Confidence: {confidence:.1f}%"
                ]

                text_y = y + THUMBNAIL_SIZE[1] + 8

                for line in text:

                    draw.text(
                        (x + 8, text_y),
                        line,
                        fill="black"
                    )

                    text_y += 17

            output_path = contact_dir / (
                f"{category}_page_{page_number + 1}.png"
            )

            sheet.save(
                output_path
            )

            print("Saved:", output_path)


# ============================================================
# GENERATE ERROR ANALYSIS SUMMARY
# ============================================================

def generate_summary(results, category_counts):

    print("\n========== ERROR ANALYSIS SUMMARY ==========")

    total = len(results)

    correct = int(results["correct"].sum())

    incorrect = total - correct

    summary = {

        "model": "VisionGuard_Baseline_CNN",

        "test_images": total,

        "correct_predictions": correct,

        "incorrect_predictions": incorrect,

        "accuracy": float(correct / total),

        "classification_threshold": 0.5,

        "categories": category_counts,

        "analysis": {

            "real_correct": category_counts.get(
                "correct_real", 0
            ),

            "real_misclassified_as_ai": category_counts.get(
                "incorrect_real", 0
            ),

            "ai_correct": category_counts.get(
                "correct_ai", 0
            ),

            "ai_misclassified_as_real": category_counts.get(
                "incorrect_ai", 0
            )

        }

    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print("\nSummary saved:", SUMMARY_JSON)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n============================================")

    print("VISIONGUARD BASELINE ERROR ANALYSIS")

    print("============================================")

    contact_dir = prepare_directories()

    model = load_baseline_model()

    results = generate_predictions(model)

    results.to_csv(
        PREDICTION_CSV,
        index=False
    )

    print("\nPrediction CSV saved:", PREDICTION_CSV)

    category_counts = organize_images(
        results
    )

    create_contact_sheets(
        results,
        contact_dir
    )

    generate_summary(
        results,
        category_counts
    )

    print("\n============================================")

    print("BASELINE ERROR ANALYSIS COMPLETE")

    print("============================================")

    print("\nOutput directory:", OUTPUT_DIR)

    print("\nPrediction report:", PREDICTION_CSV)

    print("\nSummary:", SUMMARY_JSON)


if __name__ == "__main__":

    main()
