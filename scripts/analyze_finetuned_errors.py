
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat

from scripts.input_pipeline import (
    PROJECT_DIR,
    TEST_CSV,
    load_split_csv,
    create_dataset
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_DIR / "models" /
    "mobilenetv2_finetuned_best.keras"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "error_analysis"
)

REPORT_DIR = (
    PROJECT_DIR / "outputs" / "reports"
)

FIGURE_DIR = (
    PROJECT_DIR / "outputs" / "figures"
)

PREDICTION_CSV = (
    REPORT_DIR / "task_2_6_finetuned_predictions.csv"
)

ERROR_CSV = (
    REPORT_DIR / "task_2_6_misclassified_images.csv"
)

SUMMARY_JSON = (
    REPORT_DIR / "task_2_6_error_analysis_summary.json"
)

CONTACT_SHEET = (
    FIGURE_DIR / "task_2_6_misclassified_images.png"
)

THRESHOLD = 0.5

MAX_IMAGES = 10

IMAGES_PER_ERROR_TYPE = 5

THUMBNAIL_SIZE = (320, 220)

CELL_WIDTH = 360

CELL_HEIGHT = 310


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Fine-tuned model not found: {MODEL_PATH}"
        )

    print("\n========== LOADING FINE-TUNED MOBILENETV2 ==========")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded:", MODEL_PATH)

    return model


# ============================================================
# GENERATE TEST PREDICTIONS
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

    actual_labels = (
        test_df["label"].to_numpy(dtype=int)
    )

    predicted_labels = (
        probabilities >= THRESHOLD
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
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

        "correct": (
            actual_labels == predicted_labels
        )

    })

    return results


# ============================================================
# ANALYZE IMAGE CHARACTERISTICS
# ============================================================

def inspect_image(image_path):

    try:

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            width, height = image.size

            grayscale = image.convert("L")

            edges = grayscale.filter(
                ImageFilter.FIND_EDGES
            )

            edge_detail = (
                ImageStat.Stat(edges).stddev[0]
            )

            return width, height, edge_detail

    except Exception as error:

        print(
            f"Could not inspect {image_path}: {error}"
        )

        return 0, 0, 0.0


# ============================================================
# CREATE ERROR HYPOTHESES
# ============================================================

def create_hypothesis(
    width,
    height,
    edge_detail,
    actual_class,
    predicted_class
):

    if width == 0 or height == 0:

        return (
            "Image could not be inspected; "
            "manual visual review required."
        )

    if width < 300 or height < 300:

        return (
            "Low source resolution may have removed "
            "fine visual details."
        )

    if edge_detail < 20:

        return (
            "Low edge detail may indicate soft focus, "
            "smooth textures or limited visual features."
        )

    if actual_class == "Real":

        return (
            "Possible natural texture or photographic "
            "details may resemble AI-generated patterns."
        )

    return (
        "Possible realistic textures or image composition "
        "may resemble a real photograph."
    )


# ============================================================
# SELECT 10 MISCLASSIFIED IMAGES
# ============================================================

def select_misclassified_images(results):

    print("\n========== SELECTING MISCLASSIFIED IMAGES ==========")

    errors = results[
        results["correct"] == False
    ].copy()

    if errors.empty:

        raise ValueError(
            "No misclassified images were found."
        )

    errors["error_type"] = np.where(
        errors["actual_label"] == 0,
        "False Positive",
        "False Negative"
    )

    false_positives = errors[
        errors["error_type"] == "False Positive"
    ].sort_values(
        "confidence",
        ascending=False
    )

    false_negatives = errors[
        errors["error_type"] == "False Negative"
    ].sort_values(
        "confidence",
        ascending=False
    )

    selected = pd.concat([

        false_positives.head(
            IMAGES_PER_ERROR_TYPE
        ),

        false_negatives.head(
            IMAGES_PER_ERROR_TYPE
        )

    ])

    if len(selected) < MAX_IMAGES:

        remaining = errors.drop(
            selected.index
        ).sort_values(
            "confidence",
            ascending=False
        )

        needed = MAX_IMAGES - len(selected)

        selected = pd.concat([
            selected,
            remaining.head(needed)
        ])

    selected = selected.head(
        MAX_IMAGES
    ).reset_index(drop=True)

    print("Total test errors:", len(errors))

    print("Selected images:", len(selected))

    print(
        "False Positives:",
        int((selected["actual_label"] == 0).sum())
    )

    print(
        "False Negatives:",
        int((selected["actual_label"] == 1).sum())
    )

    return selected


# ============================================================
# GENERATE IMAGE ANALYSIS REPORT
# ============================================================

def generate_error_report(selected):

    records = []

    for index, row in selected.iterrows():

        image_path = Path(
            row["filepath"]
        )

        width, height, edge_detail = inspect_image(
            image_path
        )

        hypothesis = create_hypothesis(
            width,
            height,
            edge_detail,
            row["actual_class"],
            row["predicted_class"]
        )

        records.append({

            "image_number": index + 1,

            "filepath": str(image_path),

            "actual_class": row["actual_class"],

            "predicted_class": row["predicted_class"],

            "error_type": row["error_type"],

            "confidence_percent": round(
                float(row["confidence"]) * 100,
                2
            ),

            "ai_probability": round(
                float(row["ai_probability"]),
                4
            ),

            "image_width": width,

            "image_height": height,

            "edge_detail_proxy": round(
                float(edge_detail),
                2
            ),

            "hypothesis": hypothesis

        })

    report = pd.DataFrame(records)

    report.to_csv(
        ERROR_CSV,
        index=False
    )

    print("\nError analysis CSV saved:", ERROR_CSV)

    return report


# ============================================================
# GENERATE 10-IMAGE CONTACT SHEET
# ============================================================

def generate_contact_sheet(report):

    print("\n========== GENERATING IMAGE GRID ==========")

    rows = (
        len(report) + 1
    ) // 2

    sheet = Image.new(
        "RGB",
        (
            2 * CELL_WIDTH,
            rows * CELL_HEIGHT
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            14
        )

    except OSError:

        font = ImageFont.load_default()

    for index, record in report.iterrows():

        image_path = Path(
            record["filepath"]
        )

        try:

            with Image.open(image_path) as source:

                image = source.convert("RGB")

                image.thumbnail(
                    THUMBNAIL_SIZE
                )

        except Exception as error:

            print(
                f"Skipping unreadable image: {image_path}: {error}"
            )

            continue

        column = index % 2

        row = index // 2

        x = column * CELL_WIDTH

        y = row * CELL_HEIGHT

        image_x = (
            x + (CELL_WIDTH - image.width) // 2
        )

        sheet.paste(
            image,
            (image_x, y + 5)
        )

        confidence = record["confidence_percent"]

        lines = [

            f"Image {record['image_number']}",

            f"Actual: {record['actual_class']}",

            f"Predicted: {record['predicted_class']}",

            f"Error: {record['error_type']}",

            f"Confidence: {confidence:.1f}%"

        ]

        text_y = y + 230

        for line in lines:

            draw.text(
                (x + 10, text_y),
                line,
                fill="black",
                font=font
            )

            text_y += 17

    sheet.save(
        CONTACT_SHEET
    )

    print("Image grid saved:", CONTACT_SHEET)


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(results, report):

    total = len(results)

    errors = int(
        (~results["correct"]).sum()
    )

    summary = {

        "task": "Official Task 2.6 - Error Analysis",

        "model": "MobileNetV2 Fine-Tuned",

        "test_images": total,

        "misclassified_images": errors,

        "correct_predictions": total - errors,

        "accuracy": float(
            results["correct"].mean()
        ),

        "threshold": THRESHOLD,

        "selected_for_analysis": len(report),

        "false_positives_selected": int(
            (report["error_type"] == "False Positive").sum()
        ),

        "false_negatives_selected": int(
            (report["error_type"] == "False Negative").sum()
        ),

        "note": (
            "Hypotheses are preliminary suggestions based "
            "on image characteristics. Visually inspect "
            "each image before finalizing the report."
        )

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

    print("\n========== TASK 2.6 SUMMARY ==========")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n============================================")

    print("VISIONGUARD - TASK 2.6 ERROR ANALYSIS")

    print("FINE-TUNED MOBILENETV2")

    print("============================================")

    model = load_model()

    results = generate_predictions(model)

    results.to_csv(
        PREDICTION_CSV,
        index=False
    )

    print(
        "\nAll predictions saved:",
        PREDICTION_CSV
    )

    selected = select_misclassified_images(
        results
    )

    report = generate_error_report(
        selected
    )

    generate_contact_sheet(
        report
    )

    save_summary(
        results,
        report
    )

    print("\n============================================")

    print("TASK 2.6 ERROR ANALYSIS COMPLETE")

    print("============================================")


if __name__ == "__main__":

    main()
