
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    log_loss
)

from scripts.input_pipeline import (
    PROJECT_DIR,
    VAL_CSV,
    TEST_CSV,
    load_split_csv,
    create_dataset
)


MODEL_PATH = (
    PROJECT_DIR / "models" /
    "mobilenetv2_finetuned_best.keras"
)

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def select_validation_threshold(model):

    print("\n========== VALIDATION THRESHOLD SELECTION ==========")

    val_df = load_split_csv(VAL_CSV)

    val_dataset = create_dataset(
        val_df,
        training=False
    )

    actual_labels = val_df["label"].to_numpy(dtype=int)

    probabilities = model.predict(
        val_dataset,
        verbose=1
    ).flatten()

    precision, recall, thresholds = precision_recall_curve(
        actual_labels,
        probabilities
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to calculate validation thresholds.")

    denominator = precision[:-1] + recall[:-1]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    best_index = int(np.argmax(f1_values))

    best_threshold = float(thresholds[best_index])

    results = {
        "threshold": best_threshold,
        "validation_f1": float(f1_values[best_index]),
        "validation_precision": float(precision[best_index]),
        "validation_recall": float(recall[best_index]),
        "validation_images": int(len(actual_labels)),
        "selection_method": "Maximum validation F1-score"
    }

    output_path = REPORT_DIR / "task_2_5_threshold_selection.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("Selected threshold:", best_threshold)
    print("Validation F1:", results["validation_f1"])
    print("Threshold report:", output_path)

    return best_threshold


def evaluate_test_set(model, threshold):

    print("\n========== FINAL TEST EVALUATION ==========")

    test_df = load_split_csv(TEST_CSV)

    test_dataset = create_dataset(
        test_df,
        training=False
    )

    actual_labels = test_df["label"].to_numpy(dtype=int)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    if len(probabilities) != len(actual_labels):
        raise ValueError("Prediction count does not match test images.")

    predicted_labels = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    auc = roc_auc_score(
        actual_labels,
        probabilities
    )

    test_loss = log_loss(
        actual_labels,
        probabilities,
        labels=[0, 1]
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    metrics = {
        "model": "MobileNetV2 Fine-Tuned",
        "threshold": float(threshold),
        "test_images": int(len(actual_labels)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "test_loss": float(test_loss)
    }

    metrics_path = REPORT_DIR / "task_2_5_final_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    matrix_path = REPORT_DIR / "task_2_5_confusion_matrix.csv"

    pd.DataFrame(
        matrix,
        index=["Actual Real", "Actual AI"],
        columns=["Predicted Real", "Predicted AI"]
    ).to_csv(matrix_path)

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Real", "AI Generated"],
        zero_division=0
    )

    report_path = REPORT_DIR / "task_2_5_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("TASK 2.5 - FINAL TEST EVALUATION\n\n")
        file.write(f"Selected threshold: {threshold:.6f}\n\n")
        file.write(report)

    fpr, tpr, _ = roc_curve(
        actual_labels,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"MobileNetV2 (AUC = {auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("VisionGuard - MobileNetV2 ROC Curve")
    plt.legend()
    plt.grid(True)

    roc_path = FIGURE_DIR / "task_2_5_mobilenetv2_roc_curve.png"

    plt.savefig(
        roc_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\n========== FINAL METRICS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    print("Metrics saved:", metrics_path)
    print("Confusion matrix saved:", matrix_path)
    print("Classification report saved:", report_path)
    print("ROC curve saved:", roc_path)


def main():

    print("\n========== TASK 2.5 EVALUATION ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading existing fine-tuned model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

    threshold = select_validation_threshold(model)

    evaluate_test_set(
        model,
        threshold
    )

    print("\n========== TASK 2.5 COMPLETED ==========")


if __name__ == "__main__":
    main()
