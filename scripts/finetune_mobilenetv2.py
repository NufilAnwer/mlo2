
import os

os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from scripts.input_pipeline import load_datasets


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

INPUT_MODEL = MODEL_DIR / "mobilenetv2_best.keras"

BEST_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_finetuned_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_finetuned_final.keras"
)

HISTORY_PATH = (
    REPORT_DIR / "mobilenetv2_finetuned_history.csv"
)

METRICS_PATH = (
    REPORT_DIR / "mobilenetv2_finetuned_metrics.csv"
)

CLASSIFICATION_PATH = (
    REPORT_DIR / "mobilenetv2_finetuned_classification_report.txt"
)

CONFUSION_PATH = (
    REPORT_DIR / "mobilenetv2_finetuned_confusion_matrix.csv"
)

LAYERS_PATH = (
    REPORT_DIR / "mobilenetv2_finetuned_layers.csv"
)

EPOCHS = 6
LEARNING_RATE = 0.00001
UNFREEZE_LAST_LAYERS = 30
SEED = 42

MODEL_NAME = "VisionGuard_MobileNetV2_FineTuned"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("\n========== MOBILENETV2 FINE-TUNING ==========")

print("TensorFlow:", tf.__version__)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Layers to unfreeze:", UNFREEZE_LAST_LAYERS)


# ============================================================
# LOAD AND CONFIGURE MODEL
# ============================================================

def load_and_configure_model():

    if not INPUT_MODEL.exists():

        raise FileNotFoundError(
            f"Starting model not found: {INPUT_MODEL}"
        )

    print("\nLoading best MobileNetV2 checkpoint...")

    model = tf.keras.models.load_model(
        INPUT_MODEL
    )

    # Locate the pretrained MobileNetV2 base model.
    base_model = None

    for layer in model.layers:

        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):

            base_model = layer
            break

    if base_model is None:

        raise ValueError(
            "Could not locate MobileNetV2 base model."
        )

    print("Found base model:", base_model.name)

    # Enable the base model for selective fine-tuning.
    base_model.trainable = True

    # Freeze all pretrained layers first.
    for layer in base_model.layers:

        layer.trainable = False

    # Unfreeze only the last selected layers.
    selected_layers = base_model.layers[
        -UNFREEZE_LAST_LAYERS:
    ]

    for layer in selected_layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    # Count trainable layers.
    trainable_layers = sum(
        1 for layer in base_model.layers
        if layer.trainable
    )

    print("Total pretrained layers:", len(base_model.layers))
    print("Trainable pretrained layers:", trainable_layers)

    # Recompile with a small learning rate.
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model, base_model


# ============================================================
# SAVE LAYER INFORMATION
# ============================================================

def save_layer_information(model, base_model):

    rows = []

    for layer in base_model.layers:

        rows.append({

            "layer_name": layer.name,

            "layer_type": layer.__class__.__name__,

            "trainable": layer.trainable,

            "parameters": layer.count_params()

        })

    pd.DataFrame(rows).to_csv(
        LAYERS_PATH,
        index=False
    )

    print("Layer information saved:", LAYERS_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    history_df = pd.DataFrame(
        history.history
    )

    history_df.insert(
        0,
        "epoch",
        range(1, len(history_df) + 1)
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    print("Training history saved:", HISTORY_PATH)


# ============================================================
# SAVE TRAINING FIGURES
# ============================================================

def save_figures(history):

    data = history.history

    epochs = range(
        1,
        len(data["loss"]) + 1
    )

    # Accuracy.
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        data["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        data["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("MobileNetV2 Fine-Tuning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        FIGURE_DIR / "mobilenetv2_finetuned_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Loss.
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        data["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        data["val_loss"],
        label="Validation Loss"
    )

    plt.title("MobileNetV2 Fine-Tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        FIGURE_DIR / "mobilenetv2_finetuned_loss.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Training figures saved.")


# ============================================================
# EVALUATE FINE-TUNED MODEL
# ============================================================

def evaluate_model(model, test_dataset):

    print("\n========== FINAL TEST EVALUATION ==========")

    evaluation = model.evaluate(
        test_dataset,
        verbose=1
    )

    test_loss = float(evaluation[0])

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    actual_labels = np.concatenate([

        labels.numpy().reshape(-1)

        for _, labels in test_dataset

    ]).astype(int)

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    precision = precision_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        actual_labels,
        probabilities
    )

    print("\nFINE-TUNED TEST METRICS")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC:  {auc:.4f}")

    report = classification_report(

        actual_labels,
        predictions,

        labels=[0, 1],

        target_names=[
            "Real",
            "AI Generated"
        ],

        zero_division=0
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    with open(
        CLASSIFICATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1]
    )

    pd.DataFrame(

        matrix,

        index=[
            "Actual Real",
            "Actual AI Generated"
        ],

        columns=[
            "Predicted Real",
            "Predicted AI Generated"
        ]

    ).to_csv(CONFUSION_PATH)

    # Confusion matrix visualization.
    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title("MobileNetV2 Fine-Tuned Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(
        [0, 1],
        ["Real", "AI Generated"]
    )

    plt.yticks(
        [0, 1],
        ["Real", "AI Generated"]
    )

    for row in range(2):

        for column in range(2):

            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.savefig(
        FIGURE_DIR / "mobilenetv2_finetuned_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    metrics = {

        "model": MODEL_NAME,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "test_loss": test_loss,

        "auc": auc,

        "threshold": 0.5

    }

    pd.DataFrame(
        [metrics]
    ).to_csv(
        METRICS_PATH,
        index=False
    )

    print("\nEvaluation reports saved.")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n========== LOADING DATASETS ==========")

    train_dataset, val_dataset, test_dataset = (
        load_datasets()
    )

    model, base_model = load_and_configure_model()

    print("\n========== FINE-TUNED MODEL SUMMARY ==========")

    model.summary()

    save_layer_information(
        model,
        base_model
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(

            filepath=str(BEST_MODEL_PATH),

            monitor="val_loss",

            save_best_only=True,

            mode="min",

            verbose=1

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=2,

            restore_best_weights=True,

            mode="min",

            verbose=1

        ),

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=1,

            min_lr=0.0000001,

            verbose=1

        )

    ]

    print("\n========== STARTING FINE-TUNING ==========")

    history = model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1

    )

    # Save the restored best weights.
    model.save(
        FINAL_MODEL_PATH
    )

    print("\nSaved fine-tuned model:", FINAL_MODEL_PATH)

    save_history(history)

    save_figures(history)

    # Load the best validation checkpoint.
    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    evaluate_model(
        best_model,
        test_dataset
    )

    print("\n========== FINE-TUNING COMPLETE ==========")

    print("\nGenerated models:")
    print(BEST_MODEL_PATH)
    print(FINAL_MODEL_PATH)

    print("\nGenerated reports:")
    print(HISTORY_PATH)
    print(METRICS_PATH)
    print(CLASSIFICATION_PATH)
    print(CONFUSION_PATH)
    print(LAYERS_PATH)


if __name__ == "__main__":

    main()
