
import os
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from scripts.input_pipeline import (
    IMAGE_SIZE,
    BATCH_SIZE,
    load_datasets,
    create_augmentation
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"

FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

REPORT_DIR = PROJECT_DIR / "outputs" / "reports"

BEST_MODEL_PATH = MODEL_DIR / "baseline_best.keras"

FINAL_MODEL_PATH = MODEL_DIR / "baseline_final.keras"

HISTORY_PATH = REPORT_DIR / "baseline_training_history.csv"

METRICS_PATH = REPORT_DIR / "baseline_metrics.csv"

CLASSIFICATION_PATH = REPORT_DIR / "baseline_classification_report.txt"

CONFUSION_PATH = REPORT_DIR / "baseline_confusion_matrix.csv"

EPOCHS = 20

LEARNING_RATE = 0.001

SEED = 42

MODEL_NAME = "VisionGuard_Baseline_CNN"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY AND CPU CONFIGURATION
# ============================================================

os.environ["TF_DETERMINISTIC_OPS"] = "1"

np.random.seed(SEED)

tf.random.set_seed(SEED)

print("\n========== VISIONGUARD BASELINE TRAINING ==========")

print("TensorFlow version:", tf.__version__)

print("Image size:", IMAGE_SIZE)

print("Batch size:", BATCH_SIZE)

print("Epochs:", EPOCHS)

print("Execution device: CPU / available TensorFlow devices")

print(tf.config.list_physical_devices())


# ============================================================
# BUILD BASELINE CNN
# ============================================================

def build_baseline_model():

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="input_image"
    )

    # Normalize pixel values from 0-255 to 0-1.
    x = tf.keras.layers.Rescaling(
        1.0 / 255.0,
        name="rescaling"
    )(inputs)

    # Augmentation happens only during training.
    augmentation = create_augmentation()

    x = augmentation(x)

    # --------------------------------------------------------
    # CONVOLUTION BLOCK 1
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_1"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_1"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling_1"
    )(x)

    # --------------------------------------------------------
    # CONVOLUTION BLOCK 2
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_2"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_2"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling_2"
    )(x)

    # --------------------------------------------------------
    # CONVOLUTION BLOCK 3
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv2d_3"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="batch_normalization_3"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        name="max_pooling_3"
    )(x)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dropout(
        0.4,
        name="dropout_1"
    )(x)

    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_1"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout_2"
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME
    )

    # --------------------------------------------------------
    # COMPILE MODEL
    # --------------------------------------------------------

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

    return model


# ============================================================
# SAVE MODEL ARCHITECTURE
# ============================================================

def save_model_summary(model):

    print("\n========== BASELINE CNN SUMMARY ==========\n")

    model.summary()

    layer_information = []

    for layer in model.layers:

        try:
            output_shape = str(layer.output.shape)
        except (AttributeError, ValueError):
            output_shape = "Multiple outputs"

        layer_information.append({
            "Layer": layer.name,
            "Type": layer.__class__.__name__,
            "Output Shape": output_shape,
            "Parameters": layer.count_params()
        })

    summary_df = pd.DataFrame(layer_information)

    summary_df.to_csv(
        REPORT_DIR / "baseline_model_layers.csv",
        index=False
    )

    print("\nModel architecture saved.")


# ============================================================
# TRAINING CALLBACKS
# ============================================================

def get_callbacks():

    callbacks = [

        # Save the model with the best validation loss.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),

        # Stop training if validation loss stops improving.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when progress plateaus.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        ),

        # Save epoch-by-epoch training metrics.
        tf.keras.callbacks.CSVLogger(
            filename=str(HISTORY_PATH)
        )
    ]

    return callbacks


# ============================================================
# PLOT TRAINING ACCURACY
# ============================================================

def plot_accuracy(history):

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("VisionGuard Baseline CNN - Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "baseline_accuracy.png",
        dpi=200
    )

    plt.close()


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_loss(history):

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("VisionGuard Baseline CNN - Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Binary Crossentropy Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "baseline_loss.png",
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    history_df = pd.DataFrame(history.history)

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
# EVALUATE ON TEST DATA
# ============================================================

def evaluate_model(model, test_dataset):

    print("\n========== FINAL TEST EVALUATION ==========")

    test_results = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True
    )

    # Collect actual labels.
    actual_labels = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset
    ])

    actual_labels = actual_labels.astype(int)

    # Generate predictions.
    predictions = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_probabilities = predictions.flatten()

    predicted_labels = (
        predicted_probabilities >= 0.5
    ).astype(int)

    # Calculate classification metrics.
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

    cm = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    print("\nTEST METRICS")

    print("Accuracy:", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall:", round(recall, 4))

    print("F1 Score:", round(f1, 4))

    print("Test Loss:", round(test_results["loss"], 4))

    print("Test AUC:", round(test_results["auc"], 4))

    # Save metrics.
    metrics = {
        "model": MODEL_NAME,
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "batch_size": BATCH_SIZE,
        "test_images": len(actual_labels),
        "test_loss": float(test_results["loss"]),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(test_results["auc"])
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_PATH,
        index=False
    )

    # Save confusion matrix.
    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Real", "Actual AI"],
        columns=["Predicted Real", "Predicted AI"]
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    # Save classification report.
    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Real", "AI Generated"],
        zero_division=0
    )

    with open(
        CLASSIFICATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    print("\nCLASSIFICATION REPORT")

    print(report)

    # Save confusion matrix visualization.
    plt.figure(figsize=(7, 6))

    plt.imshow(cm, interpolation="nearest")

    plt.title("Baseline CNN - Confusion Matrix")

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Real", "AI Generated"]
    )

    plt.yticks(
        [0, 1],
        ["Real", "AI Generated"]
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    for row in range(2):

        for column in range(2):

            plt.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "baseline_confusion_matrix.png",
        dpi=200
    )

    plt.close()

    print("\nEvaluation reports saved.")

    return metrics


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n========== LOADING DATA ==========")

    train_dataset, val_dataset, test_dataset = (
        load_datasets()
    )

    print("\n========== BUILDING BASELINE CNN ==========")

    model = build_baseline_model()

    save_model_summary(model)

    print("\n========== STARTING BASELINE CNN TRAINING ==========")

    history = model.fit(
        train_dataset,

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=get_callbacks(),

        verbose=1
    )

    print("\n========== TRAINING COMPLETED ==========")

    # Save final model.
    model.save(FINAL_MODEL_PATH)

    print("Final model saved:", FINAL_MODEL_PATH)

    # Save plots and training history.
    save_history(history)

    plot_accuracy(history)

    plot_loss(history)

    # Load best validation-loss checkpoint.
    if BEST_MODEL_PATH.exists():

        print("\nLoading best checkpoint...")

        del model

        gc.collect()

        model = tf.keras.models.load_model(
            BEST_MODEL_PATH
        )

    # Evaluate the best model on untouched test data.
    evaluate_model(
        model,
        test_dataset
    )

    print("\n========== BASELINE CNN PIPELINE COMPLETE ==========")

    print("\nGenerated models:")

    print(BEST_MODEL_PATH)

    print(FINAL_MODEL_PATH)

    print("\nGenerated figures:")

    print("baseline_accuracy.png")

    print("baseline_loss.png")

    print("baseline_confusion_matrix.png")

    print("\nGenerated reports:")

    print("baseline_training_history.csv")

    print("baseline_metrics.csv")

    print("baseline_classification_report.txt")

    print("baseline_confusion_matrix.csv")

    print("baseline_model_layers.csv")


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
