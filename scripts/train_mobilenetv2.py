
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

BEST_MODEL_PATH = MODEL_DIR / "mobilenetv2_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "mobilenetv2_final.keras"

HISTORY_PATH = REPORT_DIR / "mobilenetv2_training_history.csv"
METRICS_PATH = REPORT_DIR / "mobilenetv2_metrics.csv"

CLASSIFICATION_PATH = (
    REPORT_DIR / "mobilenetv2_classification_report.txt"
)

CONFUSION_PATH = (
    REPORT_DIR / "mobilenetv2_confusion_matrix.csv"
)

LAYERS_PATH = REPORT_DIR / "mobilenetv2_model_layers.csv"

EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42

MODEL_NAME = "VisionGuard_MobileNetV2"


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

print("\n========== VISIONGUARD MOBILENETV2 ==========")

print("TensorFlow:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Execution device: CPU")


# ============================================================
# BUILD MOBILENETV2 MODEL
# ============================================================

def build_mobilenetv2_model():

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="input_image"
    )

    # Convert original pixels from 0-255 to 0-1.
    x = tf.keras.layers.Rescaling(
        1.0 / 255.0,
        name="initial_rescaling"
    )(inputs)

    # Apply the same augmentation strategy as the baseline.
    augmentation = create_augmentation()

    x = augmentation(x)

    # MobileNetV2 expects values between -1 and 1.
    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing"
    )(x)

    # Load ImageNet pretrained MobileNetV2.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze all pretrained layers initially.
    base_model.trainable = False

    # Keep pretrained BatchNormalization layers in inference mode.
    x = base_model(
        x,
        training=False
    )

    # Extract features.
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    # Binary classification head.
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        name="dense_features"
    )(x)

    x = tf.keras.layers.Dropout(
        0.3,
        name="classification_dropout"
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
# SAVE MODEL ARCHITECTURE
# ============================================================

def save_model_layers(model):

    layer_data = []

    for layer in model.layers:

        layer_data.append({
            "layer_name": layer.name,
            "layer_type": layer.__class__.__name__,
            "trainable": layer.trainable,
            "parameters": layer.count_params()
        })

    pd.DataFrame(layer_data).to_csv(
        LAYERS_PATH,
        index=False
    )

    print("Saved model architecture:", LAYERS_PATH)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

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

    print("Saved training history:", HISTORY_PATH)


# ============================================================
# PLOT TRAINING RESULTS
# ============================================================

def save_training_figures(history):

    history_data = history.history

    epochs = range(1, len(history_data["loss"]) + 1)

    # Accuracy plot.
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history_data["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("MobileNetV2 Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        FIGURE_DIR / "mobilenetv2_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Loss plot.
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history_data["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history_data["val_loss"],
        label="Validation Loss"
    )

    plt.title("MobileNetV2 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        FIGURE_DIR / "mobilenetv2_loss.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Training figures saved.")


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model, test_dataset):

    print("\n========== TEST EVALUATION ==========")

    evaluation = model.evaluate(
        test_dataset,
        verbose=1
    )

    test_loss = float(evaluation[0])

    # Predict probabilities.
    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    # Convert probabilities to binary predictions.
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Extract actual labels from the test dataset.
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

    print("\nTEST METRICS")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC:  {auc:.4f}")

    # Classification report.
    report = classification_report(
        actual_labels,
        predictions,
        labels=[0, 1],
        target_names=["Real", "AI Generated"],
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

    # Confusion matrix.
    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1]
    )

    pd.DataFrame(
        matrix,
        index=["Actual Real", "Actual AI Generated"],
        columns=["Predicted Real", "Predicted AI Generated"]
    ).to_csv(CONFUSION_PATH)

    # Confusion matrix visualization.
    plt.figure(figsize=(7, 6))

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title("MobileNetV2 Confusion Matrix")
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
        FIGURE_DIR / "mobilenetv2_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Save numerical metrics.
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

    pd.DataFrame([metrics]).to_csv(
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

    print("\n========== BUILDING MOBILENETV2 ==========")

    model, base_model = build_mobilenetv2_model()

    print("\nPretrained MobileNetV2 loaded.")

    print("Frozen pretrained layers:", len(base_model.layers))

    print("Trainable parameters:", model.count_params())

    model.summary()

    save_model_layers(model)

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
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.000001,
            verbose=1
        )
    ]

    print("\n========== STARTING TRAINING ==========")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\n========== SAVING MODEL ==========")

    model.save(FINAL_MODEL_PATH)

    print("Best model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)

    save_training_history(history)

    save_training_figures(history)

    # Load the best validation checkpoint for final evaluation.
    best_model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    evaluate_model(
        best_model,
        test_dataset
    )

    print("\n========== MOBILENETV2 PIPELINE COMPLETE ==========")

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
