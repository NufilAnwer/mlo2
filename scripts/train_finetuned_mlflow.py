
import os

os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    log_loss
)

from scripts.input_pipeline import (
    PROJECT_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    VAL_CSV,
    TEST_CSV,
    load_datasets,
    load_split_csv,
    create_dataset
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "mlflow" / "task_3_3"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_MODEL = MODEL_DIR / "mobilenetv2_best.keras"

BEST_MODEL_PATH = (
    MODEL_DIR / "task_3_3_mobilenetv2_best.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "task_3_3_mobilenetv2_final.keras"
)

EPOCHS = 6
LEARNING_RATE = 0.00001
UNFREEZE_LAST_LAYERS = 30

MODEL_NAME = "VisionGuard_MobileNetV2_Task_3_3"

EXPERIMENT_NAME = "VisionGuard_Task_3_3_Tracked_Training"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# VALIDATE DAGSHUB CONFIGURATION
# ============================================================

def configure_mlflow():

    if not TRACKING_URI:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is missing."
        )

    if not os.getenv("MLFLOW_TRACKING_USERNAME"):
        raise EnvironmentError(
            "MLFLOW_TRACKING_USERNAME is missing."
        )

    if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        raise EnvironmentError(
            "MLFLOW_TRACKING_PASSWORD is missing."
        )

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Automatically capture TensorFlow/Keras training information.
    mlflow.tensorflow.autolog()

    print("MLflow tracking URI:", TRACKING_URI)
    print("MLflow experiment:", EXPERIMENT_NAME)


# ============================================================
# REPOSITORY INFORMATION
# ============================================================

def get_command_output(command):

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True
    )

    return result.returncode, result.stdout.strip(), result.stderr.strip()


def log_repository_tags():

    git_code, git_commit, git_error = get_command_output(
        ["git", "rev-parse", "HEAD"]
    )

    if git_code == 0:
        mlflow.set_tag("git_commit", git_commit)
    else:
        mlflow.set_tag("git_commit", "unavailable")

    _, git_status, _ = get_command_output(
        ["git", "status", "--short"]
    )

    mlflow.set_tag(
        "git_worktree_status",
        "clean" if not git_status else "modified_or_untracked"
    )

    dvc_code, dvc_output, dvc_error = get_command_output(
        ["dvc", "--version"]
    )

    mlflow.set_tag(
        "dvc_version",
        dvc_output if dvc_code == 0 else "unavailable"
    )

    status_code, status_output, status_error = get_command_output(
        ["dvc", "status"]
    )

    status_text = status_output + " " + status_error

    if "not inside of a DVC repository" in status_text:
        dvc_status = "not_initialized"
    elif status_code == 0:
        dvc_status = "initialized"
    else:
        dvc_status = "status_check_failed"

    mlflow.set_tag("dvc_repo_status", dvc_status)
    mlflow.set_tag("dvc_revision", "not_available")

    mlflow.set_tag("random_seed", str(SEED))

    print("Git commit:", git_commit)
    print("DVC version:", dvc_output)
    print("DVC repository status:", dvc_status)


# ============================================================
# LOAD AND CONFIGURE FINE-TUNING MODEL
# ============================================================

def load_and_configure_model():

    if not INPUT_MODEL.exists():
        raise FileNotFoundError(
            f"Starting model not found: {INPUT_MODEL}"
        )

    print("\nLoading existing MobileNetV2 checkpoint...")

    model = tf.keras.models.load_model(INPUT_MODEL)

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

    base_model.trainable = True

    for layer in base_model.layers:
        layer.trainable = False

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

    trainable_layers = sum(
        1 for layer in base_model.layers
        if layer.trainable
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print("Base model:", base_model.name)
    print("Trainable pretrained layers:", trainable_layers)

    return model


# ============================================================
# MANUAL MLFLOW EPOCH CALLBACK
# ============================================================

class MLflowEpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        if not logs:
            return

        for metric_name, metric_value in logs.items():

            if metric_value is None:
                continue

            mlflow.log_metric(
                f"manual_{metric_name}",
                float(metric_value),
                step=epoch + 1
            )

        print(
            f"Manually logged MLflow metrics for epoch {epoch + 1}"
        )


# ============================================================
# VALIDATION THRESHOLD SELECTION
# ============================================================

def select_validation_threshold(model, val_dataset):

    val_df = load_split_csv(VAL_CSV)

    actual_labels = val_df["label"].to_numpy(dtype=int)

    probabilities = model.predict(
        val_dataset,
        verbose=1
    ).flatten()

    precision, recall, thresholds = precision_recall_curve(
        actual_labels,
        probabilities
    )

    denominator = precision[:-1] + recall[:-1]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    best_index = int(np.argmax(f1_values))

    threshold = float(thresholds[best_index])

    validation_f1 = float(f1_values[best_index])

    mlflow.log_metric(
        "validation_selected_threshold",
        threshold
    )

    mlflow.log_metric(
        "validation_selected_f1",
        validation_f1
    )

    print("Validation-selected threshold:", threshold)
    print("Validation F1:", validation_f1)

    return threshold


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_and_log(model, threshold, test_dataset):

    test_df = load_split_csv(TEST_CSV)

    actual_labels = test_df["label"].to_numpy(dtype=int)

    probabilities = model.predict(
        test_dataset,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {

        "test_accuracy": float(
            accuracy_score(actual_labels, predictions)
        ),

        "test_precision": float(
            precision_score(
                actual_labels,
                predictions,
                zero_division=0
            )
        ),

        "test_recall": float(
            recall_score(
                actual_labels,
                predictions,
                zero_division=0
            )
        ),

        "test_f1": float(
            f1_score(
                actual_labels,
                predictions,
                zero_division=0
            )
        ),

        "test_roc_auc": float(
            roc_auc_score(actual_labels, probabilities)
        ),

        "test_loss": float(
            log_loss(
                actual_labels,
                probabilities,
                labels=[0, 1]
            )
        )
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.log_metric("test_threshold", threshold)

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1]
    )

    matrix_path = OUTPUT_DIR / "confusion_matrix.csv"

    pd.DataFrame(
        matrix,
        index=["Actual Real", "Actual AI"],
        columns=["Predicted Real", "Predicted AI"]
    ).to_csv(matrix_path)

    report = classification_report(
        actual_labels,
        predictions,
        labels=[0, 1],
        target_names=["Real", "AI Generated"],
        zero_division=0
    )

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    fpr, tpr, _ = roc_curve(
        actual_labels,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"VisionGuard (AUC = {metrics['test_roc_auc']:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Task 3.3 - VisionGuard ROC Curve")
    plt.legend()
    plt.grid(True)

    roc_path = OUTPUT_DIR / "roc_curve.png"

    plt.savefig(
        roc_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    metrics_path = OUTPUT_DIR / "test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    mlflow.log_artifact(
        str(matrix_path),
        artifact_path="evaluation"
    )

    mlflow.log_artifact(
        str(report_path),
        artifact_path="evaluation"
    )

    mlflow.log_artifact(
        str(roc_path),
        artifact_path="evaluation"
    )

    mlflow.log_artifact(
        str(metrics_path),
        artifact_path="evaluation"
    )

    print("\nTask 3.3 test metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(report)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== TASK 3.3: VISIONGUARD MLFLOW ==========")

    configure_mlflow()

    print("\nLoading datasets...")

    train_dataset, val_dataset, test_dataset = load_datasets()

    with mlflow.start_run(
        run_name="VisionGuard_MobileNetV2_FineTuning"
    ):

        log_repository_tags()

        # Log training configuration.
        mlflow.log_params({

            "model": MODEL_NAME,

            "starting_checkpoint": INPUT_MODEL.name,

            "epochs": EPOCHS,

            "learning_rate": LEARNING_RATE,

            "unfreeze_last_layers": UNFREEZE_LAST_LAYERS,

            "batch_size": BATCH_SIZE,

            "image_height": IMAGE_SIZE[0],

            "image_width": IMAGE_SIZE[1],

            "seed": SEED,

            "optimizer": "Adam",

            "loss": "binary_crossentropy",

            "execution_device": "CPU"
        })

        model = load_and_configure_model()

        model.summary()

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
            ),

            MLflowEpochLogger()
        ]

        print("\nStarting MLflow-tracked fine-tuning...")

        history = model.fit(

            train_dataset,

            validation_data=val_dataset,

            epochs=EPOCHS,

            callbacks=callbacks,

            verbose=1
        )

        model.save(FINAL_MODEL_PATH)

        mlflow.log_artifact(
            str(FINAL_MODEL_PATH),
            artifact_path="keras_checkpoint"
        )

        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Best model checkpoint was not created."
            )

        best_model = tf.keras.models.load_model(
            BEST_MODEL_PATH,
            compile=False
        )

        # Save the training history as an MLflow artifact.
        history_path = OUTPUT_DIR / "training_history.csv"

        pd.DataFrame(history.history).to_csv(
            history_path,
            index_label="epoch"
        )

        mlflow.log_artifact(
            str(history_path),
            artifact_path="training"
        )

        # Log the best trained model using MLflow's TensorFlow flavor.
        mlflow.tensorflow.log_model(
            best_model,
            name="visionguard_finetuned_best"
        )

        # Select threshold on validation data, then evaluate test once.
        val_df = load_split_csv(VAL_CSV)

        val_eval_dataset = create_dataset(
            val_df,
            training=False
        )

        threshold = select_validation_threshold(
            best_model,
            val_eval_dataset
        )

        test_df = load_split_csv(TEST_CSV)

        test_eval_dataset = create_dataset(
            test_df,
            training=False
        )

        evaluate_and_log(
            best_model,
            threshold,
            test_eval_dataset
        )

        print("\nBest model:", BEST_MODEL_PATH)
        print("Final model:", FINAL_MODEL_PATH)

        print("\n========== TASK 3.3 RUN COMPLETED ==========")


if __name__ == "__main__":
    main()
