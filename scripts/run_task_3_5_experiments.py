
import os

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from sklearn.metrics import precision_recall_curve

from scripts.input_pipeline import (
    PROJECT_DIR,
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    load_datasets,
)


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(PROJECT_DIR)

INPUT_MODEL = (
    PROJECT_DIR
    / "models"
    / "mobilenetv2_best.keras"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "mlflow"
    / "task_3_5"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_DIR = OUTPUT_DIR / "histories"

CHAMPION_MODEL = (
    PROJECT_DIR
    / "models"
    / "task_3_5_champion.keras"
)

COMPARISON_CSV = OUTPUT_DIR / "experiment_comparison.csv"

SUMMARY_JSON = OUTPUT_DIR / "experiment_summary.json"

EXPERIMENT_NAME = "VisionGuard_Task_3_5_Comparison"

EPOCHS = 6

BASE_LEARNING_RATE = 0.00001
BASE_UNFROZEN_LAYERS = 30
BASE_DROPOUT = 0.30


# ============================================================
# SIX CONTROLLED EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = [

    {
        "name": "run_1_baseline",
        "learning_rate": 0.00001,
        "unfrozen_layers": 30,
        "dropout_rate": 0.30,
    },

    {
        "name": "run_2_high_learning_rate",
        "learning_rate": 0.00005,
        "unfrozen_layers": 30,
        "dropout_rate": 0.30,
    },

    {
        "name": "run_3_low_learning_rate",
        "learning_rate": 0.000005,
        "unfrozen_layers": 30,
        "dropout_rate": 0.30,
    },

    {
        "name": "run_4_fewer_unfrozen_layers",
        "learning_rate": 0.00001,
        "unfrozen_layers": 20,
        "dropout_rate": 0.30,
    },

    {
        "name": "run_5_more_unfrozen_layers",
        "learning_rate": 0.00001,
        "unfrozen_layers": 40,
        "dropout_rate": 0.30,
    },

    {
        "name": "run_6_higher_dropout",
        "learning_rate": 0.00001,
        "unfrozen_layers": 30,
        "dropout_rate": 0.50,
    },

]


# ============================================================
# CONFIGURE DAGSHUB MLFLOW
# ============================================================

def configure_mlflow():

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    username = os.environ.get(
        "MLFLOW_TRACKING_USERNAME"
    )

    password = os.environ.get(
        "MLFLOW_TRACKING_PASSWORD"
    )

    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    if not username or not password:
        raise EnvironmentError(
            "DagsHub username or token is missing."
        )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nDagsHub MLflow configured.")
    print("Tracking URI:", tracking_uri)
    print("Experiment:", EXPERIMENT_NAME)


# ============================================================
# FIND NESTED MOBILENETV2 MODEL
# ============================================================

def find_base_model(model):

    for layer in model.layers:

        if (
            isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        ):

            return layer

        if isinstance(layer, tf.keras.Model):

            try:
                return find_base_model(layer)

            except ValueError:
                pass

    raise ValueError(
        "MobileNetV2 backbone was not found."
    )


# ============================================================
# UPDATE DROPOUT LAYERS
# ============================================================

def update_dropout_rates(model, dropout_rate):

    updated_layers = 0

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Dropout
        ):

            layer.rate = float(dropout_rate)

            updated_layers += 1

        elif isinstance(layer, tf.keras.Model):

            updated_layers += update_dropout_rates(
                layer,
                dropout_rate
            )

    return updated_layers


# ============================================================
# BUILD EXPERIMENT MODEL
# ============================================================

def build_model(config):

    model = tf.keras.models.load_model(
        INPUT_MODEL,
        compile=False
    )

    base_model = find_base_model(model)

    # Initially freeze the complete backbone.

    base_model.trainable = True

    for layer in base_model.layers:

        layer.trainable = False

    # Unfreeze only the selected final layers.

    number_of_layers = config["unfrozen_layers"]

    if number_of_layers > len(base_model.layers):

        raise ValueError(
            "Requested unfrozen layers exceed backbone size."
        )

    selected_layers = base_model.layers[
        -number_of_layers:
    ]

    for layer in selected_layers:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True

    # Update both classification-head Dropout layers.

    updated_dropout_layers = update_dropout_rates(
        model,
        config["dropout_rate"]
    )

    if updated_dropout_layers == 0:

        raise RuntimeError(
            "No Dropout layers were found in the model."
        )

    print(
        "Updated Dropout layers:",
        updated_dropout_layers
    )

    print(
        "Dropout rate:",
        config["dropout_rate"]
    )

    # Compile the model.

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"]
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
            ),

        ],
    )

    trainable_parameters = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    return model, trainable_parameters


# ============================================================
# MLFLOW EPOCH LOGGER
# ============================================================

class MLflowEpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        for metric_name, metric_value in logs.items():

            if metric_value is None:
                continue

            mlflow.log_metric(
                metric_name,
                float(metric_value),
                step=epoch + 1
            )

        print(
            f"MLflow logged epoch {epoch + 1}"
        )


# ============================================================
# CALCULATE VALIDATION F1 AND THRESHOLD
# ============================================================

def calculate_validation_metrics(
    model,
    val_dataset
):

    predictions = model.predict(
        val_dataset,
        verbose=0
    ).reshape(-1)

    actual_labels = []

    for _, labels in val_dataset:

        actual_labels.extend(
            labels.numpy().reshape(-1).tolist()
        )

    actual_labels = np.asarray(
        actual_labels,
        dtype=np.int32
    )

    if len(actual_labels) != len(predictions):

        raise ValueError(
            "Validation labels and predictions have different sizes."
        )

    precision, recall, thresholds = precision_recall_curve(
        actual_labels,
        predictions
    )

    if len(thresholds) == 0:

        threshold = 0.5

        predicted_labels = (
            predictions >= threshold
        ).astype(int)

        tp = np.sum(
            (actual_labels == 1)
            & (predicted_labels == 1)
        )

        fp = np.sum(
            (actual_labels == 0)
            & (predicted_labels == 1)
        )

        fn = np.sum(
            (actual_labels == 1)
            & (predicted_labels == 0)
        )

        validation_f1 = (
            2 * tp / (2 * tp + fp + fn)
            if (2 * tp + fp + fn) > 0
            else 0.0
        )

    else:

        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            /
            (
                precision[:-1]
                + recall[:-1]
                + 1e-8
            )
        )

        best_index = int(
            np.argmax(f1_scores)
        )

        validation_f1 = float(
            f1_scores[best_index]
        )

        threshold = float(
            thresholds[best_index]
        )

    return validation_f1, threshold


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    config,
    train_dataset,
    val_dataset
):

    run_name = config["name"]

    print("\n" + "=" * 55)
    print("STARTING:", run_name)
    print("=" * 55)

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{run_name}_best.keras"
    )

    history_path = (
        HISTORY_DIR
        / f"{run_name}_history.csv"
    )

    model, trainable_parameters = build_model(
        config
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        run_id = mlflow.active_run().info.run_id

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params({

            "learning_rate": config["learning_rate"],

            "unfrozen_layers": config["unfrozen_layers"],

            "dropout_rate": config["dropout_rate"],

            "epochs": EPOCHS,

            "batch_size": BATCH_SIZE,

            "image_height": IMAGE_SIZE[0],

            "image_width": IMAGE_SIZE[1],

            "seed": SEED,

            "optimizer": "Adam",

            "loss": "binary_crossentropy",

            "trainable_parameters": trainable_parameters,

        })

        mlflow.set_tags({

            "model": "MobileNetV2",

            "dataset": "VisionGuard",

            "task": "Task 3.5",

            "selection_metric": "validation_f1",

            "test_set_used_for_selection": "false",

        })

        # ----------------------------------------------------
        # CALLBACKS
        # ----------------------------------------------------

        callbacks = [

            tf.keras.callbacks.ModelCheckpoint(

                filepath=str(checkpoint_path),

                monitor="val_loss",

                save_best_only=True,

                mode="min",

                verbose=1,

            ),

            tf.keras.callbacks.EarlyStopping(

                monitor="val_loss",

                patience=2,

                restore_best_weights=True,

                mode="min",

                verbose=1,

            ),

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=1,

                min_lr=0.0000001,

                verbose=1,

            ),

            MLflowEpochLogger(),

        ]

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        print("\nTraining configuration:")

        print("Learning rate:", config["learning_rate"])

        print("Unfrozen layers:", config["unfrozen_layers"])

        print("Dropout:", config["dropout_rate"])

        print("Epochs:", EPOCHS)

        history = model.fit(

            train_dataset,

            validation_data=val_dataset,

            epochs=EPOCHS,

            callbacks=callbacks,

            verbose=1,

        )

        # ----------------------------------------------------
        # SAVE TRAINING HISTORY
        # ----------------------------------------------------

        history_df = pd.DataFrame(
            history.history
        )

        history_df.index = (
            history_df.index + 1
        )

        history_df.index.name = "epoch"

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        history_df.to_csv(
            history_path
        )

        mlflow.log_artifact(
            str(history_path),
            artifact_path="training_history"
        )

        # ----------------------------------------------------
        # LOAD BEST CHECKPOINT
        # ----------------------------------------------------

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Best checkpoint was not created: {checkpoint_path}"
            )

        best_model = tf.keras.models.load_model(
            checkpoint_path,
            compile=False
        )

        # ----------------------------------------------------
        # VALIDATION EVALUATION
        # ----------------------------------------------------

        validation_f1, threshold = calculate_validation_metrics(
            best_model,
            val_dataset
        )

        best_val_loss = float(
            min(history.history["val_loss"])
        )

        best_epoch = int(
            np.argmin(
                history.history["val_loss"]
            ) + 1
        )

        best_val_accuracy = float(
            history.history["val_accuracy"][best_epoch - 1]
        )

        mlflow.log_metrics({

            "best_validation_loss": best_val_loss,

            "best_validation_accuracy": best_val_accuracy,

            "best_validation_f1": validation_f1,

            "best_epoch": best_epoch,

        })

        mlflow.log_param(
            "validation_f1_threshold",
            threshold
        )

        # ----------------------------------------------------
        # EXPERIMENT RESULTS
        # ----------------------------------------------------

        print("\nExperiment results:")

        print("Run ID:", run_id)

        print("Best epoch:", best_epoch)

        print("Validation loss:", best_val_loss)

        print("Validation accuracy:", best_val_accuracy)

        print("Validation F1:", validation_f1)

        print("Validation threshold:", threshold)

        result = {

            "run_name": run_name,

            "run_id": run_id,

            "learning_rate": config["learning_rate"],

            "unfrozen_layers": config["unfrozen_layers"],

            "dropout_rate": config["dropout_rate"],

            "best_epoch": best_epoch,

            "validation_loss": best_val_loss,

            "validation_accuracy": best_val_accuracy,

            "validation_f1": validation_f1,

            "validation_threshold": threshold,

            "checkpoint": str(checkpoint_path),

        }

    del model
    del best_model

    tf.keras.backend.clear_session()

    return result


# ============================================================
# MAIN EXPERIMENT WORKFLOW
# ============================================================

def main():

    print(
        "\n========== TASK 3.5: SIX MLFLOW EXPERIMENTS =========="
    )

    if not INPUT_MODEL.exists():

        raise FileNotFoundError(
            f"Starting model missing: {INPUT_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    configure_mlflow()

    results = []

    # --------------------------------------------------------
    # RUN ALL SIX EXPERIMENTS
    # --------------------------------------------------------

    for index, config in enumerate(
        EXPERIMENTS,
        start=1
    ):

        print(
            f"\nPreparing datasets for experiment {index}/6"
        )

        tf.keras.utils.set_random_seed(
            SEED
        )

        train_dataset, val_dataset, _ = load_datasets()

        result = run_experiment(
            config,
            train_dataset,
            val_dataset
        )

        results.append(result)

        del train_dataset
        del val_dataset

        tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # COMPARE EXPERIMENTS
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        results
    )

    comparison_df = comparison_df.sort_values(

        by=[
            "validation_f1",
            "validation_loss"
        ],

        ascending=[
            False,
            True
        ],

    ).reset_index(drop=True)

    comparison_df.to_csv(
        COMPARISON_CSV,
        index=False
    )

    print("\n========== EXPERIMENT COMPARISON ==========")

    print(
        comparison_df[
            [
                "run_name",
                "learning_rate",
                "unfrozen_layers",
                "dropout_rate",
                "validation_loss",
                "validation_accuracy",
                "validation_f1",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # SELECT CHAMPION
    # --------------------------------------------------------

    winner = comparison_df.iloc[0].to_dict()

    winner_run_id = winner["run_id"]

    winner_checkpoint = Path(
        winner["checkpoint"]
    )

    if not winner_checkpoint.exists():

        raise FileNotFoundError(
            "Champion checkpoint is missing."
        )

    # --------------------------------------------------------
    # SAVE CHAMPION MODEL LOCALLY
    # --------------------------------------------------------

    CHAMPION_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        winner_checkpoint,
        CHAMPION_MODEL
    )

    print("\nChampion selected:", winner["run_name"])

    print(
        "Champion validation F1:",
        winner["validation_f1"]
    )

    print(
        "Champion validation loss:",
        winner["validation_loss"]
    )

    # --------------------------------------------------------
    # SAVE EXPERIMENT SUMMARY
    # --------------------------------------------------------

    summary = {

        "experiment_name": EXPERIMENT_NAME,

        "total_experiments": len(results),

        "selection_metric": "validation_f1",

        "tie_breaker": "validation_loss",

        "test_set_used_for_selection": False,

        "seed": SEED,

        "epochs": EPOCHS,

        "champion_run_name": winner["run_name"],

        "champion_run_id": winner_run_id,

        "champion_validation_f1": float(
            winner["validation_f1"]
        ),

        "champion_validation_loss": float(
            winner["validation_loss"]
        ),

        "champion_model": str(CHAMPION_MODEL),

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

    # --------------------------------------------------------
    # LOG CHAMPION ARTIFACTS TO WINNING MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        run_id=winner_run_id
    ):

        mlflow.log_artifact(
            str(COMPARISON_CSV),
            artifact_path="experiment_comparison"
        )

        mlflow.log_artifact(
            str(SUMMARY_JSON),
            artifact_path="experiment_summary"
        )

        mlflow.log_artifact(
            str(CHAMPION_MODEL),
            artifact_path="champion_model"
        )

        mlflow.set_tag(
            "champion",
            "true"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n========== TASK 3.5 COMPLETED ==========")

    print("Experiments:", len(results))

    print("Champion:", winner["run_name"])

    print("Champion run ID:", winner_run_id)

    print(
        "Champion validation F1:",
        winner["validation_f1"]
    )

    print("Champion model:", CHAMPION_MODEL)

    print("Comparison CSV:", COMPARISON_CSV)

    print("Summary JSON:", SUMMARY_JSON)

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    tracking_uri = os.environ[
        "MLFLOW_TRACKING_URI"
    ]

    print(
        "\nDagsHub experiment:",
        f"{tracking_uri}/#/experiments/{experiment.experiment_id}"
    )

    print(
        "\nTest data was not used for experiment selection."
    )


if __name__ == "__main__":

    main()
