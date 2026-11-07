
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

from scripts.input_pipeline import (
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    load_split_csv,
    create_dataset,
    create_augmentation
)


PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_DIR / "data" / "splits" / "train.csv"
VAL_CSV = PROJECT_DIR / "data" / "splits" / "val.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "keras_tuner"
TOY_OUTPUT_DIR = OUTPUT_DIR / "toy"
VISIONGUARD_OUTPUT_DIR = OUTPUT_DIR / "visionguard"

FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
MODEL_DIR = PROJECT_DIR / "models"

for folder in [
    TOY_OUTPUT_DIR,
    VISIONGUARD_OUTPUT_DIR,
    FIGURE_DIR,
    MODEL_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def save_trial_results(tuner, output_path):

    records = []

    for trial_id, trial in tuner.oracle.trials.items():

        record = {
            "trial_id": trial_id,
            "status": trial.status,
            "validation_loss": trial.score
        }

        for name, value in trial.hyperparameters.values.items():
            record[name] = value

        records.append(record)

    results_df = pd.DataFrame(records)

    results_df.to_csv(
        output_path,
        index=False
    )

    return results_df


def save_best_hyperparameters(tuner, output_path):

    best_hp = tuner.get_best_hyperparameters(
        num_trials=1
    )[0]

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_hp.values,
            file,
            indent=4
        )

    return best_hp


def plot_trial_results(results_df, output_path, title):

    if results_df.empty:
        print("No trial results available.")
        return

    valid_results = results_df.dropna(
        subset=["validation_loss"]
    )

    if valid_results.empty:
        print("No successful trial results available.")
        return

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(valid_results) + 1),
        valid_results["validation_loss"].values,
        marker="o"
    )

    plt.xlabel("Trial")
    plt.ylabel("Best Validation Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PART A: TOY MNIST EXPERIMENT
# Tune learning rate and Dense units
# ============================================================

class ToyMNISTHyperModel(kt.HyperModel):

    def build(self, hp):

        units = hp.Choice(
            "dense_units",
            values=[64, 128, 256]
        )

        learning_rate = hp.Choice(
            "learning_rate",
            values=[0.001, 0.0005, 0.0001]
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=(28, 28)
                ),

                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(
                    units,
                    activation="relu"
                ),

                tf.keras.layers.Dense(
                    10,
                    activation="softmax"
                )
            ],
            name="toy_mnist_model"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model


def run_toy_experiment():

    print("\n========== TASK 2.7 PART A: TOY MNIST ==========")

    (x_all, y_all), _ = tf.keras.datasets.mnist.load_data()

    # Training and validation samples come from MNIST training data.
    x_train = x_all[:10000].astype("float32") / 255.0
    y_train = y_all[:10000]

    x_val = x_all[10000:12000].astype("float32") / 255.0
    y_val = y_all[10000:12000]

    tuner = kt.RandomSearch(
        hypermodel=ToyMNISTHyperModel(),
        objective="val_loss",
        max_trials=4,
        executions_per_trial=1,
        directory=str(TOY_OUTPUT_DIR),
        project_name="mnist_search",
        overwrite=True,
        seed=SEED
    )

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    results_df = save_trial_results(
        tuner,
        TOY_OUTPUT_DIR / "toy_trial_results.csv"
    )

    best_hp = save_best_hyperparameters(
        tuner,
        TOY_OUTPUT_DIR / "toy_best_hyperparameters.json"
    )

    best_model = tuner.get_best_models(
        num_models=1
    )[0]

    best_model.save(
        MODEL_DIR / "toy_mnist_tuned_best.keras"
    )

    plot_trial_results(
        results_df,
        FIGURE_DIR / "task_2_7_toy_trials.png",
        "Toy MNIST Hyperparameter Tuning"
    )

    print("\nBest toy hyperparameters:")
    print(best_hp.values)

    print("\nToy MNIST experiment completed.")

    del best_model
    del tuner

    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# PART B: VISIONGUARD MOBILENETV2 EXPERIMENT
# Tune learning rate, dropout and unfrozen layers
# ============================================================

class VisionGuardHyperModel(kt.HyperModel):

    def build(self, hp):

        learning_rate = hp.Choice(
            "learning_rate",
            values=[
                0.0001,
                0.00005,
                0.00001,
                0.000005
            ]
        )

        dropout_rate = hp.Choice(
            "dropout_rate",
            values=[
                0.2,
                0.3,
                0.4,
                0.5
            ]
        )

        unfrozen_layers = hp.Choice(
            "unfrozen_layers",
            values=[
                10,
                20,
                30,
                40
            ]
        )

        inputs = tf.keras.Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            ),
            name="input_image"
        )

        # Match the existing MobileNetV2 preprocessing.
        x = tf.keras.layers.Rescaling(
            1.0 / 255.0,
            name="initial_rescaling"
        )(inputs)

        # Use the project's existing augmentation.
        x = create_augmentation()(x)

        x = tf.keras.layers.Rescaling(
            scale=2.0,
            offset=-1.0,
            name="mobilenet_preprocessing"
        )(x)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            ),
            include_top=False,
            weights="imagenet"
        )

        # Freeze all layers first.
        base_model.trainable = True

        for layer in base_model.layers:
            layer.trainable = False

        # Unfreeze the selected final layers.
        for layer in base_model.layers[-unfrozen_layers:]:

            if not isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = True

        # Keep pretrained BatchNormalization layers in inference mode.
        x = base_model(
            x,
            training=False
        )

        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )(x)

        x = tf.keras.layers.Dropout(
            dropout_rate,
            name="dropout"
        )(x)

        x = tf.keras.layers.Dense(
            64,
            activation="relu",
            name="dense_features"
        )(x)

        x = tf.keras.layers.Dropout(
            dropout_rate,
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
            name="visionguard_tuned_mobilenetv2"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
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


def run_visionguard_experiment():

    print("\n========== TASK 2.7 PART B: VISIONGUARD ==========")

    # Only training and validation splits are loaded.
    train_df = load_split_csv(TRAIN_CSV)
    val_df = load_split_csv(VAL_CSV)

    # Confirm no image appears in both splits.
    train_files = set(train_df["filepath"])
    val_files = set(val_df["filepath"])

    if train_files.intersection(val_files):
        raise ValueError(
            "Data leakage: training and validation images overlap."
        )

    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))

    train_dataset = create_dataset(
        train_df,
        training=True
    )

    val_dataset = create_dataset(
        val_df,
        training=False
    )

    tuner = kt.RandomSearch(
        hypermodel=VisionGuardHyperModel(),
        objective="val_loss",
        max_trials=8,
        executions_per_trial=1,
        directory=str(VISIONGUARD_OUTPUT_DIR),
        project_name="mobilenetv2_search",
        overwrite=True,
        seed=SEED
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=4,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=0.000001
            )
        ],
        verbose=1
    )

    results_df = save_trial_results(
        tuner,
        VISIONGUARD_OUTPUT_DIR / "visionguard_trial_results.csv"
    )

    best_hp = save_best_hyperparameters(
        tuner,
        VISIONGUARD_OUTPUT_DIR / "visionguard_best_hyperparameters.json"
    )

    best_trials = tuner.oracle.get_best_trials(
        num_trials=1
    )

    if not best_trials or best_trials[0].score is None:
        raise RuntimeError(
            "No successful VisionGuard tuning trial was completed."
        )

    best_model = tuner.get_best_models(
        num_models=1
    )[0]

    best_model.save(
        MODEL_DIR / "visionguard_tuned_best.keras"
    )

    plot_trial_results(
        results_df,
        FIGURE_DIR / "task_2_7_visionguard_trials.png",
        "VisionGuard MobileNetV2 Hyperparameter Tuning"
    )

    summary = {
        "experiment": "VisionGuard MobileNetV2",
        "objective": "val_loss",
        "requested_trials": 8,
        "recorded_trials": len(results_df),
        "successful_trials": int(
            (results_df["status"] == "COMPLETED").sum()
        ),
        "best_validation_loss": float(
            best_trials[0].score
        ),
        "best_hyperparameters": best_hp.values,
        "training_images": len(train_df),
        "validation_images": len(val_df),
        "test_data_used": False,
        "random_seed": SEED
    }

    with open(
        VISIONGUARD_OUTPUT_DIR / "visionguard_tuning_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nBest VisionGuard hyperparameters:")
    print(best_hp.values)

    print("\nBest validation loss:")
    print(best_trials[0].score)

    print("\nSuccessful trials:")
    print(summary["successful_trials"])

    print("\nVisionGuard experiment completed.")

    del best_model
    del tuner
    del train_dataset
    del val_dataset

    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========== OFFICIAL TASK 2.7 ==========")

    print("TensorFlow:", tf.__version__)
    print("Keras Tuner:", kt.__version__)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Random seed:", SEED)
    print("Device: CPU")

    run_toy_experiment()

    tf.keras.backend.clear_session()
    gc.collect()

    run_visionguard_experiment()

    print("\n========== TASK 2.7 COMPLETED ==========")


if __name__ == "__main__":
    main()
