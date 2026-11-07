
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_DIR / "outputs" / "mlflow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = PROJECT_DIR / "mlflow.db"

TRACKING_URI = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)

EXPERIMENT_NAME = "VisionGuard_Task_3_1_Toy_Keras"

EPOCHS = 3
BATCH_SIZE = 128
SEED = 42

LEARNING_RATES = [0.001, 0.0001]


def set_seeds():

    random.seed(SEED)
    np.random.seed(SEED)

    tf.keras.utils.set_random_seed(SEED)


def load_data():

    print("\n========== LOADING MNIST DATASET ==========")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Use 8,000 images for training.
    # Use 2,000 images for validation.
    x_train = x_train[:10000]
    y_train = y_train[:10000]

    x_val = x_train[8000:]
    y_val = y_train[8000:]

    x_train = x_train[:8000]
    y_train = y_train[:8000]

    print("Training images:", len(x_train))
    print("Validation images:", len(x_val))
    print("Test images loaded but not used:", len(x_test))

    return x_train, y_train, x_val, y_val


def build_model(learning_rate):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(28, 28)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class MLflowEpochLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        metrics = {}

        for key, value in logs.items():

            if value is not None:
                metrics[key] = float(value)

        mlflow.log_metrics(
            metrics,
            step=epoch + 1
        )

        print(
            f"MLflow logged metrics for epoch {epoch + 1}"
        )


def create_training_plots(history, learning_rate):

    # Accuracy plot.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"MNIST Accuracy - Learning Rate {learning_rate}"
    )

    plt.legend()
    plt.grid(True)

    accuracy_path = (
        OUTPUT_DIR /
        f"task_3_1_accuracy_lr_{learning_rate}.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Loss plot.
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"MNIST Loss - Learning Rate {learning_rate}"
    )

    plt.legend()
    plt.grid(True)

    loss_path = (
        OUTPUT_DIR /
        f"task_3_1_loss_lr_{learning_rate}.png"
    )

    plt.savefig(
        loss_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return accuracy_path, loss_path


def run_experiment(
    learning_rate,
    x_train,
    y_train,
    x_val,
    y_val
):

    set_seeds()

    run_name = f"learning_rate_{learning_rate}"

    print("\n========== STARTING EXPERIMENT ==========")
    print("Run:", run_name)

    with mlflow.start_run(run_name=run_name):

        # Log experiment parameters.
        mlflow.log_params({

            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "dense_units": 128,
            "dataset": "MNIST",
            "random_seed": SEED

        })

        model = build_model(learning_rate)

        # Train the model.
        history = model.fit(

            x_train,
            y_train,

            validation_data=(
                x_val,
                y_val
            ),

            epochs=EPOCHS,
            batch_size=BATCH_SIZE,

            callbacks=[
                MLflowEpochLogger()
            ],

            verbose=2

        )

        # Create accuracy and loss plots.
        accuracy_path, loss_path = (
            create_training_plots(
                history,
                learning_rate
            )
        )

        # Log training plots.
        mlflow.log_artifact(
            str(accuracy_path)
        )

        mlflow.log_artifact(
            str(loss_path)
        )

        # Log the trained Keras model.
        mlflow.tensorflow.log_model(
            model,
            name="keras_model"
        )

        # Log final validation metrics.
        final_val_accuracy = float(
            history.history["val_accuracy"][-1]
        )

        final_val_loss = float(
            history.history["val_loss"][-1]
        )

        mlflow.log_metrics({

            "final_validation_accuracy": final_val_accuracy,
            "final_validation_loss": final_val_loss

        })

        print("\nRun completed:", run_name)
        print("Final validation accuracy:", final_val_accuracy)
        print("Final validation loss:", final_val_loss)


def main():

    print("\n========== TASK 3.1: LOCAL MLFLOW ==========")

    # Use SQLite instead of the deprecated filesystem backend.
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Tracking URI:", TRACKING_URI)

    # Create or select the experiment.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load MNIST once for both experiments.
    x_train, y_train, x_val, y_val = load_data()

    # Compare two learning rates.
    for learning_rate in LEARNING_RATES:

        run_experiment(
            learning_rate,
            x_train,
            y_train,
            x_val,
            y_val
        )

    print("\n========== TASK 3.1 COMPLETED ==========")

    print("Tracking database:", DATABASE_PATH)
    print("Experiment:", EXPERIMENT_NAME)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
