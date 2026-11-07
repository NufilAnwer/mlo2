
import os
from pathlib import Path

import mlflow
import tensorflow as tf
from mlflow import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "VisionGuard_MobileNetV2"
EXPERIMENT_NAME = "VisionGuard_Task_3_6_Model_Registry"

MODELS = [
    {
        "name": "Task_3_3_FineTuned",
        "path": PROJECT_DIR / "models" / "task_3_3_mobilenetv2_best.keras",
        "source_run_id": "fc023b8bc06e4cf687a801773e336df0",
    },
    {
        "name": "Task_3_5_Champion",
        "path": PROJECT_DIR / "models" / "task_3_5_champion.keras",
        "source_run_id": "39642dc0decd482c937c9682cdb00b49",
    },
]


def configure_mlflow():

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri or not username or not password:
        raise EnvironmentError(
            "DagsHub MLflow environment variables are missing. "
            "Set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME "
            "and MLFLOW_TRACKING_PASSWORD."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Registry URI:", mlflow.get_registry_uri())


def register_model(model_info):

    path = model_info["path"]

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    print("\nRegistering:", model_info["name"])
    print("Source file:", path)

    # Load the existing trained model. No training takes place.
    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    with mlflow.start_run(run_name=model_info["name"]):

        mlflow.set_tags({
            "assignment_task": "3.6",
            "source_model": model_info["name"],
            "source_run_id": model_info["source_run_id"],
            "source_file": path.name,
            "framework": "TensorFlow/Keras",
        })

        logged_model = mlflow.tensorflow.log_model(
            model=model,
            name="visionguard_model",
            registered_model_name=MODEL_NAME,
            await_registration_for=300,
        )

        version = logged_model.registered_model_version

        if version is None:
            raise RuntimeError(
                "MLflow did not return a registered model version."
            )

        print("Registered version:", version)

    return str(version)


def main():

    print("\n========== TASK 3.6: MODEL REGISTRY ==========")

    configure_mlflow()

    client = MlflowClient()

    # Register the two existing model files.
    versions = {}

    for model_info in MODELS:
        versions[model_info["name"]] = register_model(model_info)

    # Point champion to the Task 3.5 best model.
    champion_version = versions["Task_3_5_Champion"]

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        champion_version
    )

    # Verify the registry.
    registered = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    print("\n========== REGISTRY VERIFICATION ==========")
    print("Registered model:", MODEL_NAME)
    print("Number of versions:", len(registered))

    for version in registered:
        print(
            "Version:", version.version,
            "| Run ID:", version.run_id
        )

    print("\nChampion alias points to version:", champion.version)
    print("Champion run ID:", champion.run_id)

    if len(registered) < 2:
        raise RuntimeError("Fewer than two model versions were registered.")

    if champion.version != champion_version:
        raise RuntimeError("Champion alias points to the wrong version.")

    print("\nTASK 3.6 REGISTRATION SUCCESSFUL")
    print("Model URI:", f"models:/{MODEL_NAME}@champion")


if __name__ == "__main__":
    main()
