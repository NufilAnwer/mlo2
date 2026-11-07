
import os
import sys
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parents[1]

RUN_ID = "fc023b8bc06e4cf687a801773e336df0"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "mlflow" / "task_3_3"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"

BEST_MODEL = (
    PROJECT_DIR / "models" / "task_3_3_mobilenetv2_best.keras"
)

SPLIT_FILES = {
    "train.csv": PROJECT_DIR / "data" / "splits" / "train.csv",
    "val.csv": PROJECT_DIR / "data" / "splits" / "val.csv",
    "test.csv": PROJECT_DIR / "data" / "splits" / "test.csv",
}


def create_environment_report():
    report_path = OUTPUT_DIR / "environment.txt"

    packages = [
        "tensorflow",
        "keras",
        "mlflow",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "dagshub",
        "dvc",
    ]

    lines = [
        f"Python version: {sys.version}",
        f"Operating system: {platform.platform()}",
        "",
        "Important package versions:",
    ]

    for package in packages:
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "Not installed"

        lines.append(f"{package}: {package_version}")

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return report_path


def main():
    print("\n========== TASK 3.4: MLFLOW ARTIFACT LOGGING ==========")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not configured.")

    if not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        raise RuntimeError("DagsHub username is missing.")

    if not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError("DagsHub token is missing.")

    required_files = [
        REQUIREMENTS_FILE,
        BEST_MODEL,
        *SPLIT_FILES.values(),
    ]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Task 3.3 output directory not found: {OUTPUT_DIR}"
        )

    environment_report = create_environment_report()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    print("Tracking URI:", tracking_uri)
    print("Existing run ID:", RUN_ID)

    with mlflow.start_run(run_id=RUN_ID):

        print("\nLogging dataset split files...")

        for filename, file_path in SPLIT_FILES.items():
            mlflow.log_artifact(
                str(file_path),
                artifact_path="dataset_splits"
            )

            print("Logged:", filename)

        print("\nLogging package requirements...")

        mlflow.log_artifact(
            str(REQUIREMENTS_FILE),
            artifact_path="environment"
        )

        mlflow.log_artifact(
            str(environment_report),
            artifact_path="environment"
        )

        print("Logged requirements.txt")
        print("Logged environment.txt")

        print("\nLogging Task 3.3 output directory...")

        mlflow.log_artifacts(
            str(OUTPUT_DIR),
            artifact_path="task_3_3_outputs"
        )

        print("Logged Task 3.3 output files.")

        print("\nLogging best Keras model file...")

        mlflow.log_artifact(
            str(BEST_MODEL),
            artifact_path="keras_model_file"
        )

        print("Logged best Keras model.")

    print("\n========== VERIFYING MLFLOW ARTIFACTS ==========")

    artifacts = client.list_artifacts(RUN_ID)

    for artifact in artifacts:
        print(artifact.path)

    print("\n========== TASK 3.4 COMPLETED ==========")
    print("Run ID:", RUN_ID)
    print(
        "DagsHub run:",
        f"{tracking_uri}/#/experiments/1/runs/{RUN_ID}"
    )


if __name__ == "__main__":
    main()
