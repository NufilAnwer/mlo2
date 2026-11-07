
import os
import argparse

import numpy as np
import tensorflow as tf
import mlflow


# ============================================================
# TASK 3.6: LOAD AND PREDICT USING THE CHAMPION ALIAS
# ============================================================

MODEL_NAME = "VisionGuard_MobileNetV2"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

IMAGE_SIZE = (160, 160)

# Selected using the Task 3.5 validation set.
THRESHOLD = 0.4980708956718445


def load_registered_model():

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is missing."
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print("Loading model:", MODEL_URI)

    model = mlflow.tensorflow.load_model(MODEL_URI)

    print("Champion model loaded successfully.")

    return model


def preprocess_image(image_path):

    image_data = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    )

    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)

    return image


def predict_image(model, image_path):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = preprocess_image(image_path)

    prediction = model.predict(
        image,
        verbose=0
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    predicted_class = (
        "AI Generated"
        if probability >= THRESHOLD
        else "Real"
    )

    print("\n========== VISIONGUARD PREDICTION ==========")
    print("Image:", image_path)
    print(f"AI Generated probability: {probability:.4f}")
    print("Selected threshold:", THRESHOLD)
    print("Predicted class:", predicted_class)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image to classify"
    )

    args = parser.parse_args()

    model = load_registered_model()

    predict_image(
        model,
        args.image
    )


if __name__ == "__main__":
    main()
