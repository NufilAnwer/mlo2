import os
from pathlib import Path

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "task_3_5_champion.keras")
    )
)

THRESHOLD = float(
    os.getenv("THRESHOLD", "0.4980708956718445")
)

IMAGE_SIZE = (160, 160)
MODEL_NAME = MODEL_PATH.stem


if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Champion model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("\n========== VISIONGUARD INFERENCE ==========")
print("Champion model loaded:", MODEL_PATH)
print("Model identifier:", MODEL_NAME)
print("Model input shape:", model.input_shape)
print("Classification threshold:", THRESHOLD)


def preprocess_image(image_bytes):

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        method="bilinear"
    )

    image = tf.cast(image, tf.float32)

    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE[0], IMAGE_SIZE[1], 3]
    )

    image = tf.expand_dims(image, axis=0)

    return image


def predict_image(image_bytes):

    if not image_bytes:
        raise ValueError("No image data was provided.")

    image_batch = preprocess_image(image_bytes)

    prediction = model.predict(
        image_batch,
        verbose=0
    )

    ai_probability = float(prediction[0][0])

    if ai_probability >= THRESHOLD:
        predicted_label = "AI Generated"
        confidence = ai_probability
    else:
        predicted_label = "Real"
        confidence = 1.0 - ai_probability

    return {
        "label": predicted_label,
        "confidence": confidence,
        "ai_probability": ai_probability,
        "threshold": THRESHOLD,
        "model_name": MODEL_NAME
    }
