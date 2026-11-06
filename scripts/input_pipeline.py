
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = PROJECT_DIR / "data" / "splits"

OUTPUT_FIGURES = PROJECT_DIR / "outputs" / "figures"
OUTPUT_REPORTS = PROJECT_DIR / "outputs" / "reports"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 4
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD AND VALIDATE CSV FILES
# ============================================================

def load_split_csv(csv_path):

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = ["filepath", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in {csv_path}"
            )

    if df.empty:
        raise ValueError(
            f"Dataset split is empty: {csv_path}"
        )

    df = df[required_columns].copy()

    # Convert labels into numeric values.
    if not pd.api.types.is_numeric_dtype(df["label"]):

        label_mapping = {
            "real": 0,
            "ai": 1,
            "ai-generated": 1,
            "fake": 1
        }

        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(label_mapping)
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    if df["label"].isna().any():
        raise ValueError(
            f"Invalid labels found in {csv_path}"
        )

    df["label"] = df["label"].astype("int32")

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"Labels must be 0 (Real) or 1 (AI): {csv_path}"
        )

    # Resolve image paths.
    resolved_paths = []

    for filepath in df["filepath"]:

        path_string = str(filepath).strip()

        image_path = Path(path_string)

        if image_path.is_absolute():
            resolved_path = image_path
        else:
            project_relative = PROJECT_DIR / image_path

            if project_relative.exists():
                resolved_path = project_relative
            else:
                resolved_path = image_path.resolve()

        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Image does not exist: {resolved_path}"
            )

        resolved_paths.append(str(resolved_path))

    df["filepath"] = resolved_paths

    return df


# ============================================================
# IMAGE LOADING AND PREPROCESSING
# ============================================================

def preprocess_image(filepath, label):

    image_bytes = tf.io.read_file(filepath)

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

    label = tf.cast(label, tf.float32)

    return image, label


# ============================================================
# BUILD TF.DATA DATASETS
# ============================================================

def create_dataset(df, training=False):

    image_paths = df["filepath"].values

    labels = df["label"].values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    # Keep prefetch limited to reduce RAM consumption.
    dataset = dataset.prefetch(1)

    options = tf.data.Options()

    options.deterministic = True

    options.threading.private_threadpool_size = 2

    options.threading.max_intra_op_parallelism = 1

    dataset = dataset.with_options(options)

    return dataset


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_augmentation():

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=SEED
            ),

            tf.keras.layers.RandomRotation(
                factor=0.08,
                seed=SEED
            ),

            tf.keras.layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                seed=SEED
            )
        ],
        name="data_augmentation"
    )

    return augmentation


# ============================================================
# LOAD ALL DATASETS
# ============================================================

def load_datasets():

    print("\n========== LOADING DATASETS ==========")

    train_df = load_split_csv(TRAIN_CSV)

    val_df = load_split_csv(VAL_CSV)

    test_df = load_split_csv(TEST_CSV)

    # Check that there are no duplicate images across splits.
    train_files = set(train_df["filepath"])

    val_files = set(val_df["filepath"])

    test_files = set(test_df["filepath"])

    if train_files.intersection(val_files):
        raise ValueError(
            "Data leakage: training and validation overlap."
        )

    if train_files.intersection(test_files):
        raise ValueError(
            "Data leakage: training and test overlap."
        )

    if val_files.intersection(test_files):
        raise ValueError(
            "Data leakage: validation and test overlap."
        )

    print("\nTRAIN DATA")
    print("Images:", len(train_df))
    print(train_df["label"].value_counts().sort_index())

    print("\nVALIDATION DATA")
    print("Images:", len(val_df))
    print(val_df["label"].value_counts().sort_index())

    print("\nTEST DATA")
    print("Images:", len(test_df))
    print(test_df["label"].value_counts().sort_index())

    print("\nTotal images:",
          len(train_df) + len(val_df) + len(test_df))

    print("\nData leakage check: PASSED")

    train_dataset = create_dataset(
        train_df,
        training=True
    )

    val_dataset = create_dataset(
        val_df,
        training=False
    )

    test_dataset = create_dataset(
        test_df,
        training=False
    )

    return train_dataset, val_dataset, test_dataset


# ============================================================
# VISUALIZE AUGMENTED TRAINING BATCH
# ============================================================

def save_augmented_batch():

    print("\nGenerating augmented training visualization...")

    train_df = load_split_csv(TRAIN_CSV)

    sample_df = train_df.sample(
        n=min(8, len(train_df)),
        random_state=SEED
    )

    sample_dataset = create_dataset(
        sample_df,
        training=False
    )

    images, labels = next(iter(sample_dataset))

    augmentation = create_augmentation()

    augmented_images = augmentation(
        images,
        training=True
    )

    plt.figure(figsize=(12, 8))

    for index in range(len(images)):

        plt.subplot(2, 4, index + 1)

        image = augmented_images[index].numpy()

        image = np.clip(image, 0, 255).astype(np.uint8)

        plt.imshow(image)

        label = int(labels[index].numpy())

        class_name = "AI Generated" if label == 1 else "Real"

        plt.title(class_name)

        plt.axis("off")

    plt.suptitle(
        "VisionGuard - Augmented Training Images"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FIGURES / "augmented_training_batch.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset = load_datasets()

    save_augmented_batch()

    print("\n========== INPUT PIPELINE COMPLETE ==========")

    print("Image size:", IMAGE_SIZE)

    print("Batch size:", BATCH_SIZE)

    print("Prefetch: 1")

    print("All datasets loaded successfully.")
