"""Local training script for the EvoTrack AI Siamese model."""

from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    from evotrack_ai.data_ingestion import list_image_paths
    from evotrack_ai.siamese_model import build_siamese_model
    from evotrack_ai.tf_data_pipeline import create_tf_dataset
except ModuleNotFoundError:
    from data_ingestion import list_image_paths
    from siamese_model import build_siamese_model
    from tf_data_pipeline import create_tf_dataset


DEFAULT_DATASET_DIR = Path("data/raw/Br35H/no")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "evotrack_siamese_best.keras"
DEFAULT_HISTORY_PLOT_PATH = DEFAULT_OUTPUT_DIR / "training_history.png"


def ensure_output_dirs() -> None:
    """Create output folders used for model checkpoints and training plots."""
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_image_paths(
    image_paths,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Split image paths into train and validation lists.

    Only paths are shuffled and split. Images are never loaded into memory.

    Args:
        image_paths: Sequence of image paths.
        validation_fraction: Fraction of paths used for validation.
        seed: Random seed for deterministic shuffling.

    Returns:
        ``train_paths, val_paths``.

    Raises:
        ValueError: If the split would produce an empty train or validation set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    paths = list(image_paths)

    if len(paths) < 2:
        raise ValueError("At least two image paths are required for splitting.")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(paths))
    shuffled_paths = [paths[index] for index in shuffled_indices]

    num_val = int(round(len(shuffled_paths) * validation_fraction))
    num_val = max(1, min(num_val, len(shuffled_paths) - 1))

    val_paths = shuffled_paths[:num_val]
    train_paths = shuffled_paths[num_val:]

    if not train_paths or not val_paths:
        raise ValueError("Train and validation splits must not be empty.")

    return train_paths, val_paths


def prepare_datasets(
    train_paths,
    val_paths,
    batch_size: int = 32,
    seed: int = 42,
):
    """Create TensorFlow train and validation datasets."""
    train_dataset = create_tf_dataset(
        train_paths,
        batch_size=batch_size,
        seed=seed,
    )
    val_dataset = create_tf_dataset(
        val_paths,
        batch_size=batch_size,
        seed=seed + 1,
    )
    train_dataset = train_dataset.map(
        _format_batch_for_keras,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_dataset = val_dataset.map(
        _format_batch_for_keras,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_dataset, val_dataset


def _format_batch_for_keras(
    image_1: tf.Tensor,
    image_2: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Format a dataset batch for a two-input Keras model."""
    return (image_1, image_2), label


def build_callbacks(
    model_path: Path = DEFAULT_MODEL_PATH,
    patience: int = 3,
) -> list:
    """Build Keras callbacks for checkpointing and early stopping."""
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    return [model_checkpoint, early_stopping]


def plot_training_history(
    history,
    output_path: Path = DEFAULT_HISTORY_PLOT_PATH,
) -> None:
    """Save loss and accuracy curves from a Keras training history."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_data = history.history
    epochs = range(1, len(history_data.get("loss", [])) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_data.get("loss", []), label="loss")
    plt.plot(epochs, history_data.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if "accuracy" in history_data and "val_accuracy" in history_data:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history_data["accuracy"], label="accuracy")
        plt.plot(epochs, history_data["val_accuracy"], label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = 3,
    steps_per_epoch: int = 20,
    validation_steps: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
):
    """Train the Siamese model locally on synthetic pairs.

    Args:
        dataset_dir: Directory containing healthy T0 images.
        epochs: Number of training epochs.
        steps_per_epoch: Training batches per epoch.
        validation_steps: Validation batches per epoch.
        batch_size: Number of pairs per batch.
        learning_rate: Adam optimizer learning rate.

    Returns:
        ``model, history`` from Keras training.
    """
    ensure_output_dirs()

    image_paths = list_image_paths(dataset_dir)
    train_paths, val_paths = split_image_paths(image_paths)
    train_dataset, val_dataset = prepare_datasets(
        train_paths,
        val_paths,
        batch_size=batch_size,
    )

    model = build_siamese_model(learning_rate=learning_rate)
    callbacks = build_callbacks()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    plot_training_history(history)

    return model, history


if __name__ == "__main__":
    train(
        epochs=2,
        steps_per_epoch=10,
        validation_steps=3,
        batch_size=16,
    )

    print("Training completed")
    print(f"Best model path: {DEFAULT_MODEL_PATH}")
    print(f"History plot path: {DEFAULT_HISTORY_PLOT_PATH}")
