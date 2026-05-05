"""TensorFlow data pipeline for balanced Siamese MRI batches."""

import time
from pathlib import Path

import tensorflow as tf


try:
    from evotrack_ai.data_ingestion import list_image_paths
    from evotrack_ai.siamese_generator import SiameseDataGenerator
except ModuleNotFoundError:
    from data_ingestion import list_image_paths
    from siamese_generator import SiameseDataGenerator


IMAGE_SIZE = (224, 224)


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """Resize, convert to RGB, and normalize one grayscale image.

    Args:
        image: Grayscale image tensor with shape ``(H, W)`` or ``(H, W, 1)``.

    Returns:
        Tensor with shape ``(224, 224, 3)`` and values in ``[-1, 1]``.
    """
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)

    if image.shape.rank == 2:
        image = image[..., tf.newaxis]
    else:
        image = tf.cond(
            tf.equal(tf.rank(image), 2),
            lambda: image[..., tf.newaxis],
            lambda: image,
        )

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.repeat(image, repeats=3, axis=-1)
    image = (image / 127.5) - 1.0

    return image


def preprocess_pair(
    image_1: tf.Tensor,
    image_2: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Preprocess a Siamese pair and cast its label."""
    image_1 = preprocess_image(image_1)
    image_2 = preprocess_image(image_2)
    label = tf.cast(label, tf.int32)

    return image_1, image_2, label


def create_tf_dataset(
    image_paths,
    batch_size: int = 32,
    shuffle_buffer_size: int = 256,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Create an infinite TensorFlow dataset of balanced Siamese batches.

    Args:
        image_paths: Paths to healthy MRI images.
        batch_size: Number of samples per batch.
        shuffle_buffer_size: Buffer size used by ``dataset.shuffle``.
        seed: Optional seed for image path sampling and shuffling.

    Returns:
        A ``tf.data.Dataset`` yielding ``(images_1, images_2, labels)``.
    """
    siamese_generator = SiameseDataGenerator(image_paths=image_paths, seed=seed)
    pair_generator = siamese_generator.pair_generator

    dataset = tf.data.Dataset.from_generator(
        pair_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    dataset = dataset.map(preprocess_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def benchmark_dataset(dataset: tf.data.Dataset, num_batches: int = 10) -> float:
    """Return the average time needed to retrieve one batch."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    start_time = time.perf_counter()

    for batch_index, _ in enumerate(dataset.take(num_batches), start=1):
        if batch_index >= num_batches:
            break

    elapsed_time = time.perf_counter() - start_time

    return elapsed_time / num_batches


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"

    paths = list_image_paths(dataset_dir)
    dataset = create_tf_dataset(paths)

    images_1, images_2, labels = next(iter(dataset))
    average_batch_time = benchmark_dataset(dataset, num_batches=10)

    print(f"images_1 shape: {images_1.shape}")
    print(f"images_2 shape: {images_2.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"images_1 min/max: {tf.reduce_min(images_1).numpy()} / {tf.reduce_max(images_1).numpy()}")
    print(f"images_2 min/max: {tf.reduce_min(images_2).numpy()} / {tf.reduce_max(images_2).numpy()}")
    print(f"labels unique values: {tf.unique(labels).y.numpy()}")
    print(f"average batch time: {average_batch_time:.4f} seconds")

    assert images_1.shape == (32, 224, 224, 3)
    assert images_2.shape == (32, 224, 224, 3)
    assert labels.shape == (32,)
    assert tf.reduce_max(images_1) <= 1.0
    assert tf.reduce_min(images_1) >= -1.0
