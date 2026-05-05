"""Federated data simulation with Non-IID scanner biases."""

from pathlib import Path

import numpy as np
import tensorflow as tf


try:
    from evotrack_ai.data_ingestion import list_image_paths
    from evotrack_ai.tf_data_pipeline import create_tf_dataset
except ModuleNotFoundError:
    from data_ingestion import list_image_paths
    from tf_data_pipeline import create_tf_dataset


CLIENT_IDS = ("Client_A", "Client_B", "Client_C")


def split_paths_for_clients(
    image_paths,
    num_clients: int = 3,
    seed: int = 42,
) -> dict:
    """Split image paths into non-overlapping client partitions.

    Args:
        image_paths: Sequence of image paths.
        num_clients: Number of simulated clients.
        seed: Random seed used for shuffling paths.

    Returns:
        Dictionary mapping client IDs to path lists.

    Raises:
        ValueError: If the requested split is invalid or a client would be empty.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    if num_clients > len(CLIENT_IDS):
        raise ValueError(f"num_clients cannot exceed {len(CLIENT_IDS)}.")

    paths = list(image_paths)

    if len(paths) < num_clients:
        raise ValueError("Each client must receive at least one image path.")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(paths))
    shuffled_paths = [paths[index] for index in shuffled_indices]
    path_partitions = np.array_split(shuffled_paths, num_clients)

    client_partitions = {}

    for client_id, partition in zip(CLIENT_IDS[:num_clients], path_partitions):
        client_paths = partition.tolist()

        if not client_paths:
            raise ValueError(f"{client_id} received no image paths.")

        client_partitions[client_id] = client_paths

    return client_partitions


def apply_scanner_bias_to_pair(
    image_1,
    image_2,
    label,
    client_id: str,
):
    """Apply a client-specific scanner bias to a preprocessed image pair.

    Images are expected to be float tensors normalized in the range [-1, 1].
    """
    if client_id == "Client_A":
        biased_image_1 = image_1
        biased_image_2 = image_2
    elif client_id == "Client_B":
        biased_image_1 = image_1 * 0.65
        biased_image_2 = image_2 * 0.65
    elif client_id == "Client_C":
        biased_image_1 = image_1 + 0.25
        biased_image_2 = image_2 + 0.25
    else:
        raise ValueError(f"Unknown client_id: {client_id}")

    biased_image_1 = tf.clip_by_value(biased_image_1, -1.0, 1.0)
    biased_image_2 = tf.clip_by_value(biased_image_2, -1.0, 1.0)

    return biased_image_1, biased_image_2, label


def create_client_dataset(
    image_paths,
    client_id: str,
    batch_size: int = 16,
    shuffle_buffer_size: int = 128,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Create one client dataset with its scanner-specific bias."""
    if client_id not in CLIENT_IDS:
        raise ValueError(f"Unknown client_id: {client_id}")

    dataset = create_tf_dataset(
        image_paths,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )
    dataset = dataset.map(
        lambda image_1, image_2, label: apply_scanner_bias_to_pair(
            image_1,
            image_2,
            label,
            client_id,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_federated_datasets(
    dataset_dir: str | Path = "data/raw/Br35H/no",
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Create one independent ``tf.data.Dataset`` for each simulated client."""
    image_paths = list_image_paths(dataset_dir)
    client_paths = split_paths_for_clients(
        image_paths,
        num_clients=len(CLIENT_IDS),
        seed=seed,
    )

    federated_datasets = {}

    for client_index, client_id in enumerate(CLIENT_IDS):
        federated_datasets[client_id] = create_client_dataset(
            client_paths[client_id],
            client_id=client_id,
            batch_size=batch_size,
            seed=seed + client_index,
        )

    return federated_datasets


def compute_batch_mean(dataset: tf.data.Dataset) -> float:
    """Compute the global pixel mean from one dataset batch."""
    image_1, image_2, _ = next(iter(dataset))
    mean_image_1 = tf.reduce_mean(image_1)
    mean_image_2 = tf.reduce_mean(image_2)
    global_mean = (mean_image_1 + mean_image_2) / 2.0

    return float(global_mean.numpy())


if __name__ == "__main__":
    federated_datasets = create_federated_datasets()
    client_means = []

    for client_id, dataset in federated_datasets.items():
        images_1, images_2, labels = next(iter(dataset))
        global_pixel_mean = float(
            ((tf.reduce_mean(images_1) + tf.reduce_mean(images_2)) / 2.0).numpy()
        )
        client_means.append(global_pixel_mean)

        print(f"client_id: {client_id}")
        print(f"images_1 shape: {images_1.shape}")
        print(f"images_2 shape: {images_2.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"global pixel mean: {global_pixel_mean}")

        assert images_1.shape == (16, 224, 224, 3)

    assert len(federated_datasets) == 3
    assert len(set(round(mean, 6) for mean in client_means)) > 1
