"""Synthetic image transforms for patient movement and sensor noise."""

from pathlib import Path

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Raise an error if an image is empty or has an unsupported shape."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    if image.ndim not in (2, 3):
        raise ValueError("Image must be grayscale or RGB.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image has invalid height or width.")


def simulate_patient_movement(
    image: np.ndarray,
    max_rotation_degrees: float = 3.0,
    max_translation_pixels: int = 5,
) -> np.ndarray:
    """Apply small random rotation and translation to simulate patient motion.

    Args:
        image: OpenCV image in grayscale or RGB format.
        max_rotation_degrees: Maximum absolute rotation angle in degrees.
        max_translation_pixels: Maximum absolute translation in pixels.

    Returns:
        The transformed image with the same shape and dtype as the input image.

    Raises:
        ValueError: If the image is empty or invalid.
    """
    _validate_image(image)

    if max_rotation_degrees < 0:
        raise ValueError("max_rotation_degrees must be non-negative.")

    if max_translation_pixels < 0:
        raise ValueError("max_translation_pixels must be non-negative.")

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    angle = np.random.uniform(-max_rotation_degrees, max_rotation_degrees)
    translation_x = np.random.uniform(-max_translation_pixels, max_translation_pixels)
    translation_y = np.random.uniform(-max_translation_pixels, max_translation_pixels)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotation_matrix[0, 2] += translation_x
    rotation_matrix[1, 2] += translation_y

    transformed_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return transformed_image.astype(image.dtype, copy=False)


def inject_sensor_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 5.0,
) -> np.ndarray:
    """Add Gaussian sensor noise to an image.

    Args:
        image: OpenCV image in grayscale or RGB format.
        mean: Mean of the Gaussian noise.
        std: Standard deviation of the Gaussian noise.

    Returns:
        A noisy image with dtype ``np.uint8`` and pixel values in [0, 255].

    Raises:
        ValueError: If the image is empty or invalid.
    """
    _validate_image(image)

    if std < 0:
        raise ValueError("std must be non-negative.")

    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    clipped_image = np.clip(noisy_image, 0, 255)

    return clipped_image.astype(np.uint8)


def simulate_acquisition_variation(image: np.ndarray) -> np.ndarray:
    """Simulate one acquisition variation with movement followed by noise."""
    moved_image = simulate_patient_movement(image)
    transformed_image = inject_sensor_noise(moved_image)

    return transformed_image


if __name__ == "__main__":
    try:
        from evotrack_ai.data_ingestion import (
            list_image_paths,
            read_image_grayscale,
        )
    except ModuleNotFoundError:
        from data_ingestion import list_image_paths, read_image_grayscale

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"

    image_paths = list_image_paths(dataset_dir)
    original_image = read_image_grayscale(image_paths[0])
    transformed_image = simulate_acquisition_variation(original_image)

    absolute_difference_sum = int(
        np.abs(
            original_image.astype(np.int16) - transformed_image.astype(np.int16)
        ).sum()
    )

    print(f"original shape: {original_image.shape}")
    print(f"transformed shape: {transformed_image.shape}")
    print(f"original min/max: {original_image.min()} / {original_image.max()}")
    print(f"transformed min/max: {transformed_image.min()} / {transformed_image.max()}")
    print(f"absolute difference sum: {absolute_difference_sum}")

    assert absolute_difference_sum > 0
    assert transformed_image.min() >= 0
    assert transformed_image.max() <= 255
