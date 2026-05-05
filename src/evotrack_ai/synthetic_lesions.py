"""Synthetic lesion generation utilities for grayscale brain MRI images."""

from pathlib import Path

import cv2
import numpy as np


def _validate_grayscale_image(image: np.ndarray) -> None:
    """Raise an error if the image is empty or not grayscale."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("Image is empty.")

    if image.ndim != 2:
        raise ValueError("Image must be a grayscale image.")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image has invalid height or width.")


def create_brain_mask(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Create a binary mask of brain tissue from a grayscale MRI image.

    Args:
        image: Grayscale MRI image.
        threshold: Pixel intensity threshold used to separate brain from
            black background.

    Returns:
        A ``uint8`` binary mask containing only 0 and 255.

    Raises:
        ValueError: If the image is empty or invalid.
    """
    _validate_grayscale_image(image)

    _, brain_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    brain_mask = brain_mask.astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    return np.where(brain_mask > 0, 255, 0).astype(np.uint8)


def sample_lesion_center(brain_mask: np.ndarray, margin: int = 25) -> tuple[int, int]:
    """Sample a random lesion center inside a binary brain mask.

    Args:
        brain_mask: Binary brain mask with brain pixels marked as 255.
        margin: Number of pixels to avoid around image borders.

    Returns:
        A random center point as ``(x, y)``.

    Raises:
        ValueError: If no valid brain pixel is available.
    """
    _validate_grayscale_image(brain_mask)

    if margin < 0:
        raise ValueError("margin must be non-negative.")

    height, width = brain_mask.shape
    valid_mask = brain_mask > 0

    if margin > 0:
        border_mask = np.zeros_like(valid_mask, dtype=bool)
        border_mask[margin : height - margin, margin : width - margin] = True
        valid_mask = valid_mask & border_mask

    valid_pixels = np.argwhere(valid_mask)

    if valid_pixels.size == 0:
        raise ValueError("No valid pixel available for lesion center sampling.")

    random_index = np.random.randint(0, len(valid_pixels))
    y, x = valid_pixels[random_index]

    return int(x), int(y)


def generate_synthetic_lesion(
    image: np.ndarray,
    min_radius: int = 8,
    max_radius: int = 25,
    intensity_boost: int = 60,
    blur_kernel_size: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one bright synthetic lesion and its ground-truth mask.

    Args:
        image: Healthy grayscale MRI image.
        min_radius: Minimum ellipse radius in pixels.
        max_radius: Maximum ellipse radius in pixels.
        intensity_boost: Local pixel intensity increase inside the lesion.
        blur_kernel_size: Gaussian blur kernel size used to soften lesion edges.

    Returns:
        A tuple ``(infected_image, ground_truth_mask)``.

    Raises:
        ValueError: If the image or parameters are invalid.
    """
    _validate_grayscale_image(image)

    if min_radius <= 0:
        raise ValueError("min_radius must be positive.")

    if max_radius < min_radius:
        raise ValueError("max_radius must be greater than or equal to min_radius.")

    if blur_kernel_size <= 0:
        raise ValueError("blur_kernel_size must be positive.")

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    brain_mask = create_brain_mask(image)
    center_x, center_y = sample_lesion_center(brain_mask, margin=max_radius)

    radius_x = np.random.randint(min_radius, max_radius + 1)
    radius_y = np.random.randint(min_radius, max_radius + 1)
    angle = np.random.uniform(0, 180)

    ground_truth_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.ellipse(
        ground_truth_mask,
        (center_x, center_y),
        (radius_x, radius_y),
        angle,
        0,
        360,
        255,
        thickness=-1,
    )

    ground_truth_mask = cv2.bitwise_and(ground_truth_mask, brain_mask)
    ground_truth_mask = np.where(ground_truth_mask > 0, 255, 0).astype(np.uint8)

    soft_lesion_mask = cv2.GaussianBlur(
        ground_truth_mask,
        (blur_kernel_size, blur_kernel_size),
        sigmaX=0,
    )
    boost_map = (soft_lesion_mask.astype(np.float32) / 255.0) * intensity_boost

    infected_float = image.astype(np.float32) + boost_map
    infected_image = np.clip(infected_float, 0, 255).astype(np.uint8)

    return infected_image, ground_truth_mask


def validate_binary_mask(mask: np.ndarray) -> bool:
    """Return True when a mask contains only binary values 0 and 255."""
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return False

    unique_values = np.unique(mask)
    return set(unique_values.tolist()).issubset({0, 255})


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
    infected_image, ground_truth_mask = generate_synthetic_lesion(original_image)

    mask_unique_values = np.unique(ground_truth_mask)
    lesion_pixel_count = cv2.countNonZero(ground_truth_mask)

    print(f"original shape: {original_image.shape}")
    print(f"infected shape: {infected_image.shape}")
    print(f"mask shape: {ground_truth_mask.shape}")
    print(f"original min/max: {original_image.min()} / {original_image.max()}")
    print(f"infected min/max: {infected_image.min()} / {infected_image.max()}")
    print(f"mask unique values: {mask_unique_values}")
    print(f"mask is binary: {validate_binary_mask(ground_truth_mask)}")
    print(f"lesion pixel count: {lesion_pixel_count}")

    assert np.array_equal(mask_unique_values, np.array([0, 255], dtype=np.uint8))
    assert lesion_pixel_count > 0
