"""Utilities for listing and checking Br35H MRI image files."""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_paths(dataset_dir: str | Path) -> list[Path]:
    """Return all supported image paths from a dataset directory.

    The search is recursive and only file paths are stored, so images are not
    loaded into memory at this step.

    Args:
        dataset_dir: Path to the Br35H healthy MRI folder.

    Returns:
        A sorted list of absolute image paths.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
        ValueError: If no supported image file is found.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {dataset_path}")

    image_paths = sorted(
        path.resolve()
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No supported image files found in: {dataset_path}")

    return image_paths


def read_image_grayscale(image_path: str | Path) -> np.ndarray:
    """Read an image as grayscale with OpenCV.

    Args:
        image_path: Path to an image file.

    Returns:
        The grayscale image as a NumPy array.

    Raises:
        ValueError: If the image cannot be read.
    """
    path = Path(image_path).expanduser().resolve()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Image is unreadable or corrupted: {path}")

    return image


def verify_image_integrity(image_paths: Sequence[str | Path]) -> dict:
    """Check which images can be read and summarize their dimensions.

    Args:
        image_paths: Sequence of image paths to verify.

    Returns:
        A report dictionary containing valid paths, corrupted paths, counts,
        and the minimum and maximum image shapes as ``(height, width)``.
    """
    valid_paths: list[Path] = []
    corrupted_paths: list[Path] = []
    shapes: list[tuple[int, int]] = []

    for image_path in image_paths:
        path = Path(image_path).expanduser().resolve()

        try:
            image = read_image_grayscale(path)
        except ValueError:
            corrupted_paths.append(path)
            continue

        valid_paths.append(path)
        shapes.append((int(image.shape[0]), int(image.shape[1])))

    min_shape = min(shapes) if shapes else None
    max_shape = max(shapes) if shapes else None

    return {
        "valid_paths": valid_paths,
        "corrupted_paths": corrupted_paths,
        "num_valid": len(valid_paths),
        "num_corrupted": len(corrupted_paths),
        "min_shape": min_shape,
        "max_shape": max_shape,
    }


def print_dataset_report(report: dict) -> None:
    """Print a clear summary of a dataset integrity report."""
    print("Dataset integrity report")
    print("------------------------")
    print(f"Valid images: {report['num_valid']}")
    print(f"Corrupted images: {report['num_corrupted']}")
    print(f"Minimum dimensions (height, width): {report['min_shape']}")
    print(f"Maximum dimensions (height, width): {report['max_shape']}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    default_dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"

    paths = list_image_paths(default_dataset_dir)
    integrity_report = verify_image_integrity(paths)
    print_dataset_report(integrity_report)
