"""Balanced Siamese pair generator for synthetic MRI evolution samples."""

from pathlib import Path
from typing import Sequence

import numpy as np


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation


class SiameseDataGenerator:
    """Generate balanced Siamese training triplets from image paths.

    The generator stores only file paths. Images are loaded one by one when a
    pair is created.
    """

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        seed: int | None = None,
    ) -> None:
        """Initialize the generator with image paths and an optional seed.

        Args:
            image_paths: Paths to healthy T0 images.
            seed: Optional random seed for image path sampling.

        Raises:
            ValueError: If no image path is provided.
        """
        if not image_paths:
            raise ValueError("image_paths must not be empty.")

        self.image_paths = [Path(image_path) for image_path in image_paths]
        self.rng = np.random.default_rng(seed)

    def _sample_image_path(self) -> Path:
        """Return one randomly selected image path."""
        random_index = self.rng.integers(0, len(self.image_paths))
        return self.image_paths[int(random_index)]

    def create_class_0_pair(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Create a similar pair: T0 and T1 with movement and sensor noise."""
        image_path = self._sample_image_path()
        t0_image = read_image_grayscale(image_path).astype(np.uint8, copy=False)
        t1_image = simulate_acquisition_variation(t0_image).astype(
            np.uint8,
            copy=False,
        )

        return t0_image, t1_image, 0

    def create_class_1_pair(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Create an evolved pair: T0 and T1 with movement, noise, and lesion."""
        image_path = self._sample_image_path()
        t0_image = read_image_grayscale(image_path).astype(np.uint8, copy=False)
        altered_image = simulate_acquisition_variation(t0_image).astype(
            np.uint8,
            copy=False,
        )
        t1_infected, _ = generate_synthetic_lesion(altered_image)
        t1_infected = t1_infected.astype(np.uint8, copy=False)

        return t0_image, t1_infected, 1

    def pair_generator(self):
        """Yield an infinite sequence of balanced ``(T0, T1, label)`` triplets."""
        next_label = 0

        while True:
            if next_label == 0:
                yield self.create_class_0_pair()
                next_label = 1
            else:
                yield self.create_class_1_pair()
                next_label = 0


def count_labels_from_generator(generator, num_samples: int = 100) -> dict:
    """Count class labels from a Python generator.

    Args:
        generator: Generator yielding ``(T0, T1, label)`` triplets.
        num_samples: Number of generated samples to inspect.

    Returns:
        A dictionary with counts for class 0, class 1, and total samples.
    """
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")

    class_0_count = 0
    class_1_count = 0

    for _ in range(num_samples):
        _, _, label = next(generator)

        if label == 0:
            class_0_count += 1
        elif label == 1:
            class_1_count += 1

    return {
        "class_0": class_0_count,
        "class_1": class_1_count,
        "total": class_0_count + class_1_count,
    }


def _absolute_difference_sum(first_image: np.ndarray, second_image: np.ndarray) -> int:
    """Return the sum of absolute pixel differences between two images."""
    return int(
        np.abs(
            first_image.astype(np.int16) - second_image.astype(np.int16)
        ).sum()
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"

    paths = list_image_paths(dataset_dir)
    data_generator = SiameseDataGenerator(paths)
    generator = data_generator.pair_generator()

    label_counts = count_labels_from_generator(generator, num_samples=100)

    print(f"total samples: {label_counts['total']}")
    print(f"class 0 count: {label_counts['class_0']}")
    print(f"class 1 count: {label_counts['class_1']}")

    t0_class_0, t1_class_0, label_class_0 = data_generator.create_class_0_pair()
    difference_sum_class_0 = _absolute_difference_sum(t0_class_0, t1_class_0)

    print("class 0 example")
    print(f"T0 shape: {t0_class_0.shape}")
    print(f"T1 shape: {t1_class_0.shape}")
    print(f"label: {label_class_0}")
    print(f"difference sum: {difference_sum_class_0}")

    t0_class_1, t1_class_1, label_class_1 = data_generator.create_class_1_pair()
    difference_sum_class_1 = _absolute_difference_sum(t0_class_1, t1_class_1)

    print("class 1 example")
    print(f"T0 shape: {t0_class_1.shape}")
    print(f"T1 shape: {t1_class_1.shape}")
    print(f"label: {label_class_1}")
    print(f"difference sum: {difference_sum_class_1}")

    assert label_counts["class_0"] == 50
    assert label_counts["class_1"] == 50
