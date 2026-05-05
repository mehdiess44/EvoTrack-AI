"""Extract simple semantic metrics from heatmaps or difference maps."""

import cv2
import numpy as np


IMAGE_SIZE = (224, 224)


def validate_diff_map(diff_map: np.ndarray) -> None:
    """Validate that a difference map can be processed.

    Args:
        diff_map: Difference map as a 2D grayscale image or a 3-channel image.

    Raises:
        ValueError: If the map is missing, empty, or has an unsupported shape.
    """
    if diff_map is None:
        raise ValueError("diff_map must not be None.")

    if not isinstance(diff_map, np.ndarray):
        raise ValueError("diff_map must be a NumPy array.")

    if diff_map.size == 0:
        raise ValueError("diff_map must not be empty.")

    if diff_map.ndim == 2:
        return

    if diff_map.ndim == 3 and diff_map.shape[2] == 3:
        return

    raise ValueError("diff_map must be a 2D grayscale map or a 3-channel image.")


def ensure_uint8_map(diff_map: np.ndarray) -> np.ndarray:
    """Convert a difference map to a uint8 grayscale map.

    Float maps are normalized to the range [0, 255]. Constant maps are converted
    to zeros.
    """
    validate_diff_map(diff_map)

    if diff_map.ndim == 3:
        diff_map = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)

    if diff_map.dtype == np.uint8:
        return diff_map.copy()

    min_value = float(np.min(diff_map))
    max_value = float(np.max(diff_map))

    if max_value == min_value:
        return np.zeros(diff_map.shape, dtype=np.uint8)

    normalized = (diff_map.astype(np.float32) - min_value) / (max_value - min_value)
    normalized = normalized * 255.0

    return normalized.astype(np.uint8)


def binarize_diff_map(diff_map: np.ndarray) -> np.ndarray:
    """Create a binary mask from a difference map using Otsu thresholding."""
    validate_diff_map(diff_map)
    diff_map_uint8 = ensure_uint8_map(diff_map)

    _, mask = cv2.threshold(
        diff_map_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def compute_surface(mask: np.ndarray) -> int:
    """Return the number of non-zero pixels in a binary mask."""
    return int(cv2.countNonZero(mask))


def classify_surface(surface: int) -> str:
    """Classify a changed surface area from its pixel count."""
    if surface == 0:
        return "absente"

    if surface < 200:
        return "microscopique"

    if surface < 1500:
        return "modérée"

    return "massive"


def compute_signal_intensity(diff_map: np.ndarray, mask: np.ndarray) -> float:
    """Compute the mean difference intensity inside a binary mask."""
    diff_map_uint8 = ensure_uint8_map(diff_map)
    surface = compute_surface(mask)

    if surface == 0:
        return 0.0

    return float(cv2.mean(diff_map_uint8, mask=mask)[0])


def classify_intensity(intensity: float) -> str:
    """Classify a mean signal intensity."""
    if intensity == 0:
        return "nulle"

    if intensity < 60:
        return "subtile"

    if intensity < 150:
        return "franche"

    return "haute intensité"


def compute_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Compute the centroid of the largest contour in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)

    if moments["m00"] == 0:
        return None

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    return centroid_x, centroid_y


def classify_location(
    centroid: tuple[int, int] | None,
    image_shape: tuple[int, int],
) -> str:
    """Classify a centroid location into one of four image quadrants."""
    if centroid is None:
        return "aucune localisation significative"

    x, y = centroid
    height, width = image_shape[:2]

    if x < width / 2 and y < height / 2:
        return "quadrant supérieur gauche"

    if x >= width / 2 and y < height / 2:
        return "quadrant supérieur droit"

    if x < width / 2 and y >= height / 2:
        return "quadrant inférieur gauche"

    return "quadrant inférieur droit"


def extract_heatmap_metrics(diff_map: np.ndarray) -> dict:
    """Extract status, surface, intensity, centroid, and location metrics."""
    mask = binarize_diff_map(diff_map)
    surface_pixels = compute_surface(mask)
    surface_category = classify_surface(surface_pixels)
    intensity_mean = compute_signal_intensity(diff_map, mask)
    intensity_category = classify_intensity(intensity_mean)
    centroid = compute_centroid(mask)
    location = classify_location(centroid, mask.shape)
    status = "stabilité" if surface_pixels == 0 else "évolution"

    return {
        "status": status,
        "surface_pixels": surface_pixels,
        "surface_category": surface_category,
        "intensity_mean": intensity_mean,
        "intensity_category": intensity_category,
        "centroid": centroid,
        "location": location,
    }


if __name__ == "__main__":
    test_diff_map = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    test_diff_map[20:30, 170:180] = 255

    metrics = extract_heatmap_metrics(test_diff_map)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    assert metrics["surface_pixels"] == 100
    assert metrics["location"] == "quadrant supérieur droit"
    assert metrics["status"] == "évolution"
