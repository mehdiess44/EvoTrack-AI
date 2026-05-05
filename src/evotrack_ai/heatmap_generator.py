"""Feature-map heatmap generation for the EvoTrack AI Siamese model."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.siamese_model import absolute_difference, build_siamese_model
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
    from evotrack_ai.tf_data_pipeline import preprocess_image
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from siamese_model import absolute_difference, build_siamese_model
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation
    from tf_data_pipeline import preprocess_image


IMAGE_SIZE = (224, 224)


def load_or_build_model(
    model_path: str | Path = "models/evotrack_siamese_best.keras",
) -> tf.keras.Model:
    """Load a trained Siamese model or build a fresh one if loading fails.

    Args:
        model_path: Path to a saved Keras model.

    Returns:
        A Keras Siamese model.
    """
    path = Path(model_path)

    if path.exists():
        try:
            return tf.keras.models.load_model(
                path,
                custom_objects={"absolute_difference": absolute_difference},
                safe_mode=False,
            )
        except TypeError:
            try:
                return tf.keras.models.load_model(
                    path,
                    custom_objects={"absolute_difference": absolute_difference},
                )
            except Exception as error:
                print(f"Could not load model from {path}. Building a new model.")
                print(f"Reason: {error}")
        except Exception as error:
            print(f"Could not load model from {path}. Building a new model.")
            print(f"Reason: {error}")
    else:
        print(f"Model file not found at {path}. Building a new model.")

    return build_siamese_model()


def get_shared_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the shared feature-extraction backbone from a Siamese model.

    Args:
        model: Siamese Keras model.

    Returns:
        The shared backbone model.

    Raises:
        ValueError: If no suitable backbone can be found.
    """
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue

        layer_name = layer.name.lower()

        if "mobilenet" in layer_name:
            return layer

        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer

        try:
            if len(layer.output.shape) == 4:
                return layer
        except AttributeError:
            continue

    raise ValueError("No shared backbone found in the model.")


def prepare_image_for_model(image: np.ndarray) -> tf.Tensor:
    """Preprocess one grayscale image and add a batch dimension."""
    image_tensor = preprocess_image(tf.convert_to_tensor(image))
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    return image_tensor


def compute_feature_difference_map(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    backbone: tf.keras.Model,
) -> np.ndarray:
    """Compute a spatial feature-difference map between T0 and T1."""
    input_t0 = prepare_image_for_model(img_t0)
    input_t1 = prepare_image_for_model(img_t1)

    features_t0 = backbone(input_t0, training=False)
    features_t1 = backbone(input_t1, training=False)

    feature_difference = tf.abs(features_t1 - features_t0)
    diff_map = tf.reduce_mean(feature_difference, axis=-1)
    diff_map = tf.squeeze(diff_map, axis=0)

    return diff_map.numpy().astype(np.float32)


def normalize_heatmap(diff_map: np.ndarray) -> np.ndarray:
    """Normalize a difference map to the uint8 range [0, 255]."""
    if not isinstance(diff_map, np.ndarray) or diff_map.size == 0:
        raise ValueError("diff_map must be a non-empty NumPy array.")

    min_value = float(np.min(diff_map))
    max_value = float(np.max(diff_map))

    if max_value == min_value:
        return np.zeros(diff_map.shape, dtype=np.uint8)

    normalized = (diff_map - min_value) / (max_value - min_value)
    normalized = normalized * 255.0

    return normalized.astype(np.uint8)


def colorize_heatmap(
    diff_map: np.ndarray,
    output_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Resize and colorize a feature-difference map."""
    normalized_heatmap = normalize_heatmap(diff_map)
    resized_heatmap = cv2.resize(
        normalized_heatmap,
        output_size,
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap_bgr = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    return heatmap_bgr.astype(np.uint8)


def prepare_overlay_base(image: np.ndarray) -> np.ndarray:
    """Resize a grayscale image and convert it to BGR for overlay display."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array.")

    resized_image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    resized_image = resized_image.astype(np.uint8, copy=False)
    base_bgr = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    return base_bgr


def overlay_heatmap(
    base_image: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a BGR heatmap on a BGR base image."""
    if base_image.shape != heatmap_bgr.shape:
        raise ValueError("base_image and heatmap_bgr must have the same shape.")

    overlay = cv2.addWeighted(
        base_image.astype(np.uint8),
        1.0 - alpha,
        heatmap_bgr.astype(np.uint8),
        alpha,
        0,
    )

    return overlay.astype(np.uint8)


def generate_heatmap(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    model: tf.keras.Model,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an overlay and raw color heatmap for a T0/T1 pair."""
    backbone = get_shared_backbone(model)
    diff_map = compute_feature_difference_map(img_t0, img_t1, backbone)
    heatmap_bgr = colorize_heatmap(diff_map)
    base_image = prepare_overlay_base(img_t1)
    overlay = overlay_heatmap(base_image, heatmap_bgr)

    return overlay, heatmap_bgr


def save_debug_outputs(
    original_t0,
    original_t1,
    overlay,
    heatmap,
    output_dir: str | Path = "outputs/heatmaps",
) -> None:
    """Save T0, T1, heatmap, and overlay images for visual inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "t0.png"), prepare_overlay_base(original_t0))
    cv2.imwrite(str(output_path / "t1.png"), prepare_overlay_base(original_t1))
    cv2.imwrite(str(output_path / "heatmap.png"), heatmap)
    cv2.imwrite(str(output_path / "overlay.png"), overlay)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"
    output_dir = project_root / "outputs" / "heatmaps"

    image_paths = list_image_paths(dataset_dir)
    t0 = read_image_grayscale(image_paths[0])
    varied_t1 = simulate_acquisition_variation(t0)
    t1, _ = generate_synthetic_lesion(varied_t1)

    siamese_model = load_or_build_model(project_root / "models" / "evotrack_siamese_best.keras")
    overlay_image, heatmap_image = generate_heatmap(t0, t1, siamese_model)
    save_debug_outputs(t0, t1, overlay_image, heatmap_image, output_dir=output_dir)

    print(f"T0 shape: {t0.shape}")
    print(f"T1 shape: {t1.shape}")
    print(f"heatmap shape: {heatmap_image.shape}")
    print(f"overlay shape: {overlay_image.shape}")
    print(f"heatmap min/max: {heatmap_image.min()} / {heatmap_image.max()}")
    print(f"overlay min/max: {overlay_image.min()} / {overlay_image.max()}")
    print(f"output directory: {output_dir}")

    assert heatmap_image.shape == (224, 224, 3)
    assert overlay_image.shape == (224, 224, 3)
