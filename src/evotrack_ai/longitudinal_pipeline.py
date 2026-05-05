"""End-to-end longitudinal MRI analysis pipeline for EvoTrack AI."""

from pathlib import Path

import cv2
import numpy as np


try:
    from evotrack_ai.clinical_summary import (
        deterministic_summary_from_payload,
        generate_clinical_summary,
    )
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.heatmap_generator import (
        compute_feature_difference_map,
        generate_heatmap,
        get_shared_backbone,
        load_or_build_model,
    )
    from evotrack_ai.metrics_extraction import extract_heatmap_metrics
    from evotrack_ai.nlp_payload import build_payload_from_metrics
    from evotrack_ai.registration import align_images
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
except ModuleNotFoundError:
    from clinical_summary import (
        deterministic_summary_from_payload,
        generate_clinical_summary,
    )
    from data_ingestion import list_image_paths, read_image_grayscale
    from heatmap_generator import (
        compute_feature_difference_map,
        generate_heatmap,
        get_shared_backbone,
        load_or_build_model,
    )
    from metrics_extraction import extract_heatmap_metrics
    from nlp_payload import build_payload_from_metrics
    from registration import align_images
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation


def validate_input_image(image, name: str = "image") -> None:
    """Validate a grayscale image for longitudinal analysis.

    Args:
        image: Image to validate.
        name: Name used in error messages.

    Raises:
        ValueError: If the image is invalid.
    """
    if image is None:
        raise ValueError(f"{name} must not be None.")

    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")

    if image.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if image.ndim != 2:
        raise ValueError(f"{name} must be a grayscale 2D image.")


def analyze_longitudinal_scan(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    model=None,
    generator=None,
    use_registration: bool = True,
    use_nlp_fallback: bool = True,
) -> tuple[np.ndarray, str]:
    """Analyze a longitudinal pair and return an overlay plus summary.

    Args:
        img_t0: Baseline grayscale image.
        img_t1: Follow-up grayscale image.
        model: Optional Siamese model.
        generator: Optional Hugging Face text generator.
        use_registration: Whether to align T1 onto T0 before analysis.
        use_nlp_fallback: Whether to use deterministic summary if NLP fails.

    Returns:
        ``overlay, summary``.
    """
    validate_input_image(img_t0, name="img_t0")
    validate_input_image(img_t1, name="img_t1")

    if model is None:
        model = load_or_build_model()

    aligned_t1 = img_t1

    if use_registration:
        try:
            aligned_t1, _ = align_images(img_t0, img_t1)
        except ValueError as error:
            print(
                "Warning: registration failed. "
                "Using original T1 image as fallback."
            )
            print(f"Reason: {error}")

    overlay, _ = generate_heatmap(img_t0, aligned_t1, model)

    backbone = get_shared_backbone(model)
    diff_map = compute_feature_difference_map(img_t0, aligned_t1, backbone)
    metrics = extract_heatmap_metrics(diff_map)
    payload = build_payload_from_metrics(metrics)

    try:
        summary = generate_clinical_summary(payload, generator=generator)
    except Exception:
        if use_nlp_fallback:
            summary = deterministic_summary_from_payload(payload)
        else:
            raise

    return overlay, summary


def analyze_longitudinal_scan_from_paths(
    t0_path: str | Path,
    t1_path: str | Path,
    **kwargs,
) -> tuple[np.ndarray, str]:
    """Read two grayscale images from paths and analyze them."""
    img_t0 = read_image_grayscale(t0_path)
    img_t1 = read_image_grayscale(t1_path)

    return analyze_longitudinal_scan(img_t0, img_t1, **kwargs)


def save_pipeline_outputs(
    overlay: np.ndarray,
    summary: str,
    output_dir: str | Path = "outputs/final_pipeline",
) -> None:
    """Save pipeline overlay and summary text to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "overlay.png"), overlay)

    with open(output_path / "summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"
    output_dir = project_root / "outputs" / "final_pipeline"

    image_paths = list_image_paths(dataset_dir)
    img_t0 = read_image_grayscale(image_paths[0])
    varied_t1 = simulate_acquisition_variation(img_t0)
    img_t1, _ = generate_synthetic_lesion(varied_t1)

    siamese_model = load_or_build_model(
        project_root / "models" / "evotrack_siamese_best.keras"
    )
    overlay_image, clinical_summary = analyze_longitudinal_scan(
        img_t0,
        img_t1,
        model=siamese_model,
    )
    save_pipeline_outputs(
        overlay_image,
        clinical_summary,
        output_dir=output_dir,
    )

    print(f"overlay shape: {overlay_image.shape}")
    print(f"overlay dtype: {overlay_image.dtype}")
    print(f"summary: {clinical_summary}")
    print(f"output directory: {output_dir}")

    assert isinstance(overlay_image, np.ndarray)
    assert isinstance(clinical_summary, str)
    assert len(clinical_summary) > 15
    assert overlay_image.shape == (224, 224, 3)
