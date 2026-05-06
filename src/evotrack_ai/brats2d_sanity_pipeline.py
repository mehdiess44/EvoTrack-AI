"""BraTS 2D sanity pipeline for EvoTrack AI."""

from pathlib import Path

import cv2
import numpy as np


try:
    from evotrack_ai.brats_2d_slices import find_best_available_pair
    from evotrack_ai.heatmap_generator import load_or_build_model
    from evotrack_ai.longitudinal_pipeline import analyze_longitudinal_scan
except ModuleNotFoundError as error:
    if error.name != "evotrack_ai":
        raise

    from brats_2d_slices import find_best_available_pair
    from heatmap_generator import load_or_build_model
    from longitudinal_pipeline import analyze_longitudinal_scan


DEFAULT_BRATS2D_DIR = Path("data/raw/BraTS2D")
DEFAULT_OUTPUT_DIR = Path("outputs/brats2d_sanity_pipeline")
DEFAULT_MODEL_PATH = Path("models/evotrack_siamese_best.keras")


def ensure_output_dir(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Create the output directory if needed and return it as a Path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_sanity_outputs(
    t0: np.ndarray,
    t1: np.ndarray,
    overlay: np.ndarray,
    summary: str,
    t0_path: Path,
    t1_path: Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Save input slices, overlay, summary, and source path metadata."""
    output_path = ensure_output_dir(output_dir)

    t0_output = output_path / "brats2d_t0.png"
    t1_output = output_path / "brats2d_t1.png"
    overlay_output = output_path / "brats2d_overlay.png"

    if not cv2.imwrite(str(t0_output), t0):
        raise RuntimeError(f"Echec de sauvegarde : {t0_output}")

    if not cv2.imwrite(str(t1_output), t1):
        raise RuntimeError(f"Echec de sauvegarde : {t1_output}")

    if not cv2.imwrite(str(overlay_output), overlay):
        raise RuntimeError(f"Echec de sauvegarde : {overlay_output}")

    (output_path / "summary.txt").write_text(summary, encoding="utf-8")
    (output_path / "pair_info.txt").write_text(
        f"T0 source path: {Path(t0_path)}\n"
        f"T1 source path: {Path(t1_path)}\n",
        encoding="utf-8",
    )


def run_brats2d_sanity_pipeline(
    dataset_dir: str | Path = DEFAULT_BRATS2D_DIR,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    gap: int = 3,
    use_registration: bool = True,
) -> dict:
    """Run the EvoTrack longitudinal pipeline on one BraTS 2D pseudo-pair."""
    t0, t1, t0_path, t1_path = find_best_available_pair(
        dataset_dir,
        gap=gap,
        resize_to_224=False,
    )
    model = load_or_build_model(model_path)
    overlay, summary = analyze_longitudinal_scan(
        t0,
        t1,
        model=model,
        use_registration=use_registration,
    )

    save_sanity_outputs(
        t0,
        t1,
        overlay,
        summary,
        t0_path,
        t1_path,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    return {
        "t0_path": str(t0_path),
        "t1_path": str(t1_path),
        "t0_shape": tuple(t0.shape),
        "t1_shape": tuple(t1.shape),
        "overlay_shape": tuple(overlay.shape),
        "summary": summary,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }


if __name__ == "__main__":
    try:
        result = run_brats2d_sanity_pipeline()
    except (FileNotFoundError, ValueError):
        print("Aucune image BraTS 2D trouvée. Place le dataset dans data/raw/BraTS2D.")
    else:
        print(f"T0 path: {result['t0_path']}")
        print(f"T1 path: {result['t1_path']}")
        print(f"T0 shape: {result['t0_shape']}")
        print(f"T1 shape: {result['t1_shape']}")
        print(f"overlay shape: {result['overlay_shape']}")
        print(f"summary: {result['summary']}")
        print(f"output directory: {result['output_dir']}")

        assert result["overlay_shape"] == (224, 224, 3)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 15
