"""Generate reliable Golden Cases for the EvoTrack AI demo."""

from pathlib import Path

import cv2
import numpy as np


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation


DEFAULT_DATASET_DIR = Path("data/raw/Br35H/no")
DEFAULT_DEMO_DIR = Path("demo_assets")


def ensure_demo_directories(demo_dir: Path = DEFAULT_DEMO_DIR) -> None:
    """Create the demo case directories."""
    for case_name in ("stability", "clear_evolution", "subtle_evolution"):
        (demo_dir / case_name).mkdir(parents=True, exist_ok=True)


def save_pair(
    case_dir: Path,
    t0: np.ndarray,
    t1: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """Save T0, T1, and optionally a ground-truth mask."""
    case_dir.mkdir(parents=True, exist_ok=True)

    files_to_save = {
        "t0.png": t0,
        "t1.png": t1,
    }

    if mask is not None:
        files_to_save["mask.png"] = mask

    for filename, image in files_to_save.items():
        output_path = case_dir / filename
        success = cv2.imwrite(str(output_path), image)

        if not success:
            raise RuntimeError(f"Could not save image: {output_path}")


def create_stability_case(
    base_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a stable T0/T1 case with an empty mask."""
    t0 = base_image
    t1 = base_image.copy()
    mask = np.zeros(base_image.shape, dtype=np.uint8)

    return t0, t1, mask


def create_clear_evolution_case(
    base_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a clearly visible synthetic evolution case."""
    t0 = base_image
    t1, mask = generate_synthetic_lesion(
        base_image,
        min_radius=18,
        max_radius=35,
        intensity_boost=90,
    )

    return t0, t1, mask


def create_subtle_evolution_case(
    base_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a subtle synthetic evolution case with patient movement."""
    t0 = base_image
    t1_base = simulate_acquisition_variation(base_image)
    t1, mask = generate_synthetic_lesion(
        t1_base,
        min_radius=8,
        max_radius=18,
        intensity_boost=40,
    )

    return t0, t1, mask


def generate_demo_assets(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    demo_dir: Path = DEFAULT_DEMO_DIR,
) -> dict:
    """Generate and save the three Golden Cases for the demo."""
    ensure_demo_directories(demo_dir)

    image_paths = list_image_paths(dataset_dir)

    if len(image_paths) < 3:
        raise ValueError("At least three dataset images are required.")

    stability_base = read_image_grayscale(image_paths[0])
    clear_evolution_base = read_image_grayscale(image_paths[1])
    subtle_evolution_base = read_image_grayscale(image_paths[2])

    case_dirs = {
        "stability": demo_dir / "stability",
        "clear_evolution": demo_dir / "clear_evolution",
        "subtle_evolution": demo_dir / "subtle_evolution",
    }

    stability_t0, stability_t1, stability_mask = create_stability_case(
        stability_base,
    )
    save_pair(
        case_dirs["stability"],
        stability_t0,
        stability_t1,
        stability_mask,
    )

    clear_t0, clear_t1, clear_mask = create_clear_evolution_case(
        clear_evolution_base,
    )
    save_pair(case_dirs["clear_evolution"], clear_t0, clear_t1, clear_mask)

    subtle_t0, subtle_t1, subtle_mask = create_subtle_evolution_case(
        subtle_evolution_base,
    )
    save_pair(
        case_dirs["subtle_evolution"],
        subtle_t0,
        subtle_t1,
        subtle_mask,
    )

    return case_dirs


if __name__ == "__main__":
    generated_paths = generate_demo_assets()

    for case_name, case_path in generated_paths.items():
        print(f"{case_name}: {case_path}")

    assert (DEFAULT_DEMO_DIR / "stability" / "t0.png").exists()
    assert (DEFAULT_DEMO_DIR / "stability" / "t1.png").exists()
    assert (DEFAULT_DEMO_DIR / "clear_evolution" / "t0.png").exists()
    assert (DEFAULT_DEMO_DIR / "clear_evolution" / "t1.png").exists()
    assert (DEFAULT_DEMO_DIR / "clear_evolution" / "mask.png").exists()
    assert (DEFAULT_DEMO_DIR / "subtle_evolution" / "t0.png").exists()
    assert (DEFAULT_DEMO_DIR / "subtle_evolution" / "t1.png").exists()
    assert (DEFAULT_DEMO_DIR / "subtle_evolution" / "mask.png").exists()
