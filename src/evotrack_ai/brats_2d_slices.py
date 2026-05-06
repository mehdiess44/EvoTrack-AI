from pathlib import Path
import re

import cv2
import numpy as np


DEFAULT_BRATS2D_DIR = Path("data/raw/BraTS2D")
DEFAULT_OUTPUT_DIR = Path("outputs/brats2d_sanity_check")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_sort_key(path: Path):
    """Return a natural sort key so Image-2.png comes before Image-10.png."""
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_brats_2d_images(dataset_dir: str | Path = DEFAULT_BRATS2D_DIR) -> list[Path]:
    """List all supported 2D image slices in a BraTS 2D dataset folder."""
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dataset_path}")

    image_paths = [
        path
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise ValueError(f"Aucune image supportée trouvée dans : {dataset_path}")

    return sorted(image_paths, key=natural_sort_key)


def group_images_by_parent(image_paths: list[Path]) -> dict:
    """Group image paths by their parent folder and keep groups with at least 2 images."""
    groups: dict[Path, list[Path]] = {}

    for image_path in image_paths:
        path = Path(image_path)
        groups.setdefault(path.parent, []).append(path)

    return {
        parent: sorted(paths, key=natural_sort_key)
        for parent, paths in groups.items()
        if len(paths) >= 2
    }


def read_slice_grayscale(image_path: str | Path) -> np.ndarray:
    """Read one image slice as grayscale uint8."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier image introuvable : {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image illisible : {path}")

    return image


def normalize_slice_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize one 2D slice to uint8 values in [0, 255]."""
    if image is None:
        raise ValueError("L'image ne doit pas être None.")

    if not isinstance(image, np.ndarray):
        raise ValueError("L'image doit être un np.ndarray.")

    if image.size == 0:
        raise ValueError("L'image est vide.")

    if image.dtype == np.uint8:
        return image.copy()

    image_float = image.astype(np.float32, copy=False)
    image_float = np.nan_to_num(image_float, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(np.min(image_float))
    max_value = float(np.max(image_float))

    if max_value == min_value:
        return np.zeros(image_float.shape, dtype=np.uint8)

    normalized = (image_float - min_value) / (max_value - min_value) * 255.0
    return normalized.astype(np.uint8)


def resize_slice(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize one grayscale slice and return it as a 2D uint8 image."""
    image_uint8 = normalize_slice_uint8(image)
    resized = cv2.resize(image_uint8, size)
    return resized.astype(np.uint8, copy=False)


def extract_pseudo_longitudinal_pair_from_group(
    image_paths: list[Path],
    z_start: int | None = None,
    gap: int = 3,
    resize_to_224: bool = False,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """Create a pseudo T0/T1 pair from two nearby slices in the same folder."""
    if gap < 1:
        raise ValueError("gap doit être supérieur ou égal à 1.")

    sorted_paths = sorted([Path(path) for path in image_paths], key=natural_sort_key)

    if len(sorted_paths) < gap + 1:
        raise ValueError(
            f"Pas assez d'images pour créer une paire avec gap={gap}."
        )

    if z_start is None:
        z_start = max(0, len(sorted_paths) // 2 - gap)

    t1_index = z_start + gap
    if z_start < 0 or t1_index >= len(sorted_paths):
        raise ValueError(
            f"z_start={z_start} et gap={gap} sont invalides pour "
            f"{len(sorted_paths)} images."
        )

    t0_path = sorted_paths[z_start]
    t1_path = sorted_paths[t1_index]

    t0 = normalize_slice_uint8(read_slice_grayscale(t0_path))
    t1 = normalize_slice_uint8(read_slice_grayscale(t1_path))

    if resize_to_224:
        t0 = resize_slice(t0, size=(224, 224))
        t1 = resize_slice(t1, size=(224, 224))

    return t0, t1, t0_path, t1_path


def find_best_available_pair(
    dataset_dir: str | Path = DEFAULT_BRATS2D_DIR,
    gap: int = 3,
    resize_to_224: bool = False,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    """Find the first folder that can provide a valid pseudo-longitudinal pair."""
    image_paths = list_brats_2d_images(dataset_dir)
    groups = group_images_by_parent(image_paths)

    for parent in sorted(groups, key=natural_sort_key):
        group_paths = groups[parent]
        if len(group_paths) >= gap + 1:
            return extract_pseudo_longitudinal_pair_from_group(
                group_paths,
                gap=gap,
                resize_to_224=resize_to_224,
            )

    raise ValueError(f"Aucune paire possible avec gap={gap}.")


def save_pair_preview(
    t0: np.ndarray,
    t1: np.ndarray,
    t0_path: Path,
    t1_path: Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Save T0/T1 preview images and a small text file with their source paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t0_output = output_path / "brats2d_t0.png"
    t1_output = output_path / "brats2d_t1.png"
    info_output = output_path / "pair_info.txt"

    if not cv2.imwrite(str(t0_output), normalize_slice_uint8(t0)):
        raise RuntimeError(f"Echec de sauvegarde : {t0_output}")

    if not cv2.imwrite(str(t1_output), normalize_slice_uint8(t1)):
        raise RuntimeError(f"Echec de sauvegarde : {t1_output}")

    info_output.write_text(
        f"T0 path: {Path(t0_path)}\nT1 path: {Path(t1_path)}\n",
        encoding="utf-8",
    )


def run_brats2d_sanity_check(
    dataset_dir: str | Path = DEFAULT_BRATS2D_DIR,
    gap: int = 3,
) -> dict:
    """Run a simple sanity check on a BraTS 2D dataset."""
    t0, t1, t0_path, t1_path = find_best_available_pair(dataset_dir, gap=gap)
    save_pair_preview(t0, t1, t0_path, t1_path, output_dir=DEFAULT_OUTPUT_DIR)

    return {
        "t0_shape": tuple(t0.shape),
        "t1_shape": tuple(t1.shape),
        "t0_path": str(t0_path),
        "t1_path": str(t1_path),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }


if __name__ == "__main__":
    try:
        result = run_brats2d_sanity_check()
    except (FileNotFoundError, ValueError):
        print("Aucune image BraTS 2D trouvée. Place le dataset dans data/raw/BraTS2D.")
    else:
        print(f"T0 path: {result['t0_path']}")
        print(f"T1 path: {result['t1_path']}")
        print(f"T0 shape: {result['t0_shape']}")
        print(f"T1 shape: {result['t1_shape']}")
        print(f"output directory: {result['output_dir']}")

        assert len(result["t0_shape"]) == 2
        assert len(result["t1_shape"]) == 2
