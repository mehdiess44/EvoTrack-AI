from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti_volume(filepath: str | Path) -> np.ndarray:
    """Load a 3D NIfTI volume from disk."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    filename = path.name.lower()
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        raise ValueError(
            f"Extension NIfTI invalide pour '{path}'. Attendu : .nii ou .nii.gz."
        )

    try:
        img = nib.load(str(path))
        volume = img.get_fdata()
    except Exception as exc:
        raise ValueError(f"Impossible de charger le fichier NIfTI '{path}' : {exc}") from exc

    return np.asarray(volume)


def normalize_volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Normalize a volume to uint8 values in the range [0, 255]."""
    if not isinstance(volume, np.ndarray):
        raise ValueError("Le volume doit être un np.ndarray.")

    if volume.size == 0:
        raise ValueError("Le volume est vide.")

    volume_float = volume.astype(np.float32, copy=False)
    volume_float = np.nan_to_num(volume_float, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(np.min(volume_float))
    max_value = float(np.max(volume_float))

    if max_value == min_value:
        return np.zeros(volume_float.shape, dtype=np.uint8)

    normalized = (volume_float - min_value) / (max_value - min_value) * 255.0
    return normalized.astype(np.uint8)


def load_and_normalize_nifti(filepath: str | Path) -> np.ndarray:
    """Load a NIfTI volume and return it normalized as uint8."""
    volume = load_nifti_volume(filepath)
    return normalize_volume_to_uint8(volume)


def describe_volume(volume: np.ndarray) -> dict:
    """Return basic statistics about a volume."""
    if not isinstance(volume, np.ndarray):
        raise ValueError("Le volume doit être un np.ndarray.")

    if volume.size == 0:
        raise ValueError("Le volume est vide.")

    return {
        "shape": volume.shape,
        "dtype": volume.dtype,
        "min": float(np.min(volume)),
        "max": float(np.max(volume)),
        "mean": float(np.mean(volume)),
    }


def _find_first_nifti_file(search_dirs: list[Path]) -> Path | None:
    for directory in search_dirs:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            filename = path.name.lower()
            if path.is_file() and (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                return path

    return None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    search_dirs = [
        project_root / "data" / "raw" / "BraTS",
        project_root / "data" / "raw" / "brats",
        project_root / "data" / "raw",
    ]

    nifti_file = _find_first_nifti_file(search_dirs)

    if nifti_file is None:
        print("Aucun fichier NIfTI trouvé. Place un fichier .nii.gz dans data/raw/BraTS.")
    else:
        normalized_volume = load_and_normalize_nifti(nifti_file)
        description = describe_volume(normalized_volume)

        print(f"file path: {nifti_file}")
        print(f"shape: {description['shape']}")
        print(f"dtype: {description['dtype']}")
        print(f"min: {description['min']}")
        print(f"max: {description['max']}")
        print(f"mean: {description['mean']}")

        assert normalized_volume.dtype == np.uint8
        assert np.min(normalized_volume) == 0
        assert np.max(normalized_volume) == 255
