"""Auto-curation SSIM — Pré-annotation automatique pour EvoTrack AI.

Génère des labels binaires (Y=0 stable, Y=1 évolution) pour des paires
longitudinales IRM en calculant le SSIM sur la ROI péri-tumorale définie
par la dilatation morphologique du Masque T0.

Entrées attendues :
    images_T0/  — PNG 224×224 (baseline)
    images_T1/  — PNG 224×224 (suivi)
    masks_T0/   — PNG 224×224 binaires (segmentation T0)

Sortie :
    auto_labels.csv — Patient_ID, Score_SSIM, Label_Auto
"""

import os
import sys

# Forcer la sortie console en UTF-8 (Windows utilise cp1252 par défaut).
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGES_T0_DIR = r"C:\Users\mehdi\EvoTrack AI\EvoTrack_Finetuning_Dataset\images_T0"
IMAGES_T1_DIR = r"C:\Users\mehdi\EvoTrack AI\EvoTrack_Finetuning_Dataset\images_T1"
MASKS_T0_DIR = r"C:\Users\mehdi\EvoTrack AI\EvoTrack_Finetuning_Dataset\masks_T0"
OUTPUT_CSV = "auto_labels.csv"

DILATION_KERNEL_SIZE = (15, 15)
DILATION_ITERATIONS = 1

SSIM_THRESHOLD_STABLE = 0.90
SSIM_THRESHOLD_EVOLUTION = 0.82

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Module 1 — Chargement & Dilatation
# ---------------------------------------------------------------------------


def load_grayscale(filepath: str) -> np.ndarray:
    """Charge une image en niveaux de gris uint8.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si l'image est illisible ou vide.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if image is None or image.size == 0:
        raise ValueError(f"Image illisible ou vide : {filepath}")

    return image


def load_mask_binary(filepath: str) -> np.ndarray:
    """Charge un masque et le binarise (0 / 255).

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le masque est illisible ou entièrement noir.
    """
    raw = load_grayscale(filepath)
    _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary) == 0:
        raise ValueError(f"Masque entièrement noir (aucune tumeur) : {filepath}")

    return binary


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    """Dilate le masque binaire avec un kernel elliptique 15×15."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE)
    dilated = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)
    return dilated


# ---------------------------------------------------------------------------
# Module 2 — Extraction ROI (Bounding Box)
# ---------------------------------------------------------------------------


def bounding_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Retourne (y_min, y_max, x_min, x_max) de la bounding box du masque.

    Raises:
        ValueError: si aucun pixel non nul n'est trouvé.
    """
    coords = cv2.findNonZero(mask)

    if coords is None:
        raise ValueError("Masque vide — impossible d'extraire une bounding box.")

    x, y, w, h = cv2.boundingRect(coords)

    return y, y + h, x, x + w


def crop_to_roi(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Découpe l'image selon la bounding box (y_min, y_max, x_min, x_max).

    Raises:
        ValueError: si le crop résultant est vide.
    """
    y_min, y_max, x_min, x_max = bbox
    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        raise ValueError(
            f"Crop vide pour bbox ({y_min}, {y_max}, {x_min}, {x_max})."
        )

    return crop


# ---------------------------------------------------------------------------
# Module 3 — Calcul SSIM
# ---------------------------------------------------------------------------


def compute_roi_ssim(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    dilated_mask: np.ndarray,
) -> float:
    """Calcule le SSIM entre les crops T0 et T1 sur la ROI dilatée.

    Returns:
        Score SSIM (float dans [-1, 1], typiquement [0, 1]).
    """
    bbox = bounding_box_from_mask(dilated_mask)

    crop_t0 = crop_to_roi(img_t0, bbox)
    crop_t1 = crop_to_roi(img_t1, bbox)

    # Garantir des dimensions identiques (sécurité si T1 a une taille
    # légèrement différente après recalage).
    if crop_t0.shape != crop_t1.shape:
        crop_t1 = cv2.resize(
            crop_t1,
            (crop_t0.shape[1], crop_t0.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # win_size doit être impair et <= min(dimension du crop).
    min_dim = min(crop_t0.shape[0], crop_t0.shape[1])
    win_size = min(7, min_dim)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    score = structural_similarity(
        crop_t0,
        crop_t1,
        win_size=win_size,
        data_range=255,
    )

    return float(score)


# ---------------------------------------------------------------------------
# Module 4 — Heuristique de Labellisation
# ---------------------------------------------------------------------------


def assign_label(ssim_score: float) -> str:
    """Applique les seuils de labellisation.

    Returns:
        "0" (stable), "1" (évolution), ou "REVIEW" (zone grise).
    """
    if ssim_score >= SSIM_THRESHOLD_STABLE:
        return "0"
    if ssim_score <= SSIM_THRESHOLD_EVOLUTION:
        return "1"
    return "REVIEW"


# ---------------------------------------------------------------------------
# Module 5 — Orchestration & Export CSV
# ---------------------------------------------------------------------------


def discover_patients() -> list[str]:
    """Identifie les patients par intersection des fichiers dans les 3 dossiers.

    Returns:
        Liste triée des noms de fichiers (sans extension) présents dans
        images_T0/, images_T1/ ET masks_T0/.

    Raises:
        FileNotFoundError: si un des répertoires n'existe pas.
        ValueError: si aucun patient commun n'est trouvé.
    """
    for dir_path, label in [
        (IMAGES_T0_DIR, "images_T0"),
        (IMAGES_T1_DIR, "images_T1"),
        (MASKS_T0_DIR, "masks_T0"),
    ]:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(
                f"Répertoire manquant : {dir_path} ({label})"
            )

    def _strip_timepoint(stem: str) -> str:
        """Supprime le suffixe _T0 ou _T1 d'un stem pour obtenir l'ID patient."""
        for suffix in ("_T0", "_T1", "_t0", "_t1"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def patient_ids(directory: str) -> set[str]:
        """Retourne l'ensemble des IDs patients (stem sans _T0/_T1)."""
        ids = set()
        for fname in os.listdir(directory):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                stem = os.path.splitext(fname)[0]
                ids.add(_strip_timepoint(stem))
        return ids

    ids_t0 = patient_ids(IMAGES_T0_DIR)
    ids_t1 = patient_ids(IMAGES_T1_DIR)
    ids_mask = patient_ids(MASKS_T0_DIR)

    common_ids = sorted(ids_t0 & ids_t1 & ids_mask)

    if not common_ids:
        raise ValueError(
            "Aucun patient commun trouvé dans images_T0/, images_T1/ et masks_T0/. "
            "Vérifiez que les noms de fichiers (sans extension) correspondent."
        )

    return common_ids


def resolve_filepath(directory: str, patient_id: str) -> str:
    """Trouve le chemin complet d'un fichier à partir de l'ID patient.

    Cherche d'abord une correspondance exacte, puis avec les suffixes
    _T0 et _T1 courants dans le dataset.
    """
    candidates = [patient_id, f"{patient_id}_T0", f"{patient_id}_T1"]
    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            stem = os.path.splitext(fname)[0]
            if stem in candidates:
                return os.path.join(directory, fname)
    raise FileNotFoundError(
        f"Fichier introuvable pour patient '{patient_id}' dans {directory}"
    )


def run_auto_curation() -> pd.DataFrame:
    """Pipeline complet d'auto-curation SSIM.

    Returns:
        DataFrame contenant Patient_ID, Score_SSIM, Label_Auto.
    """
    patients = discover_patients()

    print(f"\n{'=' * 60}")
    print(f"  EvoTrack AI — Auto-Curation SSIM")
    print(f"  Patients détectés : {len(patients)}")
    print(f"  Seuil stable  (Y=0) : SSIM >= {SSIM_THRESHOLD_STABLE}")
    print(f"  Seuil évol.   (Y=1) : SSIM <= {SSIM_THRESHOLD_EVOLUTION}")
    print(f"  Zone grise (REVIEW) : {SSIM_THRESHOLD_EVOLUTION} < SSIM < {SSIM_THRESHOLD_STABLE}")
    print(f"  Kernel dilatation   : {DILATION_KERNEL_SIZE} (elliptique)")
    print(f"{'=' * 60}\n")

    results = []
    errors = []

    for patient_id in tqdm(patients, desc="Auto-labelling", unit="patient"):
        try:
            path_t0 = resolve_filepath(IMAGES_T0_DIR, patient_id)
            path_t1 = resolve_filepath(IMAGES_T1_DIR, patient_id)
            path_mask = resolve_filepath(MASKS_T0_DIR, patient_id)

            img_t0 = load_grayscale(path_t0)
            img_t1 = load_grayscale(path_t1)
            mask_t0 = load_mask_binary(path_mask)

            dilated = dilate_mask(mask_t0)

            ssim_score = compute_roi_ssim(img_t0, img_t1, dilated)

            label = assign_label(ssim_score)

            results.append(
                {
                    "Patient_ID": patient_id,
                    "Score_SSIM": round(ssim_score, 6),
                    "Label_Auto": label,
                }
            )

        except (FileNotFoundError, ValueError, cv2.error) as exc:
            errors.append({"Patient_ID": patient_id, "Error": str(exc)})
            tqdm.write(f"  [ERREUR] {patient_id} — {exc}")

    # ---- Construction du DataFrame ----
    df = pd.DataFrame(results, columns=["Patient_ID", "Score_SSIM", "Label_Auto"])
    df.sort_values("Score_SSIM", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- Export CSV ----
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ---- Rapport console ----
    count_stable = (df["Label_Auto"] == "0").sum()
    count_evolution = (df["Label_Auto"] == "1").sum()
    count_review = (df["Label_Auto"] == "REVIEW").sum()

    print(f"\n{'=' * 60}")
    print(f"  RÉSULTATS AUTO-CURATION")
    print(f"{'=' * 60}")
    print(f"  Patients traités  : {len(results)}")
    print(f"  Erreurs           : {len(errors)}")
    print(f"  ─────────────────────────────────")
    print(f"  Label 0 (Stable)  : {count_stable}")
    print(f"  Label 1 (Évol.)   : {count_evolution}")
    print(f"  REVIEW (ambigu)   : {count_review}")
    print(f"  ─────────────────────────────────")

    if len(df) > 0:
        print(f"  SSIM min          : {df['Score_SSIM'].min():.4f}")
        print(f"  SSIM max          : {df['Score_SSIM'].max():.4f}")
        print(f"  SSIM médian       : {df['Score_SSIM'].median():.4f}")

    print(f"\n  CSV exporté → {os.path.abspath(OUTPUT_CSV)}")
    print(f"{'=' * 60}\n")

    if errors:
        print("  Patients en erreur :")
        for err in errors:
            print(f"    • {err['Patient_ID']} — {err['Error']}")
        print()

    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        df_labels = run_auto_curation()
    except (FileNotFoundError, ValueError) as fatal:
        print(f"\n[FATAL] {fatal}", file=sys.stderr)
        sys.exit(1)
