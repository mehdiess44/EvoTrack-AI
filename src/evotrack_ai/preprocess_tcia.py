"""Pretraitement longitudinal de la collection TCIA UPENN-GBM pour EvoTrack AI."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


LOGGER = logging.getLogger("preprocess_tcia")
NIFTI_SUFFIXES = (".nii.gz", ".nii")
FLAIR_ENDINGS = ("_flair.nii.gz", "_flair.nii")
T0_TIMEPOINT = "11"
T1_TIMEPOINT = "21"
T0_SEGMENTATION_ENDINGS = (
    "_11_automated_approx_segm.nii.gz",
    "_11_automated_approx_segm.nii",
)
TIMEPOINT_PATTERN = re.compile(
    r"^(?P<patient>.+?)[_-](?P<visit>(?:t|tp|timepoint|visit|session)?\d+)$",
    flags=re.IGNORECASE,
)
SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Exam:
    """Chemins d'un examen IRM a une date donnee."""

    patient_id: str
    timepoint: str
    flair_path: Path
    seg_path: Path | None


@dataclass(frozen=True)
class LongitudinalPair:
    """Paire chronologique d'examens pour un patient."""

    patient_id: str
    t0: Exam
    t1: Exam


def natural_sort_key(value: str | Path) -> list[str | int]:
    """Retourne une cle de tri qui respecte l'ordre des nombres."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def remove_nifti_suffix(filename: str) -> str:
    """Retire l'extension NIfTI, y compris l'extension composee .nii.gz."""

    lower_name = filename.lower()
    for suffix in NIFTI_SUFFIXES:
        if lower_name.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def safe_patient_name(patient_id: str) -> str:
    """Construit un identifiant utilisable comme nom de fichier Windows."""

    normalized = SAFE_NAME_PATTERN.sub("_", patient_id).strip("._")
    return normalized or "patient_inconnu"


class TciaExamParser:
    """Decouvre et associe les examens longitudinaux UPENN-GBM."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def discover_pairs(self) -> list[LongitudinalPair]:
        """Retourne les paires disposant de FLAIR _11/_21 et du masque _11."""

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dossier de donnees introuvable : {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Chemin racine invalide : {self.root_dir}")

        flair_paths = sorted(self._find_flair_paths(), key=natural_sort_key)
        if not flair_paths:
            LOGGER.warning("Aucun volume FLAIR NIfTI trouve sous %s.", self.root_dir)
            return []

        grouped_exams: dict[str, list[Exam]] = {}
        for flair_path in flair_paths:
            exam = self._build_exam(flair_path)
            grouped_exams.setdefault(exam.patient_id, []).append(exam)

        t0_segmentations = self._find_t0_segmentations()
        pairs: list[LongitudinalPair] = []
        for patient_id, exams in sorted(grouped_exams.items(), key=lambda item: natural_sort_key(item[0])):
            exams_by_timepoint = {
                exam.timepoint.lower(): exam for exam in self._unique_sorted_exams(exams)
            }
            if T0_TIMEPOINT not in exams_by_timepoint or T1_TIMEPOINT not in exams_by_timepoint:
                LOGGER.info("Patient ignore (FLAIR _11/_21 incomplets) : %s", patient_id)
                continue
            segmentation_path = t0_segmentations.get(patient_id.lower())
            if segmentation_path is None:
                LOGGER.error(
                    "Patient %s ignore : masque _11_automated_approx_segm manquant.",
                    patient_id,
                )
                continue

            t0_exam = exams_by_timepoint[T0_TIMEPOINT]
            t0_exam = Exam(t0_exam.patient_id, t0_exam.timepoint, t0_exam.flair_path, segmentation_path)
            pairs.append(LongitudinalPair(patient_id, t0_exam, exams_by_timepoint[T1_TIMEPOINT]))

        return pairs

    def _find_flair_paths(self) -> Iterable[Path]:
        """Parcourt les fichiers et ne retient que les volumes FLAIR."""

        for path in self.root_dir.rglob("*"):
            if path.is_file() and path.name.lower().endswith(FLAIR_ENDINGS):
                yield path

    def _build_exam(self, flair_path: Path) -> Exam:
        """Extrait l'identifiant patient et le point temporel d'un FLAIR."""

        exam_stem = remove_nifti_suffix(flair_path.name)[: -len("_flair")]
        patient_id, timepoint = self._infer_patient_and_timepoint(flair_path, exam_stem)
        return Exam(patient_id, timepoint, flair_path, None)

    def _infer_patient_and_timepoint(self, flair_path: Path, exam_stem: str) -> tuple[str, str]:
        """Infere le patient depuis le nom, puis depuis l'arborescence."""

        stem_match = TIMEPOINT_PATTERN.match(exam_stem)
        if stem_match:
            return stem_match.group("patient"), stem_match.group("visit")

        relative_parent = flair_path.parent.relative_to(self.root_dir)
        parent_parts = relative_parent.parts
        if parent_parts:
            folder_match = TIMEPOINT_PATTERN.match(parent_parts[0])
            if folder_match:
                return folder_match.group("patient"), folder_match.group("visit")
            if len(parent_parts) >= 2:
                return parent_parts[0], "/".join(parent_parts[1:])
            return parent_parts[0], exam_stem

        return exam_stem, exam_stem

    def _find_t0_segmentations(self) -> dict[str, Path]:
        """Indexe les masques automatiques T0 stockes dans automated_segm."""

        masks_by_patient: dict[str, Path] = {}
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue
            filename = path.name.lower()
            for ending in T0_SEGMENTATION_ENDINGS:
                if filename.endswith(ending):
                    patient_id = path.name[: -len(ending)]
                    masks_by_patient.setdefault(patient_id.lower(), path)
                    break
        return masks_by_patient

    @staticmethod
    def _unique_sorted_exams(exams: list[Exam]) -> list[Exam]:
        """Supprime les doublons de temps en conservant le premier chemin."""

        ordered = sorted(exams, key=lambda exam: natural_sort_key(exam.timepoint))
        unique_by_time: dict[str, Exam] = {}
        for exam in ordered:
            unique_by_time.setdefault(exam.timepoint.lower(), exam)
        return list(unique_by_time.values())


class TumorSliceLocator:
    """Selectionne la coupe axiale avec la plus grande surface lesionnelle."""

    @staticmethod
    def select_slice_index(segmentation_path: str | Path) -> int:
        """Binarise un masque 3D et retourne l'indice Z de surface maximale."""

        mask = load_nifti_volume(segmentation_path)
        binary_mask = mask > 0
        lesion_areas = np.count_nonzero(binary_mask, axis=(0, 1))

        if not np.any(lesion_areas):
            raise ValueError(f"Masque sans lesion positive : {segmentation_path}")

        return int(np.argmax(lesion_areas))


class SliceProcessor:
    """Extrait et normalise les tranches FLAIR et les masques binaires."""

    @staticmethod
    def extract_windowed_flair(flair_path: str | Path, z_index: int) -> np.ndarray:
        """Charge une tranche FLAIR et applique un windowing robuste en uint8."""

        volume = load_nifti_volume(flair_path)
        slice_2d = extract_axial_slice(volume, z_index, flair_path)
        return SliceProcessor.robust_window_uint8(slice_2d)

    @staticmethod
    def extract_binary_mask(seg_path: str | Path, z_index: int) -> np.ndarray:
        """Charge une tranche de segmentation et la convertit en masque uint8."""

        volume = load_nifti_volume(seg_path)
        slice_2d = extract_axial_slice(volume, z_index, seg_path)
        return ((slice_2d > 0).astype(np.uint8) * 255)

    @staticmethod
    def robust_window_uint8(slice_2d: np.ndarray) -> np.ndarray:
        """Clippe P1/P99 puis normalise une coupe dans l'intervalle [0, 255]."""

        float_slice = np.asarray(slice_2d, dtype=np.float32)
        float_slice = np.nan_to_num(float_slice, nan=0.0, posinf=0.0, neginf=0.0)
        lower, upper = np.percentile(float_slice, (1.0, 99.0))

        if upper <= lower:
            return np.zeros(float_slice.shape, dtype=np.uint8)

        clipped = np.clip(float_slice, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        return np.rint(normalized * 255.0).astype(np.uint8)


class DatasetExporter:
    """Redimensionne et exporte les images et le seul masque T0 disponible."""

    def __init__(self, output_dir: str | Path, image_size: tuple[int, int] = (224, 224)) -> None:
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.directories = {
            "images_T0": self.output_dir / "images_T0",
            "images_T1": self.output_dir / "images_T1",
            "masks_T0": self.output_dir / "masks_T0",
        }
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        legacy_masks_t1 = self.output_dir / "masks_T1"
        if legacy_masks_t1.exists():
            try:
                legacy_masks_t1.rmdir()
            except OSError:
                LOGGER.warning(
                    "Ancien dossier masks_T1 non vide conserve mais non utilise : %s",
                    legacy_masks_t1,
                )

    def export_pair(
        self,
        patient_id: str,
        flair_t0: np.ndarray,
        flair_t1: np.ndarray,
        mask_t0: np.ndarray,
    ) -> None:
        """Sauvegarde les images T0/T1 et le masque T0 en PNG 224x224."""

        prefix = safe_patient_name(patient_id)
        image_t0 = self._prepare_flair(flair_t0)
        image_t1 = self._prepare_flair(flair_t1)
        resized_mask_t0 = self._prepare_mask(mask_t0)

        self._write_png(self.directories["images_T0"] / f"{prefix}_T0.png", image_t0)
        self._write_png(self.directories["images_T1"] / f"{prefix}_T1.png", image_t1)
        self._write_png(self.directories["masks_T0"] / f"{prefix}_T0.png", resized_mask_t0)

    def _prepare_flair(self, image: np.ndarray) -> np.ndarray:
        """Redimensionne une FLAIR grise puis cree les trois canaux BGR."""

        resized = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        """Redimensionne sans interpolation de labels un masque binaire."""

        resized = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        return ((resized > 0).astype(np.uint8) * 255)

    @staticmethod
    def _write_png(destination: Path, image: np.ndarray) -> None:
        """Ecrit un PNG et signale explicitement une erreur disque."""

        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Echec d'ecriture du fichier PNG : {destination}")


def load_nifti_volume(path: str | Path) -> np.ndarray:
    """Charge un volume NIfTI 3D en memoire sous forme float32."""

    nifti_path = Path(path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"Fichier NIfTI introuvable : {nifti_path}")

    volume = np.asarray(nib.load(str(nifti_path)).dataobj, dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Volume NIfTI non 3D ({volume.shape}) : {nifti_path}")
    return volume


def extract_axial_slice(volume: np.ndarray, z_index: int, source: str | Path) -> np.ndarray:
    """Extrait une coupe Z apres validation de l'indice selectionne."""

    if not 0 <= z_index < volume.shape[2]:
        raise IndexError(
            f"Indice Z={z_index} hors limites pour {source} (profondeur={volume.shape[2]})."
        )
    return volume[:, :, z_index]


def validate_pair_files(pair: LongitudinalPair) -> None:
    """Verifie les fichiers indispensables avant tout chargement de volumes."""

    missing: list[str] = []
    if pair.t0.seg_path is None:
        missing.append("masque _11_automated_approx_segm T0")
    if missing:
        raise FileNotFoundError(
            f"{pair.patient_id} : fichier(s) manquant(s) : {', '.join(missing)}"
        )


def process_dataset(root_dir: str | Path, output_dir: str | Path) -> tuple[int, int]:
    """Execute le pipeline complet et retourne le nombre de succes et d'echecs."""

    pairs = TciaExamParser(root_dir).discover_pairs()
    exporter = DatasetExporter(output_dir)
    locator = TumorSliceLocator()
    processor = SliceProcessor()
    successes = 0
    failures = 0

    for pair in tqdm(pairs, desc="Pretraitement TCIA", unit="patient"):
        try:
            validate_pair_files(pair)
            assert pair.t0.seg_path is not None

            z_index = locator.select_slice_index(pair.t0.seg_path)
            flair_t0 = processor.extract_windowed_flair(pair.t0.flair_path, z_index)
            flair_t1 = processor.extract_windowed_flair(pair.t1.flair_path, z_index)
            mask_t0 = processor.extract_binary_mask(pair.t0.seg_path, z_index)
            exporter.export_pair(pair.patient_id, flair_t0, flair_t1, mask_t0)
            successes += 1
        except Exception as exc:
            failures += 1
            LOGGER.error("Patient %s ignore : %s", pair.patient_id, exc)

    return successes, failures


def build_argument_parser() -> argparse.ArgumentParser:
    """Construit l'interface de ligne de commande du pretraitement."""

    parser = argparse.ArgumentParser(
        description="Convertit des examens longitudinaux UPENN-GBM NIfTI en paires PNG EvoTrack."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Dossier racine contenant les volumes UPENN-GBM au format NIfTI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("EvoTrack_Finetuning_Dataset"),
        help="Dossier de sortie PNG (defaut : EvoTrack_Finetuning_Dataset).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Niveau de journalisation affiche dans le terminal.",
    )
    return parser


def main() -> int:
    """Point d'entree CLI."""

    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s - %(message)s",
    )

    try:
        successes, failures = process_dataset(args.root_dir, args.output_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info(
        "Export termine : %d patient(s) exporte(s), %d patient(s) ignore(s). Sortie : %s",
        successes,
        failures,
        args.output_dir,
    )
    return 0 if successes > 0 or failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
