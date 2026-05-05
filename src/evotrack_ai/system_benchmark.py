"""System latency and bandwidth benchmark utilities for EvoTrack AI."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.heatmap_generator import load_or_build_model
    from evotrack_ai.longitudinal_pipeline import analyze_longitudinal_scan
    from evotrack_ai.siamese_model import build_siamese_model
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from heatmap_generator import load_or_build_model
    from longitudinal_pipeline import analyze_longitudinal_scan
    from siamese_model import build_siamese_model
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation


DEFAULT_DATASET_DIR = Path("data/raw/Br35H/no")
DEFAULT_OUTPUT_DIR = Path("outputs/system_benchmarks")
DEFAULT_MODEL_PATH = Path("models/evotrack_siamese_best.keras")


def ensure_output_dir(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Create the output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def get_file_size_bytes(path: str | Path) -> int:
    """Return the size of one file in bytes."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {file_path}")

    return int(file_path.stat().st_size)


def estimate_image_pair_size_bytes(t0_path: str | Path, t1_path: str | Path) -> int:
    """Return the combined size of two image files in bytes."""
    return get_file_size_bytes(t0_path) + get_file_size_bytes(t1_path)


def benchmark_single_patient_latency(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    model,
    num_runs: int = 5,
    warmup_runs: int = 1,
) -> dict:
    """Measure the full pipeline latency for one synthetic patient pair."""
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")

    for _ in range(warmup_runs):
        analyze_longitudinal_scan(img_t0, img_t1, model=model)

    latencies: list[float] = []

    for _ in range(num_runs):
        start_time = time.perf_counter()
        analyze_longitudinal_scan(img_t0, img_t1, model=model)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    return {
        "num_runs": int(num_runs),
        "mean_latency_seconds": float(np.mean(latencies)),
        "min_latency_seconds": float(np.min(latencies)),
        "max_latency_seconds": float(np.max(latencies)),
        "latencies": latencies,
    }


def create_latency_test_pair(image_paths) -> tuple[np.ndarray, np.ndarray]:
    """Create one controlled T0/T1 pair for latency testing."""
    paths = list(image_paths)

    if not paths:
        raise ValueError("image_paths must contain at least one image path.")

    img_t0 = read_image_grayscale(paths[0])
    img_t1_base = simulate_acquisition_variation(img_t0)
    img_t1, _ = generate_synthetic_lesion(img_t1_base)

    return img_t0, img_t1


def estimate_cloud_transfer_bytes(
    image_paths,
    num_patients: int = 100,
) -> int:
    """Estimate centralized-cloud transfer size for T0 and T1 images."""
    if num_patients <= 0:
        raise ValueError("num_patients must be positive.")

    paths = list(image_paths)

    if not paths:
        raise ValueError("image_paths must contain at least one image path.")

    total_bytes = 0

    for patient_index in range(num_patients):
        t0_path = paths[(patient_index * 2) % len(paths)]
        t1_path = paths[(patient_index * 2 + 1) % len(paths)]
        total_bytes += estimate_image_pair_size_bytes(t0_path, t1_path)

    return int(total_bytes)


def estimate_clinical_cloud_transfer_bytes(
    num_patients: int = 100,
    exam_size_mb: float = 50.0,
    exams_per_patient: int = 2,
) -> int:
    """Estimate transfer size for realistic simulated clinical exams."""
    if num_patients <= 0:
        raise ValueError("num_patients must be positive.")

    if exam_size_mb <= 0:
        raise ValueError("exam_size_mb must be positive.")

    if exams_per_patient <= 0:
        raise ValueError("exams_per_patient must be positive.")

    cloud_bytes = num_patients * exams_per_patient * exam_size_mb * 1024 * 1024

    return int(cloud_bytes)


def estimate_federated_transfer_bytes(
    model_path: Path = DEFAULT_MODEL_PATH,
    fallback_model=None,
) -> int:
    """Estimate federated transfer size from the model file or its weights."""
    model_file = Path(model_path)

    if model_file.exists():
        return get_file_size_bytes(model_file)

    model = fallback_model if fallback_model is not None else build_siamese_model()
    weights_size = sum(weight.nbytes for weight in model.get_weights())

    return int(weights_size)


def bytes_to_megabytes(num_bytes: int) -> float:
    """Convert bytes to megabytes."""
    return float(num_bytes / (1024 * 1024))


def compute_bandwidth_savings(
    cloud_bytes: int,
    federated_bytes: int,
) -> dict:
    """Compute bandwidth savings between cloud and federated scenarios."""
    if cloud_bytes <= 0:
        raise ValueError("cloud_bytes must be positive.")

    cloud_mb = bytes_to_megabytes(cloud_bytes)
    federated_mb = bytes_to_megabytes(federated_bytes)
    saved_mb = bytes_to_megabytes(cloud_bytes - federated_bytes)
    savings_percent = (1 - federated_bytes / cloud_bytes) * 100

    return {
        "cloud_mb": float(cloud_mb),
        "federated_mb": float(federated_mb),
        "saved_mb": float(saved_mb),
        "savings_percent": float(savings_percent),
    }


def build_bandwidth_dataframe(
    demo_cloud_bytes: int,
    clinical_cloud_bytes: int,
    federated_bytes: int,
) -> pd.DataFrame:
    """Build a comparison table for demo, clinical, and federated transfers."""
    return pd.DataFrame(
        [
            {
                "scenario": "Cloud centralisé — Br35H JPG",
                "transfer_mb": bytes_to_megabytes(demo_cloud_bytes),
                "description": "Démo avec fichiers Br35H JPG compressés",
            },
            {
                "scenario": "Cloud centralisé — clinique simulé",
                "transfer_mb": bytes_to_megabytes(clinical_cloud_bytes),
                "description": "Simulation DICOM/NIfTI avec examens volumineux",
            },
            {
                "scenario": "Apprentissage fédéré — poids modèle",
                "transfer_mb": bytes_to_megabytes(federated_bytes),
                "description": "Envoi du modèle ou des poids appris",
            },
        ]
    )


def save_system_report(
    latency: dict,
    demo_bandwidth: dict,
    clinical_bandwidth: dict,
    output_path: Path = DEFAULT_OUTPUT_DIR / "system_benchmark_report.txt",
) -> None:
    """Save a text report for latency and bandwidth benchmark results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as report_file:
        report_file.write("System benchmark report\n")
        report_file.write("=======================\n")
        report_file.write(f"latency runs: {latency['num_runs']}\n")
        report_file.write(
            f"mean latency seconds: {latency['mean_latency_seconds']:.4f}\n"
        )
        report_file.write(
            f"min latency seconds: {latency['min_latency_seconds']:.4f}\n"
        )
        report_file.write(
            f"max latency seconds: {latency['max_latency_seconds']:.4f}\n"
        )
        report_file.write(
            f"Cloud Br35H JPG MB: {demo_bandwidth['cloud_mb']:.4f}\n"
        )
        report_file.write(
            "Cloud clinique simulé MB: "
            f"{clinical_bandwidth['cloud_mb']:.4f}\n"
        )
        report_file.write(
            f"Federated MB: {demo_bandwidth['federated_mb']:.4f}\n"
        )
        report_file.write(
            "Savings Br35H JPG %: "
            f"{demo_bandwidth['savings_percent']:.2f}\n"
        )
        report_file.write(
            "Savings clinique simulé %: "
            f"{clinical_bandwidth['savings_percent']:.2f}\n"
        )


def save_bandwidth_plot(
    dataframe: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT_DIR / "bandwidth_comparison.png",
) -> None:
    """Save a simple bar chart comparing transfer size scenarios."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Cloud Br35H JPG", "Cloud clinique simulé", "Fédéré"]
    colors = ["#4C78A8", "#F28E2B", "#59A14F"]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, dataframe["transfer_mb"], color=colors)
    plt.ylabel("Transfer size (MB)")
    plt.title("Bandwidth Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_system_benchmark(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    num_latency_runs: int = 5,
    num_patients: int = 100,
) -> dict:
    """Run latency and bandwidth benchmarks for the full system."""
    ensure_output_dir(DEFAULT_OUTPUT_DIR)

    image_paths = list_image_paths(dataset_dir)
    model = load_or_build_model(model_path)

    img_t0, img_t1 = create_latency_test_pair(image_paths)
    latency = benchmark_single_patient_latency(
        img_t0,
        img_t1,
        model=model,
        num_runs=num_latency_runs,
        warmup_runs=1,
    )

    demo_cloud_bytes = estimate_cloud_transfer_bytes(
        image_paths,
        num_patients=num_patients,
    )
    clinical_cloud_bytes = estimate_clinical_cloud_transfer_bytes(
        num_patients=num_patients,
        exam_size_mb=50.0,
        exams_per_patient=2,
    )
    federated_bytes = estimate_federated_transfer_bytes(
        model_path=model_path,
        fallback_model=model,
    )
    demo_bandwidth = compute_bandwidth_savings(demo_cloud_bytes, federated_bytes)
    clinical_bandwidth = compute_bandwidth_savings(
        clinical_cloud_bytes,
        federated_bytes,
    )
    dataframe = build_bandwidth_dataframe(
        demo_cloud_bytes,
        clinical_cloud_bytes,
        federated_bytes,
    )

    dataframe.to_csv(
        DEFAULT_OUTPUT_DIR / "bandwidth_comparison.csv",
        index=False,
        encoding="utf-8",
    )
    save_system_report(latency, demo_bandwidth, clinical_bandwidth)
    save_bandwidth_plot(dataframe)

    return {
        "latency": latency,
        "demo_bandwidth": demo_bandwidth,
        "clinical_bandwidth": clinical_bandwidth,
        "dataframe": dataframe,
    }


if __name__ == "__main__":
    results = run_system_benchmark(num_latency_runs=3, num_patients=100)

    print(
        "mean latency seconds: "
        f"{results['latency']['mean_latency_seconds']:.4f}"
    )
    print(
        "min latency seconds: "
        f"{results['latency']['min_latency_seconds']:.4f}"
    )
    print(
        "max latency seconds: "
        f"{results['latency']['max_latency_seconds']:.4f}"
    )
    print(f"Br35H cloud transfer MB: {results['demo_bandwidth']['cloud_mb']:.4f}")
    print(
        "clinical cloud transfer MB: "
        f"{results['clinical_bandwidth']['cloud_mb']:.4f}"
    )
    print(
        "federated transfer MB: "
        f"{results['demo_bandwidth']['federated_mb']:.4f}"
    )
    print(
        "Br35H savings percent: "
        f"{results['demo_bandwidth']['savings_percent']:.2f}"
    )
    print(
        "clinical savings percent: "
        f"{results['clinical_bandwidth']['savings_percent']:.2f}"
    )
    print(f"output directory: {DEFAULT_OUTPUT_DIR}")

    assert type(results) is dict
    assert "latency" in results
    assert "demo_bandwidth" in results
    assert "clinical_bandwidth" in results
    assert results["latency"]["mean_latency_seconds"] > 0
    assert results["demo_bandwidth"]["cloud_mb"] > 0
