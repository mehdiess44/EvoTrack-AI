"""Clinical benchmarking utilities for controlled synthetic EvoTrack pairs."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.heatmap_generator import (
        compute_feature_difference_map,
        get_shared_backbone,
        load_or_build_model,
    )
    from evotrack_ai.metrics_extraction import binarize_diff_map
    from evotrack_ai.synthetic_lesions import generate_synthetic_lesion
    from evotrack_ai.synthetic_transforms import simulate_acquisition_variation
    from evotrack_ai.tf_data_pipeline import preprocess_image
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from heatmap_generator import (
        compute_feature_difference_map,
        get_shared_backbone,
        load_or_build_model,
    )
    from metrics_extraction import binarize_diff_map
    from synthetic_lesions import generate_synthetic_lesion
    from synthetic_transforms import simulate_acquisition_variation
    from tf_data_pipeline import preprocess_image


DEFAULT_DATASET_DIR = Path("data/raw/Br35H/no")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmarks")
DEFAULT_MODEL_PATH = Path("models/evotrack_siamese_best.keras")


def ensure_output_dir(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Create the benchmark output directory if it does not already exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_model_input(image: np.ndarray) -> tf.Tensor:
    """Preprocess one image and add the batch dimension expected by Keras."""
    image_tensor = preprocess_image(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    return image_tensor


def predict_pair_label(
    model: tf.keras.Model,
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    threshold: float = 0.5,
) -> tuple[int, float]:
    """Predict whether a T0/T1 pair is stable or evolving."""
    x0 = prepare_model_input(img_t0)
    x1 = prepare_model_input(img_t1)

    prediction = model.predict([x0, x1], verbose=0)
    probability = float(np.asarray(prediction).reshape(-1)[0])
    predicted_label = 1 if probability >= threshold else 0

    return predicted_label, probability


def generate_benchmark_pairs(
    image_paths,
    num_pairs: int = 40,
    seed: int = 42,
) -> list[dict]:
    """Generate balanced synthetic benchmark pairs without loading all images."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")

    paths = list(image_paths)

    if not paths:
        raise ValueError("image_paths must contain at least one image path.")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pairs: list[dict] = []

    for index in range(num_pairs):
        image_path = paths[int(rng.integers(0, len(paths)))]
        img_t0 = read_image_grayscale(image_path)

        if index % 2 == 0:
            img_t1 = simulate_acquisition_variation(img_t0)
            label = 0
            ground_truth_mask = np.zeros(img_t0.shape, dtype=np.uint8)
        else:
            img_t1_base = simulate_acquisition_variation(img_t0)
            img_t1, ground_truth_mask = generate_synthetic_lesion(img_t1_base)
            label = 1

        pairs.append(
            {
                "t0": img_t0,
                "t1": img_t1,
                "label": label,
                "ground_truth_mask": ground_truth_mask,
            }
        )

    return pairs


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": sklearn_confusion_matrix(y_true, y_pred),
    }


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute the Intersection over Union between two binary masks."""
    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)

    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    union = np.logical_or(mask_a_bool, mask_b_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def predict_heatmap_mask(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """Predict a binary mask from the feature-difference heatmap."""
    backbone = get_shared_backbone(model)
    diff_map = compute_feature_difference_map(img_t0, img_t1, backbone)
    heatmap_mask = binarize_diff_map(diff_map)

    return heatmap_mask.astype(np.uint8)


def evaluate_iou_on_positive_pairs(
    pairs: list[dict],
    model: tf.keras.Model,
) -> list[float]:
    """Evaluate heatmap-mask IoU only on synthetic positive pairs."""
    iou_scores: list[float] = []

    for pair in pairs:
        if pair["label"] != 1:
            continue

        predicted_mask = predict_heatmap_mask(pair["t0"], pair["t1"], model)
        ground_truth_mask = pair["ground_truth_mask"]

        if ground_truth_mask.shape != predicted_mask.shape:
            ground_truth_mask = cv2.resize(
                ground_truth_mask,
                (predicted_mask.shape[1], predicted_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        iou_scores.append(compute_iou(predicted_mask, ground_truth_mask))

    return iou_scores


def save_confusion_matrix_plot(
    confusion_matrix_array,
    output_path: Path = DEFAULT_OUTPUT_DIR / "confusion_matrix.png",
) -> None:
    """Save an annotated confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0 = stabilité", "1 = évolution"],
        yticklabels=["0 = stabilité", "1 = évolution"],
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.title("Clinical Benchmark Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_report(
    metrics: dict,
    iou_scores: list[float],
    output_path: Path = DEFAULT_OUTPUT_DIR / "clinical_metrics_report.txt",
) -> None:
    """Save a concise text report with classification and IoU results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    with output_path.open("w", encoding="utf-8") as report_file:
        report_file.write("Clinical benchmark metrics\n")
        report_file.write("==========================\n")
        report_file.write(f"accuracy: {metrics['accuracy']:.4f}\n")
        report_file.write(f"precision: {metrics['precision']:.4f}\n")
        report_file.write(f"recall: {metrics['recall']:.4f}\n")
        report_file.write(f"f1: {metrics['f1']:.4f}\n")
        report_file.write(f"mean IoU: {mean_iou:.4f}\n")
        report_file.write(f"number of evaluated pairs: {len(iou_scores)}\n")


def run_clinical_benchmark(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    num_pairs: int = 40,
    threshold: float = 0.5,
) -> dict:
    """Run the full synthetic clinical benchmark pipeline."""
    ensure_output_dir(DEFAULT_OUTPUT_DIR)

    image_paths = list_image_paths(dataset_dir)
    model = load_or_build_model(model_path)
    pairs = generate_benchmark_pairs(image_paths, num_pairs=num_pairs)

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []

    for pair in pairs:
        predicted_label, probability = predict_pair_label(
            model=model,
            img_t0=pair["t0"],
            img_t1=pair["t1"],
            threshold=threshold,
        )
        y_true.append(int(pair["label"]))
        y_pred.append(predicted_label)
        probabilities.append(probability)

    classification_metrics = compute_classification_metrics(y_true, y_pred)
    iou_scores = evaluate_iou_on_positive_pairs(pairs, model)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    save_confusion_matrix_plot(classification_metrics["confusion_matrix"])
    save_metrics_report(classification_metrics, iou_scores)

    return {
        "classification_metrics": classification_metrics,
        "iou_scores": iou_scores,
        "mean_iou": mean_iou,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    identical_mask_a = np.zeros((10, 10), dtype=np.uint8)
    identical_mask_b = np.zeros((10, 10), dtype=np.uint8)
    identical_mask_a[2:5, 2:5] = 255
    identical_mask_b[2:5, 2:5] = 255

    non_overlapping_mask_a = np.zeros((10, 10), dtype=np.uint8)
    non_overlapping_mask_b = np.zeros((10, 10), dtype=np.uint8)
    non_overlapping_mask_a[1:3, 1:3] = 255
    non_overlapping_mask_b[7:9, 7:9] = 255

    assert compute_iou(identical_mask_a, identical_mask_b) == 1.0
    assert compute_iou(non_overlapping_mask_a, non_overlapping_mask_b) == 0.0

    results = run_clinical_benchmark(num_pairs=20)
    metrics = results["classification_metrics"]

    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1: {metrics['f1']:.4f}")
    print(f"mean IoU: {results['mean_iou']:.4f}")
    print(f"output directory: {DEFAULT_OUTPUT_DIR}")

    assert type(results) is dict
    assert "classification_metrics" in results
    assert "iou_scores" in results
