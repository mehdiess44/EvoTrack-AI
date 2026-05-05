"""Deterministic geometric registration utilities using OpenCV."""

from pathlib import Path

import cv2
import numpy as np


try:
    from evotrack_ai.data_ingestion import list_image_paths, read_image_grayscale
    from evotrack_ai.synthetic_transforms import simulate_patient_movement
except ModuleNotFoundError:
    from data_ingestion import list_image_paths, read_image_grayscale
    from synthetic_transforms import simulate_patient_movement


def validate_grayscale_image(image: np.ndarray, name: str = "image") -> None:
    """Validate that an image is a non-empty grayscale NumPy array.

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

    if len(image.shape) != 2:
        raise ValueError(f"{name} must be a grayscale image with shape (H, W).")


def extract_keypoints(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int = 1000,
) -> tuple:
    """Extract ORB keypoints and descriptors from two grayscale images.

    Args:
        img1: First grayscale image.
        img2: Second grayscale image.
        max_features: Maximum number of ORB features to detect.

    Returns:
        ``keypoints1, descriptors1, keypoints2, descriptors2``.

    Raises:
        ValueError: If images are invalid or too few points are detected.
    """
    validate_grayscale_image(img1, name="img1")
    validate_grayscale_image(img2, name="img2")

    if max_features <= 0:
        raise ValueError("max_features must be positive.")

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("ORB could not compute descriptors for both images.")

    if len(keypoints1) < 4 or len(keypoints2) < 4:
        raise ValueError("At least 4 keypoints are required in each image.")

    return keypoints1, descriptors1, keypoints2, descriptors2


def match_keypoints(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.75,
) -> list:
    """Match ORB descriptors with Lowe's ratio test.

    Args:
        descriptors1: ORB descriptors from the first image.
        descriptors2: ORB descriptors from the second image.
        ratio_threshold: Lowe ratio threshold.

    Returns:
        A list of good OpenCV DMatch objects.

    Raises:
        ValueError: If fewer than 4 good matches are found.
    """
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors must not be None.")

    if ratio_threshold <= 0:
        raise ValueError("ratio_threshold must be positive.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue

        first_match, second_match = match_pair

        if first_match.distance < ratio_threshold * second_match.distance:
            good_matches.append(first_match)

    if len(good_matches) < 4:
        raise ValueError("At least 4 good matches are required.")

    return good_matches


def estimate_homography(
    keypoints1,
    keypoints2,
    matches: list,
    ransac_reproj_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a homography matrix from matched keypoints.

    Args:
        keypoints1: Keypoints from the reference image.
        keypoints2: Keypoints from the image to align.
        matches: Good descriptor matches.
        ransac_reproj_threshold: RANSAC reprojection threshold.

    Returns:
        ``homography_matrix, inlier_mask``.

    Raises:
        ValueError: If homography estimation fails.
    """
    if len(matches) < 4:
        raise ValueError("At least 4 matches are required for homography.")

    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    homography_matrix, inlier_mask = cv2.findHomography(
        points2,
        points1,
        cv2.RANSAC,
        ransac_reproj_threshold,
    )

    if homography_matrix is None:
        raise ValueError("Homography estimation failed.")

    return homography_matrix, inlier_mask


def align_images(
    img_t0: np.ndarray,
    img_t1: np.ndarray,
    max_features: int = 1000,
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Align T1 onto T0 using ORB feature matching and homography.

    Args:
        img_t0: Reference grayscale image.
        img_t1: Moving grayscale image to align.
        max_features: Maximum number of ORB features.
        ratio_threshold: Lowe ratio threshold for descriptor matching.

    Returns:
        ``aligned_t1, homography_matrix``.
    """
    validate_grayscale_image(img_t0, name="img_t0")
    validate_grayscale_image(img_t1, name="img_t1")

    keypoints1, descriptors1, keypoints2, descriptors2 = extract_keypoints(
        img_t0,
        img_t1,
        max_features=max_features,
    )
    matches = match_keypoints(
        descriptors1,
        descriptors2,
        ratio_threshold=ratio_threshold,
    )
    homography_matrix, _ = estimate_homography(keypoints1, keypoints2, matches)

    height, width = img_t0.shape
    aligned_t1 = cv2.warpPerspective(
        img_t1,
        homography_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return aligned_t1.astype(np.uint8), homography_matrix


def compute_absolute_difference_score(
    img1: np.ndarray,
    img2: np.ndarray,
) -> float:
    """Compute the mean absolute pixel difference between two images."""
    validate_grayscale_image(img1, name="img1")
    validate_grayscale_image(img2, name="img2")

    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")

    return float(np.mean(cv2.absdiff(img1, img2)))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "raw" / "Br35H" / "no"

    image_paths = list_image_paths(dataset_dir)
    last_error = None

    for image_path in image_paths[:10]:
        try:
            t0 = read_image_grayscale(image_path)
            t1_shifted = simulate_patient_movement(t0)
            before_score = compute_absolute_difference_score(t0, t1_shifted)
            aligned_t1, homography = align_images(t0, t1_shifted)
            after_score = compute_absolute_difference_score(t0, aligned_t1)
            break
        except ValueError as error:
            last_error = error
    else:
        raise RuntimeError(
            "Registration failed on the first 10 images."
        ) from last_error

    print(f"T0 shape: {t0.shape}")
    print(f"shifted T1 shape: {t1_shifted.shape}")
    print(f"aligned T1 shape: {aligned_t1.shape}")
    print(f"difference before registration: {before_score}")
    print(f"difference after registration: {after_score}")
    print("homography matrix:")
    print(homography)

    assert aligned_t1.shape == t0.shape
    assert after_score >= 0
