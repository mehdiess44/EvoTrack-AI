"""Streamlit MVP for EvoTrack AI longitudinal scan analysis."""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evotrack_ai.heatmap_generator import load_or_build_model
from evotrack_ai.longitudinal_pipeline import analyze_longitudinal_scan


@st.cache_resource
def load_model_cached():
    """Load the Siamese model once for the Streamlit session."""
    return load_or_build_model()


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Read a Streamlit uploaded image as a grayscale uint8 NumPy array.

    Args:
        uploaded_file: File object returned by ``st.file_uploader``.

    Returns:
        Grayscale image with dtype ``np.uint8``.

    Raises:
        ValueError: If the uploaded file cannot be read as an image.
    """
    try:
        image = Image.open(uploaded_file)
        image = image.convert("L")
        return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("image invalide ou illisible") from error


def convert_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR or grayscale OpenCV image to RGB for Streamlit display."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ValueError("image invalide pour affichage")

    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
        return cv2.cvtColor(image_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)

    raise ValueError("format d'image non supporté pour affichage")


def blend_overlay_with_original(
    original_gray: np.ndarray,
    overlay_bgr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend original T1 with a cached overlay using a display opacity."""
    if original_gray is None or original_gray.size == 0:
        raise ValueError("image originale invalide")

    original_resized = cv2.resize(
        original_gray.astype(np.uint8),
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )
    original_rgb = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)
    overlay_rgb = convert_bgr_to_rgb(overlay_bgr)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    blended = cv2.addWeighted(
        original_rgb,
        1.0 - alpha,
        overlay_rgb,
        alpha,
        0,
    )

    return blended.astype(np.uint8)


st.set_page_config(page_title="EvoTrack AI", layout="wide")

st.title("EvoTrack AI — Analyse Longitudinale Assistée")

with st.sidebar:
    heatmap_alpha = st.slider(
        "Opacité de la heatmap",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
    )
    st.info(
        "Outil d'aide descriptive uniquement. "
        "Le système ne fournit pas de diagnostic médical autonome."
    )

accepted_formats = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

upload_col_1, upload_col_2 = st.columns(2)

with upload_col_1:
    uploaded_t0 = st.file_uploader(
        "Uploader l’image T0",
        type=accepted_formats,
        key="uploaded_t0",
    )

with upload_col_2:
    uploaded_t1 = st.file_uploader(
        "Uploader l’image T1",
        type=accepted_formats,
        key="uploaded_t1",
    )

if st.button("Lancer l’analyse", type="primary"):
    if uploaded_t0 is None or uploaded_t1 is None:
        st.warning("Veuillez uploader les images T0 et T1 avant de lancer l’analyse.")
    else:
        try:
            with st.spinner("Analyse longitudinale en cours..."):
                img_t0 = read_uploaded_image(uploaded_t0)
                img_t1 = read_uploaded_image(uploaded_t1)
                model = load_model_cached()
                overlay, summary = analyze_longitudinal_scan(
                    img_t0,
                    img_t1,
                    model=model,
                )

                st.session_state["overlay"] = overlay
                st.session_state["summary"] = summary
                st.session_state["img_t1"] = img_t1
        except Exception as error:
            st.error(f"Analyse impossible : {error}")

if "overlay" in st.session_state and "summary" in st.session_state:
    result_col_1, result_col_2, result_col_3 = st.columns(3)

    with result_col_1:
        st.subheader("T1 originale")
        t1_rgb = convert_bgr_to_rgb(st.session_state["img_t1"])
        st.image(t1_rgb, use_container_width=True)

    with result_col_2:
        st.subheader("T1 avec heatmap")
        blended_overlay = blend_overlay_with_original(
            st.session_state["img_t1"],
            st.session_state["overlay"],
            heatmap_alpha,
        )
        st.image(blended_overlay, use_container_width=True)

    with result_col_3:
        st.subheader("Résumé clinique")
        st.info(st.session_state["summary"])
        st.caption("Résumé descriptif uniquement, sans diagnostic autonome.")
