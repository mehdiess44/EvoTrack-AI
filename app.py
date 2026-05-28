"""EvoTrack AI — Interface de Déploiement Clinique (Streamlit).

Application web pour l'analyse longitudinale d'IRM cérébrales.
Charge le réseau Siamois (Late Fusion, MobileNetV2) fine-tuné Sim2Real
et prédit la probabilité d'évolution tumorale entre deux examens (T0, T1).

Lancement :
    streamlit run app.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from vector_search import search_similar_cases

# ============================================================================
#  CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "evotrack_sim2real_final.keras"
FALLBACK_MODEL_PATH = SCRIPT_DIR / "models" / "evotrack_siamese_best.keras"

INPUT_SIZE = (224, 224)
ACCEPTED_FORMATS = ["png", "jpg", "jpeg"]
EVOLUTION_THRESHOLD = 0.50


# ============================================================================
#  ENREGISTREMENT CUSTOM KERAS (OBLIGATOIRE AVANT CHARGEMENT)
# ============================================================================

try:
    _register = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register = tf.keras.utils.register_keras_serializable


@_register(package="evotrack")
def absolute_difference(tensors):
    """Couche Lambda Siamoise : |φ(T0) − φ(T1)|."""
    return tf.abs(tensors[0] - tensors[1])


# ============================================================================
#  CHARGEMENT DU MODÈLE (CACHE STREAMLIT)
# ============================================================================

@st.cache_resource(show_spinner="Chargement du modèle EvoTrack AI…")
def load_model():
    """Charge le réseau Siamois pré-entraîné une seule fois par session.

    Tente le modèle fine-tuné Sim2Real, puis le modèle pré-entraîné
    synthétique en fallback.

    Returns
    -------
    tf.keras.Model
        Modèle Siamois compilé.
    """
    for path in (MODEL_PATH, FALLBACK_MODEL_PATH):
        if path.exists():
            try:
                model = tf.keras.models.load_model(str(path))
                return model
            except Exception:
                continue

    st.error(
        f"**Modèle introuvable.**\n\n"
        f"Chemins testés :\n"
        f"- `{MODEL_PATH}`\n"
        f"- `{FALLBACK_MODEL_PATH}`"
    )
    st.stop()


# ============================================================================
#  PRÉTRAITEMENT D'IMAGE
# ============================================================================

def preprocess_image(uploaded_file):
    """Convertit un fichier uploadé en tenseur prêt pour MobileNetV2.

    Pipeline :
        1. Lecture via PIL (gère tous les formats courants)
        2. Conversion RGB
        3. Redimensionnement à 224×224
        4. Normalisation [-1, 1] (convention MobileNetV2)
        5. Ajout de la dimension batch

    Parameters
    ----------
    uploaded_file : UploadedFile
        Fichier image issu de ``st.file_uploader``.

    Returns
    -------
    np.ndarray
        Tenseur float32 de shape ``(1, 224, 224, 3)`` dans ``[-1, 1]``.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(INPUT_SIZE, Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)

    # Normalisation MobileNetV2 : [0, 255] → [-1, 1]
    arr = (arr / 127.5) - 1.0

    return np.expand_dims(arr, axis=0)


def uploaded_to_display_array(uploaded_file):
    """Convertit un fichier uploadé en array RGB uint8 pour affichage.

    Returns
    -------
    np.ndarray
        Image RGB uint8 (H, W, 3).
    """
    image = Image.open(uploaded_file).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def compute_difference_map(uploaded_t0, uploaded_t1):
    """Calcule la carte de différence anatomique brute avec colormap JET.

    Pipeline :
        1. Redimensionne les deux images à 224×224
        2. Conversion en niveaux de gris
        3. Différence absolue pixel-à-pixel
        4. Application de la colormap JET (OpenCV)
        5. Conversion BGR → RGB pour Streamlit

    Returns
    -------
    np.ndarray
        Image RGB uint8 (224, 224, 3) — heatmap de différence.
    """
    img_t0 = Image.open(uploaded_t0).convert("L").resize(INPUT_SIZE, Image.LANCZOS)
    img_t1 = Image.open(uploaded_t1).convert("L").resize(INPUT_SIZE, Image.LANCZOS)

    arr_t0 = np.asarray(img_t0, dtype=np.float32)
    arr_t1 = np.asarray(img_t1, dtype=np.float32)

    diff = np.abs(arr_t0 - arr_t1)

    # Normalisation min-max → [0, 255] pour la colormap
    d_min, d_max = diff.min(), diff.max()
    if d_max - d_min > 1e-6:
        diff_norm = ((diff - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        diff_norm = np.zeros_like(diff, dtype=np.uint8)

    # Colormap JET (BGR) → RGB
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


# ============================================================================
#  CONFIGURATION DE LA PAGE STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="EvoTrack AI — Suivi Longitudinal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def extract_evolution_direction(uploaded_t0, uploaded_t1):
    """
    Détermine mathématiquement si l'évolution est une progression ou une régression.
    Utilise un masquage du fond et une normalisation Z-Score pour annuler le 
    biais d'acquisition (bias field) entre les deux sessions IRM.
    """
    img_t0 = Image.open(uploaded_t0).convert("L").resize(INPUT_SIZE, Image.LANCZOS)
    img_t1 = Image.open(uploaded_t1).convert("L").resize(INPUT_SIZE, Image.LANCZOS)
    
    arr_t0 = np.asarray(img_t0, dtype=np.float32)
    arr_t1 = np.asarray(img_t1, dtype=np.float32)
    
    # 1. Masquage (Brain Masking basique pour ignorer le fond noir)
    mask_t0 = arr_t0 > 15
    mask_t1 = arr_t1 > 15
    brain_mask = mask_t0 & mask_t1  # Intersection pour avoir une zone commune
    
    if not np.any(brain_mask):
        return "Progression"  # Sécurité fallback
        
    # 2. Normalisation Z-Score intra-masque
    mean_t0, std_t0 = np.mean(arr_t0[brain_mask]), np.std(arr_t0[brain_mask])
    mean_t1, std_t1 = np.mean(arr_t1[brain_mask]), np.std(arr_t1[brain_mask])
    
    # Protection division par zéro
    std_t0 = std_t0 if std_t0 > 1e-5 else 1.0
    std_t1 = std_t1 if std_t1 > 1e-5 else 1.0
    
    norm_t0 = np.zeros_like(arr_t0)
    norm_t1 = np.zeros_like(arr_t1)
    
    norm_t0[brain_mask] = (arr_t0[brain_mask] - mean_t0) / std_t0
    norm_t1[brain_mask] = (arr_t1[brain_mask] - mean_t1) / std_t1
    
    # 3. Soustraction signée et filtrage du bruit (seuil à 0.5 écart-type)
    diff = norm_t1 - norm_t0
    noise_threshold = 0.5 
    
    growth_mass = np.sum(diff[(diff > noise_threshold) & brain_mask])
    shrink_mass = np.sum(-diff[(diff < -noise_threshold) & brain_mask])
    
    return "Progression" if growth_mass > shrink_mass else "Régression"

# ── CSS Custom ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .main-title h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-title p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
    }

    .result-card {
        background: #0e1117;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .result-card h3 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
    }

    .prob-display {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        padding: 0.5rem 0;
    }
    .prob-evolution { color: #ef4444; }
    .prob-stable    { color: #22c55e; }

    .disclaimer {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: #0a0f1a;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
    }

    .stImage > img {
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .case-card {
        background: linear-gradient(145deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.3rem 1.2rem;
        height: 100%;
        transition: border-color 0.2s ease;
    }
    .case-card:hover {
        border-color: #3b82f6;
    }
    .case-card .case-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    .case-card .case-sim {
        font-size: 0.78rem;
        color: #60a5fa;
        margin-bottom: 0.7rem;
    }
    .case-card .case-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0.6rem 0 0.15rem 0;
    }
    .case-card .case-text {
        font-size: 0.83rem;
        color: #cbd5e1;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
#  EN-TÊTE
# ============================================================================

st.markdown(
    """
    <div class="main-title">
        <h1>🧠 EvoTrack AI — Suivi Longitudinal de Glioblastome</h1>
        <p>Analyse comparative d'IRM cérébrales par réseau Siamois (MobileNetV2)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
#  SIDEBAR — UPLOAD DES IMAGES
# ============================================================================

with st.sidebar:
    st.markdown("## 📁 Chargement des Examens")

    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85rem;'>"
        "Uploadez les coupes IRM du patient aux deux temps d'acquisition."
        "</p>",
        unsafe_allow_html=True,
    )

    uploaded_t0 = st.file_uploader(
        "🔹 Examen Baseline (T₀)",
        type=ACCEPTED_FORMATS,
        key="upload_t0",
        help="IRM de référence (premier examen).",
    )

    uploaded_t1 = st.file_uploader(
        "🔸 Examen de Suivi (T₁)",
        type=ACCEPTED_FORMATS,
        key="upload_t1",
        help="IRM de suivi (examen ultérieur).",
    )

    st.markdown("---")

    st.markdown(
        "<div class='disclaimer'>"
        "⚕️ <strong>Avertissement médical</strong><br>"
        "EvoTrack AI est un outil d'aide descriptive uniquement. "
        "Il ne fournit <strong>aucun diagnostic médical autonome</strong>. "
        "Toute décision clinique doit être validée par un professionnel de santé qualifié."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
#  ZONE PRINCIPALE — AFFICHAGE DES IMAGES
# ============================================================================

if uploaded_t0 and uploaded_t1:

    col_t0, col_t1 = st.columns(2, gap="large")

    with col_t0:
        st.markdown("#### 🔹 Baseline (T₀)")
        display_t0 = uploaded_to_display_array(uploaded_t0)
        st.image(display_t0, use_container_width=True)

    with col_t1:
        st.markdown("#### 🔸 Suivi (T₁)")
        display_t1 = uploaded_to_display_array(uploaded_t1)
        st.image(display_t1, use_container_width=True)

    st.markdown("---")

    # ── Bouton d'analyse ─────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_analysis = st.button(
            "🔬  Lancer l'Analyse Longitudinale",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        with st.spinner("Analyse en cours — inférence du réseau Siamois…"):

            # ── Chargement modèle ────────────────────────────────────────
            model = load_model()

            # ── Prétraitement ────────────────────────────────────────────
            uploaded_t0.seek(0)
            uploaded_t1.seek(0)
            tensor_t0 = preprocess_image(uploaded_t0)
            tensor_t1 = preprocess_image(uploaded_t1)

            # ── Inférence ────────────────────────────────────────────────
            prediction = model.predict(
                [tensor_t0, tensor_t1], verbose=0
            )
            probability = float(prediction[0][0])

        # ==============================================================
        #  AFFICHAGE DES RÉSULTATS
        # ==============================================================

        st.markdown("---")
        st.markdown("### 📊 Résultats de l'Analyse")

        res_col1, res_col2 = st.columns([1, 1], gap="large")

        # ── Colonne 1 : Probabilité + Verdict ────────────────────────
        with res_col1:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Score de Probabilité d'Évolution</h3>", unsafe_allow_html=True)

            prob_pct = probability * 100.0
            css_class = "prob-evolution" if probability > EVOLUTION_THRESHOLD else "prob-stable"

            st.markdown(
                f"<div class='prob-display {css_class}'>"
                f"{prob_pct:.1f} %"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.progress(min(probability, 1.0))

            if probability > EVOLUTION_THRESHOLD:
                st.error(
                    "🔴 **Alerte : Évolution Tumorale Détectée**\n\n"
                    f"Le modèle estime une probabilité de **{prob_pct:.1f} %** "
                    f"d'évolution entre les deux examens."
                )
            else:
                st.success(
                    "🟢 **Stabilité Clinique**\n\n"
                    f"Le modèle estime une probabilité de **{prob_pct:.1f} %** "
                    f"d'évolution. Aucune progression significative détectée."
                )

            st.metric(
                label="P(Évolution)",
                value=f"{prob_pct:.2f} %",
                delta=f"{'Évolution' if probability > EVOLUTION_THRESHOLD else 'Stable'}",
                delta_color="inverse" if probability > EVOLUTION_THRESHOLD else "normal",
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Colonne 2 : Carte de Différence Anatomique ───────────────
        with res_col2:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(
                "<h3>Carte de Différence Anatomique</h3>",
                unsafe_allow_html=True,
            )

            uploaded_t0.seek(0)
            uploaded_t1.seek(0)
            diff_map = compute_difference_map(uploaded_t0, uploaded_t1)

            st.image(
                diff_map,
                caption="Différence absolue |T₀ − T₁| (colormap JET)",
                use_container_width=True,
            )

            st.caption(
                "Les zones chaudes (rouge/jaune) indiquent les régions de "
                "variation anatomique maximale entre les deux examens. "
                "Cette carte est purement descriptive."
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ==============================================================
        #  RAPPORT CLINIQUE DÉTERMINISTE & ROUTAGE NLP
        # ==============================================================

        st.markdown("---")
        st.markdown("### 📝 Rapport Clinique & Recommandations")

        if probability > EVOLUTION_THRESHOLD:
            # 1. Analyse mathématique de la direction (100% Automatisé)
            uploaded_t0.seek(0)
            uploaded_t1.seek(0)
            auto_direction = extract_evolution_direction(uploaded_t0, uploaded_t1)
            
            if auto_direction == "Progression":
                # Requête silencieuse optimisée pour FAISS
                search_query = "Évolution tumorale confirmée, progression spatiale de la lésion, augmentation du volume, aggravation."
                # Texte affiché au médecin
                clinical_text = (
                    f"Suspicion de progression tumorale avec un score de confiance "
                    f"de {prob_pct:.1f}%. Augmentation du volume observée. "
                    f"Recommandation : Examen radiologique approfondi requis."
                )
                st.warning(f"🔴 **Alerte (Croissance) :** {clinical_text}")
            else:
                # Requête silencieuse optimisée pour FAISS
                search_query = "Réduction du volume tumoral, réponse positive au traitement, diminution du signal, régression."
                # Texte affiché au médecin
                clinical_text = (
                    f"Suspicion de régression tumorale avec un score de confiance "
                    f"de {prob_pct:.1f}%. Réponse thérapeutique favorable. "
                    f"Recommandation : Poursuite du traitement."
                )
                st.success(f"🟢 **Alerte (Réduction) :** {clinical_text}")

        else:
            stability_pct = 100.0 - prob_pct
            search_query = "Stabilité complète, aucune modification volumétrique significative, pas de progression."
            clinical_text = (
                f"Analyse stable. Aucune progression significative mise en "
                f"évidence (Stabilité estimée à {stability_pct:.1f}%). "
                f"Recommandation : Poursuite du protocole de suivi standard."
            )
            st.info(f"⚪ **Analyse :** {clinical_text}")

        # ==============================================================
        #  AIDE À LA DÉCISION — CAS HISTORIQUES SIMILAIRES (FAISS)
        # ==============================================================

        st.markdown("---")
        st.markdown(
            "### 📚 Aide à la Décision : Cas Historiques Similaires "
            "(NLP Vector Search)"
        )

        with st.spinner(
            "Recherche d'antécédents cliniques via "
            "Sentence-Transformers et FAISS…"
        ):
            # /!\ CRITIQUE : FAISS utilise désormais search_query et non plus clinical_text
            similar_cases = search_similar_cases(search_query, top_k=3)

        if similar_cases:
            case_cols = st.columns(3, gap="medium")

            for col, case in zip(case_cols, similar_cases):
                sim_score = case.get("similarity_score", 0.0)
                if sim_score < 0.60:
                    continue
                with col:
                    st.markdown(
                        f"<div class='case-card'>"
                        f"<div class='case-header'>"
                        f"🧬 Patient {case['patient_id']}"
                        f"</div>"
                        f"<div class='case-sim'>"
                        f"Similarité : {sim_score:.2f}"
                        f"</div>"
                        f"<div class='case-label'>Description</div>"
                        f"<div class='case-text'>"
                        f"<em>{case['clinical_description']}</em>"
                        f"</div>"
                        f"<div class='case-label'>Traitement appliqué</div>"
                        f"<div class='case-text'>"
                        f"{case['treatment_applied']}"
                        f"</div>"
                        f"<div class='case-label'>Issue clinique</div>"
                        f"<div class='case-text'>"
                        f"{case['outcome']}"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Aucun cas similaire trouvé dans la base historique.")