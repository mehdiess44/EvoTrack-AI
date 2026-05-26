"""Fine-Tuning Sim2Real — EvoTrack AI.

Fine-tune le réseau Siamois (Late Fusion, MobileNetV2 partagé) pré-entraîné
sur données synthétiques vers la cohorte clinique UPENN-GBM (41 paires).

Toutes les paires réelles sont des évolutions (Y=1).
Les cas de stabilité (Y=0) sont générés dynamiquement via Self-Pairs bruitées.

Stack défensive :
    ● BatchNormalization gelées partout (inférence permanente)
    ● Blocs convolutifs 0–12 gelés, blocs 13–17 + tête dégelés
    ● L2 Anchor Penalty λ·‖θ − θ_pre‖² (λ = 1e-4)
    ● Warmup linéaire (2 epochs) → Cosine Decay (peak LR = 1e-4)
    ● Coin Flip 50 % pour invariance par symétrie
    ● 5-Fold Cross-Validation
    ● Early Stopping (patience = 4, restore_best_weights)
"""

import gc
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold


# ============================================================================
#  CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "EvoTrack_Finetuning_Dataset"
IMAGES_T0_DIR = DATASET_DIR / "images_T0"
IMAGES_T1_DIR = DATASET_DIR / "images_T1"

PRETRAINED_MODEL_KERAS = SCRIPT_DIR / "models" / "evotrack_siamese_best.keras"
PRETRAINED_MODEL_H5 = SCRIPT_DIR / "models" / "evotrack_siamese_best.h5"
OUTPUT_MODEL_PATH = SCRIPT_DIR / "evotrack_sim2real_final.keras"

INPUT_SHAPE = (224, 224, 3)
IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

NUM_FOLDS = 5
BATCH_SIZE = 8
MAX_EPOCHS = 30
WARMUP_EPOCHS = 2
PEAK_LR = 1e-4
ANCHOR_LAMBDA = 1e-4
EARLY_STOPPING_PATIENCE = 4
FREEZE_BLOCK_LIMIT = 12  # Blocs 0..12 gelés, 13+ dégelés

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================================
#  OBJET KERAS CUSTOM (Lambda Siamoise)
# ============================================================================

try:
    _register = tf.keras.saving.register_keras_serializable
except AttributeError:
    _register = tf.keras.utils.register_keras_serializable


@_register(package="evotrack")
def absolute_difference(tensors):
    """Couche Lambda intouchable : |φ(T0) − φ(T1)|."""
    return tf.abs(tensors[0] - tensors[1])


# ============================================================================
#  MODULE 1 — DONNÉES : DÉCOUVERTE & CHARGEMENT EN MÉMOIRE
# ============================================================================

def discover_pairs():
    """Trouve les paires T0/T1 par correspondance de noms de fichiers.

    Returns
    -------
    list[tuple[Path, Path]]
        Paires (chemin_T0, chemin_T1) triées par stem.

    Raises
    ------
    FileNotFoundError
        Si un des répertoires d'images est absent.
    ValueError
        Si aucune paire commune n'est détectée.
    """
    for d, label in [(IMAGES_T0_DIR, "images_T0"), (IMAGES_T1_DIR, "images_T1")]:
        if not d.is_dir():
            raise FileNotFoundError(f"Répertoire manquant : {d} ({label})")

    def _strip_timepoint(stem):
        """Supprime le suffixe _T0 ou _T1 pour obtenir l'ID patient."""
        for sfx in ("_T0", "_T1", "_t0", "_t1"):
            if stem.endswith(sfx):
                return stem[: -len(sfx)]
        return stem

    def _stems(directory):
        out = {}
        for f in sorted(os.listdir(directory)):
            p = Path(f)
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                patient_id = _strip_timepoint(p.stem)
                out[patient_id] = directory / f
        return out

    t0_map = _stems(IMAGES_T0_DIR)
    t1_map = _stems(IMAGES_T1_DIR)
    common = sorted(set(t0_map) & set(t1_map))

    if not common:
        raise ValueError(
            "Aucune paire T0/T1 commune trouvée. "
            "Vérifiez que les noms (sans extension) correspondent."
        )

    return [(t0_map[s], t1_map[s]) for s in common]


def load_all_pairs(pairs):
    """Charge toutes les images en mémoire (uint8 BGR 224×224×3).

    Returns
    -------
    (list[ndarray], list[ndarray])
        all_t0, all_t1 — listes parallèles d'images uint8.
    """
    all_t0, all_t1 = [], []

    for t0_path, t1_path in pairs:
        t0 = cv2.imread(str(t0_path), cv2.IMREAD_COLOR)
        t1 = cv2.imread(str(t1_path), cv2.IMREAD_COLOR)

        if t0 is None:
            raise ValueError(f"Lecture impossible : {t0_path}")
        if t1 is None:
            raise ValueError(f"Lecture impossible : {t1_path}")

        if t0.shape[:2] != IMAGE_SIZE:
            t0 = cv2.resize(t0, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        if t1.shape[:2] != IMAGE_SIZE:
            t1 = cv2.resize(t1, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        all_t0.append(t0)
        all_t1.append(t1)

    return all_t0, all_t1


# ============================================================================
#  MODULE 2 — AUGMENTATION
# ============================================================================

def _normalize_for_model(image_bgr):
    """BGR uint8 → RGB float32 dans [-1, 1] (range MobileNetV2)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 127.5) - 1.0


def _rotate(image, angle_deg):
    """Rotation autour du centre avec réflexion aux bords."""
    h, w = image.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _slight_zoom(image, factor):
    """Zoom central par facteur (ex : 1.03 = +3 %)."""
    h, w = image.shape[:2]
    nh, nw = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    cropped = resized[y0 : y0 + h, x0 : x0 + w]
    if cropped.shape[:2] != (h, w):
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    return cropped


def _simulate_new_acquisition(image, rng=None):
    """Simule une nouvelle session d'acquisition IRM sur une image existante.

    Applique des perturbations photométriques agressives pour rendre les
    self-pairs (Y=0) indistinguables des vraies paires multi-sessions (Y=1)
    du point de vue du bruit capteur, du contraste et de la luminosité.

    Chaîne de transformations :
        1. Variation de contraste    : α ∈ [0.8, 1.2]
        2. Variation de luminosité   : β ∈ [-20, +20]
        3. Bruit Gaussien capteur    : σ ∈ [3, 8]
        4. Rotation légère           : ±2°
        5. Zoom léger                : ×[1.00, 1.04]

    Parameters
    ----------
    image : ndarray
        Image BGR uint8 (224×224×3).
    rng : numpy.random.Generator or None
        Générateur aléatoire. Si None, un nouveau est créé.

    Returns
    -------
    ndarray
        Image perturbée, BGR uint8, mêmes dimensions.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.astype(np.float32)

    # ── 1. Variation de contraste (gain multiplicatif) ───────────────
    alpha = rng.uniform(0.8, 1.2)
    img = img * alpha

    # ── 2. Variation de luminosité (offset additif) ──────────────────
    beta = rng.uniform(-20.0, 20.0)
    img = img + beta

    # ── 3. Bruit Gaussien (simulation capteur thermique / magnétique) ─
    sigma = rng.uniform(3.0, 8.0)
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    img = img + noise

    # Clip + conversion uint8
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)

    # ── 4. Rotation légère (mouvement patient inter-session) ─────────
    img = _rotate(img, rng.uniform(-2.0, 2.0))

    # ── 5. Zoom léger (repositionnement dans le scanner) ─────────────
    img = _slight_zoom(img, rng.uniform(1.00, 1.04))

    return img


def _translate(image, tx, ty):
    """Translation rigide (tx, ty) pixels avec réflexion aux bords.

    Parameters
    ----------
    image : ndarray
        Image BGR uint8.
    tx, ty : float
        Décalage horizontal et vertical en pixels.

    Returns
    -------
    ndarray
        Image translatée, mêmes dimensions.
    """
    h, w = image.shape[:2]
    mat = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _apply_independent_jitter(image, rng):
    """Applique un jitter spatial indépendant à une image.

    Simule les micro-variations de positionnement patient entre sessions :
        1. Translation aléatoire X, Y ∈ [-12, +12] pixels
        2. Bruit Gaussien de fond léger (σ ∈ [1, 4])

    Appliqué **indépendamment** à chaque image de la paire (T0 et T1)
    pour empêcher le réseau d'exploiter l'alignement parfait des contours.

    Parameters
    ----------
    image : ndarray
        Image BGR uint8 (224×224×3).
    rng : numpy.random.Generator
        Générateur aléatoire.

    Returns
    -------
    ndarray
        Image jittée, BGR uint8, mêmes dimensions.
    """
    # ── 1. Translation aléatoire indépendante ─────────────────────────
    tx = rng.uniform(-12.0, 12.0)
    ty = rng.uniform(-12.0, 12.0)
    img = _translate(image, tx, ty)

    # ── 2. Bruit Gaussien de fond léger ──────────────────────────────
    sigma = rng.uniform(1.0, 4.0)
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    return img


# ============================================================================
#  MODULE 3 — tf.data PIPELINE (GÉNÉRATEURS)
# ============================================================================

def _training_generator(indices, all_t0, all_t1):
    """Générateur infini pour l'entraînement.

    Logique stochastique par sample :
        • 50 % → Évolution  (T0, T1)  — augmentation commune ±5°, flip.  Y=1.
        • 50 % → Stabilité  (T0, T0') — variation indépendante ±2°, zoom. Y=0.
        • 50 % → Coin Flip  (A, B) ↔ (B, A) pour l'invariance par symétrie.
    """
    rng = np.random.default_rng()

    while True:
        idx = indices[rng.integers(0, len(indices))]
        t0 = all_t0[idx].copy()

        if rng.random() < 0.5:
            # ── Évolution (Y=1) ──────────────────────────────────────────
            t1 = all_t1[idx].copy()
            angle = rng.uniform(-5.0, 5.0)
            t0, t1 = _rotate(t0, angle), _rotate(t1, angle)
            if rng.random() < 0.5:
                t0, t1 = cv2.flip(t0, 1), cv2.flip(t1, 1)
            label = 1
        else:
            # ── Stabilité / Self-Pair (Y=0) ──────────────────────────────
            # Simulation complète d'une nouvelle acquisition IRM :
            # contraste + luminosité + bruit capteur + géométrie.
            t1 = _simulate_new_acquisition(t0.copy(), rng)
            label = 0

        # ── Coin Flip (invariance par symétrie) ──────────────────────────
        if rng.random() < 0.5:
            t0, t1 = t1, t0

        # ── Independent Spatial Jitter (anti data-leakage géométrique) ───
        t0 = _apply_independent_jitter(t0, rng)
        t1 = _apply_independent_jitter(t1, rng)

        yield (
            (_normalize_for_model(t0), _normalize_for_model(t1)),
            np.float32(label),
        )


def _validation_generator(indices, all_t0, all_t1, seed=42):
    """Générateur fini et déterministe pour la validation.

    Pour chaque paire :
        1.  (T0, T1, Y=1)              — paire réelle d'évolution
        2.  (T0, T0_bruité, Y=0)       — self-pair avec variation légère

    Total : 2 × len(indices) samples, parfaitement équilibré.
    """
    rng = np.random.default_rng(seed)

    for idx in indices:
        t0 = all_t0[idx]
        t1 = all_t1[idx]

        # Y=1 : paire réelle + jitter indépendant
        t0_j1 = _apply_independent_jitter(t0.copy(), rng)
        t1_j1 = _apply_independent_jitter(t1.copy(), rng)
        yield (
            (_normalize_for_model(t0_j1), _normalize_for_model(t1_j1)),
            np.float32(1),
        )

        # Y=0 : self-pair (nouvelle acquisition) + jitter indépendant
        t0_copy = t0.copy()
        t0_aug = _simulate_new_acquisition(t0.copy(), rng)
        t0_j0 = _apply_independent_jitter(t0_copy, rng)
        t0_aug_j = _apply_independent_jitter(t0_aug, rng)
        yield (
            (_normalize_for_model(t0_j0), _normalize_for_model(t0_aug_j)),
            np.float32(0),
        )


_OUTPUT_SIGNATURE = (
    (
        tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
        tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(), dtype=tf.float32),
)


def _build_train_ds(indices, all_t0, all_t1, batch_size):
    """tf.data.Dataset d'entraînement (infini, équilibré, augmenté)."""
    ds = tf.data.Dataset.from_generator(
        lambda: _training_generator(indices, all_t0, all_t1),
        output_signature=_OUTPUT_SIGNATURE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_val_ds(indices, all_t0, all_t1, batch_size, seed=42):
    """tf.data.Dataset de validation (fini, cached pour itération multiple)."""
    ds = tf.data.Dataset.from_generator(
        lambda: _validation_generator(indices, all_t0, all_t1, seed=seed),
        output_signature=_OUTPUT_SIGNATURE,
    )
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


# ============================================================================
#  MODULE 4 — MODÈLE : CHARGEMENT, GEL, ANCRAGE
# ============================================================================

def _build_siamese_from_scratch():
    """Construit le réseau Siamois MobileNetV2 from scratch (fallback).

    Architecture :
        image_t0 ─┐
                   ├─ MobileNetV2 (partagé) → |Δ features| → GAP → Dense → σ
        image_t1 ─┘
    """
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    backbone._name = "mobilenetv2_backbone"

    inp_t0 = tf.keras.Input(shape=INPUT_SHAPE, name="image_t0")
    inp_t1 = tf.keras.Input(shape=INPUT_SHAPE, name="image_t1")

    feat_t0 = backbone(inp_t0)
    feat_t1 = backbone(inp_t1)

    diff = tf.keras.layers.Lambda(
        absolute_difference, name="absolute_difference"
    )([feat_t0, feat_t1])

    x = tf.keras.layers.GlobalAveragePooling2D()(diff)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(
        inputs=[inp_t0, inp_t1],
        outputs=output,
        name="evotrack_siamese_model",
    )


def _load_pretrained_model():
    """Charge le modèle Siamois pré-entraîné (.keras ou .h5)."""
    for path in (PRETRAINED_MODEL_KERAS, PRETRAINED_MODEL_H5):
        if path.exists():
            try:
                # Disparition de custom_objects=custom_obj
                model = tf.keras.models.load_model(str(path)) 
                print(f"  ✓ Modèle pré-entraîné Sim2Real chargé : {path.name}")
                return model
            except Exception as exc:
                print(f"  ⚠ Erreur chargement {path.name} : {exc}")

    print("  ⚠ Aucun modèle pré-entraîné → construction from scratch (ImageNet)")
    return _build_siamese_from_scratch()


def _find_backbone(model):
    """Identifie le backbone MobileNetV2 dans le modèle Siamois."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    # Fallback : sous-modèle avec le plus de couches
    candidates = [
        l for l in model.layers
        if isinstance(l, tf.keras.Model) and len(l.layers) > 10
    ]
    return candidates[0] if candidates else None


def _parse_block_number(layer_name):
    """Extrait le numéro de bloc MobileNetV2 depuis le nom de la couche.

    Returns
    -------
    int
        0..16 pour les blocs, 17 pour le conv final, -1 pour InputLayer etc.
    """
    match = re.search(r"block_(\d+)_", layer_name)
    if match:
        return int(match.group(1))
    if layer_name.startswith(("Conv1", "bn_Conv1", "expanded_conv")):
        return 0  # Couches initiales + bloc 0
    if layer_name.startswith(("Conv_1", "out_relu")):
        return 17  # Couches finales du backbone
    return -1  # InputLayer, etc.


def _apply_freezing_strategy(model):
    """Applique le gel défensif : TOUTES les BN gelées + blocs 0-12 gelés.

    Returns
    -------
    dict
        Compteurs : frozen_bn, frozen_block, unfrozen_block.
    """
    backbone = _find_backbone(model)
    if backbone is None:
        print("  ⚠ Backbone MobileNetV2 non trouvé — aucun gel appliqué")
        return {"frozen_bn": 0, "frozen_block": 0, "unfrozen_block": 0}

    # Dégeler tout le backbone puis appliquer les gels sélectifs
    backbone.trainable = True

    stats = {"frozen_bn": 0, "frozen_block": 0, "unfrozen_block": 0}

    for layer in backbone.layers:
        # ── RÈGLE 1 : TOUTES les BatchNorm sont gelées (inférence permanente)
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            stats["frozen_bn"] += 1
            continue

        # ── RÈGLE 2 : Blocs 0..12 gelés, blocs 13..17 dégelés
        block_num = _parse_block_number(layer.name)
        if block_num <= FREEZE_BLOCK_LIMIT:
            layer.trainable = False
            stats["frozen_block"] += 1
        else:
            layer.trainable = True
            stats["unfrozen_block"] += 1

    # Tête de classification toujours dégelée
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Dropout)):
            layer.trainable = True

    return stats


def _snapshot_trainable_weights(model):
    """Capture les valeurs pré-entraînées des poids dégelés.

    Returns
    -------
    list[tf.Tensor]
        Copies constantes (non-entraînables) dans le même ordre
        que ``model.trainable_variables``.
    """
    return [
        tf.constant(var.numpy(), dtype=tf.float32)
        for var in model.trainable_variables
    ]


# ============================================================================
#  MODULE 5 — LEARNING RATE SCHEDULE (WARMUP LINÉAIRE + COSINE DECAY)
# ============================================================================

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup linéaire sur ``warmup_steps``, puis Cosine Decay vers ~0.

    ::

        lr(t) = peak_lr × t / warmup_steps              si t < warmup_steps
        lr(t) = peak_lr × ½ × (1 + cos(π × progress))   sinon
    """

    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.constant(self.warmup_steps, tf.float32)
        total = tf.constant(self.total_steps, tf.float32)

        # Phase 1 : Warmup linéaire (0 → peak_lr)
        warmup_lr = self.peak_lr * step / tf.maximum(warmup, 1.0)

        # Phase 2 : Cosine Decay (peak_lr → ~0)
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.peak_lr * 0.5 * (1.0 + tf.math.cos(math.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# ============================================================================
#  MODULE 6 — BOUCLE D'ENTRAÎNEMENT CUSTOM + ÉVALUATION
# ============================================================================

def _train_one_fold(fold_idx, train_idx, val_idx, all_t0, all_t1):
    """Entraîne un fold complet avec le protocole défensif.

    Parameters
    ----------
    fold_idx : int
        Indice du fold (pour affichage).
    train_idx, val_idx : ndarray
        Indices des paires pour ce fold.
    all_t0, all_t1 : list[ndarray]
        Images en mémoire.

    Returns
    -------
    (dict, tf.keras.Model)
        Métriques de validation et modèle avec meilleurs poids restaurés.
    """
    num_train = len(train_idx)
    num_val = len(val_idx)

    # 2 × num_train car 50 % Y=1 + 50 % Y=0
    steps_per_epoch = max(1, (2 * num_train + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = _build_train_ds(train_idx, all_t0, all_t1, BATCH_SIZE)
    val_ds = _build_val_ds(val_idx, all_t0, all_t1, BATCH_SIZE)

    # ── Modèle + Gel ─────────────────────────────────────────────────────
    model = _load_pretrained_model()
    freeze_stats = _apply_freezing_strategy(model)

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    total_count = model.count_params()

    print(
        f"    Gel — BN: {freeze_stats['frozen_bn']} | "
        f"Blocs: {freeze_stats['frozen_block']} gelés, "
        f"{freeze_stats['unfrozen_block']} dégelés"
    )
    print(
        f"    Params — {trainable_count:,} entraînables / "
        f"{total_count:,} total ({100 * trainable_count / max(total_count, 1):.1f} %)"
    )

    # ── Anchor Penalty : snapshot avant tout gradient ────────────────────
    anchor_tensors = _snapshot_trainable_weights(model)
    trainable_vars = model.trainable_variables

    # ── Optimiseur (Warmup + Cosine) ─────────────────────────────────────
    lr_schedule = WarmupCosineSchedule(PEAK_LR, warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    bce_fn = tf.keras.losses.BinaryCrossentropy()

    # ── Early Stopping state ─────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    # ==================================================================
    #  BOUCLE D'ENTRAÎNEMENT
    # ==================================================================
    for epoch in range(MAX_EPOCHS):

        # ── Train ────────────────────────────────────────────────────
        epoch_losses = []

        for step, ((img_a, img_b), labels) in enumerate(train_ds):
            if step >= steps_per_epoch:
                break

            with tf.GradientTape() as tape:
                predictions = model([img_a, img_b], training=True)
                bce_loss = bce_fn(labels, predictions)

                # L2 Anchor Penalty : λ · Σ ‖θ − θ_pre‖²
                anchor_loss = tf.constant(0.0, dtype=tf.float32)
                for var, anchor in zip(trainable_vars, anchor_tensors):
                    anchor_loss = anchor_loss + tf.reduce_sum(
                        tf.square(var - anchor)
                    )
                total_loss = bce_loss + ANCHOR_LAMBDA * anchor_loss

            grads = tape.gradient(total_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            epoch_losses.append(float(total_loss))

        # ── Validation ───────────────────────────────────────────────
        val_losses = []
        for (va, vb), vlabels in val_ds:
            vpreds = model([va, vb], training=False)
            val_losses.append(float(bce_fn(vlabels, vpreds)))

        mean_train = np.mean(epoch_losses) if epoch_losses else 0.0
        mean_val = np.mean(val_losses) if val_losses else float("inf")

        approx_step = (epoch + 1) * steps_per_epoch
        current_lr = float(lr_schedule(approx_step))

        print(
            f"    Epoch {epoch + 1:02d}/{MAX_EPOCHS} — "
            f"loss: {mean_train:.4f} — val_loss: {mean_val:.4f} — "
            f"lr: {current_lr:.2e}"
        )

        # ── Early Stopping ───────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_weights = [v.numpy().copy() for v in trainable_vars]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"    ⇒ Early stopping déclenché (epoch {epoch + 1})")
                break

    # ── Restauration des meilleurs poids ─────────────────────────────────
    if best_weights is not None:
        for var, w in zip(trainable_vars, best_weights):
            var.assign(w)
        print(f"    ✓ Meilleurs poids restaurés (val_loss = {best_val_loss:.4f})")

    # ==================================================================
    #  ÉVALUATION FINALE SUR LE FOLD DE VALIDATION
    # ==================================================================
    all_preds, all_labels = [], []

    for (va, vb), vlabels in val_ds:
        vpreds = model([va, vb], training=False)
        all_preds.extend(vpreds.numpy().flatten().tolist())
        all_labels.extend(vlabels.numpy().flatten().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1

    fold_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.0,
        "val_loss": float(best_val_loss),
    }

    print(
        f"    ── Résultats fold {fold_idx + 1} ──\n"
        f"       Accuracy : {fold_metrics['accuracy']:.4f}\n"
        f"       F1-Score : {fold_metrics['f1']:.4f}\n"
        f"       AUC      : {fold_metrics['auc']:.4f}"
    )

    return fold_metrics, model


# ============================================================================
#  MODULE 7 — K-FOLD CROSS-VALIDATION (PIPELINE PRINCIPAL)
# ============================================================================

def run_kfold_finetuning():
    """Pipeline complet : découverte → K-Fold → entraînement → rapport."""

    # ── Bannière ─────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  EvoTrack AI — Fine-Tuning Sim2Real")
    print("  Architecture : Siamois Late Fusion (MobileNetV2, poids partagés)")
    print(f"  Folds: {NUM_FOLDS} | Epochs max: {MAX_EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"  LR: warmup({WARMUP_EPOCHS} ep) → {PEAK_LR} → cosine decay")
    print(f"  Anchor λ: {ANCHOR_LAMBDA} | Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Gel: blocs 0–{FREEZE_BLOCK_LIMIT} + toutes BatchNorm")
    print("=" * 66)

    # ── Découverte & chargement ──────────────────────────────────────────
    pairs = discover_pairs()
    num_pairs = len(pairs)
    print(f"\n  Paires T0/T1 découvertes : {num_pairs}")

    all_t0, all_t1 = load_all_pairs(pairs)
    mem_mb = sum(t.nbytes for t in all_t0 + all_t1) / (1024 ** 2)
    print(f"  Images chargées en mémoire ({num_pairs} × 2 = {2 * num_pairs} images, "
          f"{mem_mb:.1f} MB)")

    # ── K-Fold ───────────────────────────────────────────────────────────
    indices = np.arange(num_pairs)
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    all_fold_metrics = []
    best_global_auc = -1.0

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'━' * 66}")
        print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS} — "
              f"Train: {len(train_idx)} paires | Val: {len(val_idx)} paires")
        print(f"{'━' * 66}")

        fold_metrics, model = _train_one_fold(
            fold_idx, train_idx, val_idx, all_t0, all_t1
        )
        all_fold_metrics.append(fold_metrics)

        # ── Sauvegarde du meilleur modèle global (meilleur AUC) ──────
        if fold_metrics["auc"] > best_global_auc:
            best_global_auc = fold_metrics["auc"]
            try:
                model.save(str(OUTPUT_MODEL_PATH))
                print(
                    f"    ★ Meilleur modèle sauvegardé → {OUTPUT_MODEL_PATH.name} "
                    f"(AUC = {best_global_auc:.4f})"
                )
            except Exception as exc:
                fallback = OUTPUT_MODEL_PATH.with_suffix(".weights.h5")
                model.save_weights(str(fallback))
                print(
                    f"    ★ Poids sauvegardés → {fallback.name} "
                    f"(AUC = {best_global_auc:.4f}) [{exc}]"
                )

        # ── Libération mémoire GPU ───────────────────────────────────
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # ==================================================================
    #  RAPPORT FINAL
    # ==================================================================
    print(f"\n{'=' * 66}")
    print(f"  RAPPORT FINAL — {NUM_FOLDS}-Fold Cross-Validation Sim2Real")
    print(f"{'=' * 66}")

    for metric_name in ("accuracy", "f1", "auc"):
        values = [m[metric_name] for m in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        per_fold = "  ".join(f"{v:.3f}" for v in values)
        print(f"  {metric_name.upper():>10s} : {mean_val:.4f} ± {std_val:.4f}   "
              f"[{per_fold}]")

    val_losses = [m["val_loss"] for m in all_fold_metrics]
    print(f"  {'VAL_LOSS':>10s} : {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

    print(f"\n  Meilleur modèle (AUC = {best_global_auc:.4f}) → {OUTPUT_MODEL_PATH}")
    print(f"{'=' * 66}\n")


# ============================================================================
#  POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    try:
        run_kfold_finetuning()
    except (FileNotFoundError, ValueError) as fatal:
        print(f"\n[FATAL] {fatal}", file=sys.stderr)
        sys.exit(1)
