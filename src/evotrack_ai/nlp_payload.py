"""Build safe NLP payloads from extracted EvoTrack AI metrics."""

from pprint import pprint


STABILITY_SURFACE_THRESHOLD = 0
SAFETY_NOTE = "Résumé descriptif uniquement, sans diagnostic autonome."


def normalize_centroid(centroid):
    """Normalize a centroid into a serializable dictionary.

    Args:
        centroid: ``None`` or a tuple/list containing ``(x, y)``.

    Returns:
        ``None`` or ``{"x": int, "y": int}``.

    Raises:
        ValueError: If the centroid has an invalid format.
    """
    if centroid is None:
        return None

    if isinstance(centroid, (tuple, list)) and len(centroid) == 2:
        x, y = centroid
        return {"x": int(x), "y": int(y)}

    raise ValueError("centroid must be None or a tuple/list of two values.")


def build_nlp_payload(
    status: str,
    surface_pixels: int,
    surface_category: str,
    intensity_mean: float,
    intensity_category: str,
    location: str,
    centroid=None,
) -> dict:
    """Build a structured payload for the NLP summary engine.

    Args:
        status: Evolution status from the heatmap metrics.
        surface_pixels: Changed surface in pixels.
        surface_category: Human-readable surface category.
        intensity_mean: Mean signal intensity in the changed area.
        intensity_category: Human-readable intensity category.
        location: Human-readable location description.
        centroid: Optional centroid coordinates.

    Returns:
        A safe dictionary ready for NLP processing.
    """
    if surface_pixels <= STABILITY_SURFACE_THRESHOLD:
        return {
            "status": "stabilité",
            "surface": {
                "pixels": 0,
                "category": "absente",
            },
            "signal": {
                "mean_intensity": 0.0,
                "category": "nulle",
            },
            "location": {
                "description": "aucune localisation significative",
                "centroid": None,
            },
            "safety_note": SAFETY_NOTE,
        }

    return {
        "status": "évolution",
        "surface": {
            "pixels": int(surface_pixels),
            "category": str(surface_category),
        },
        "signal": {
            "mean_intensity": float(intensity_mean),
            "category": str(intensity_category),
        },
        "location": {
            "description": str(location),
            "centroid": normalize_centroid(centroid),
        },
        "safety_note": SAFETY_NOTE,
    }


def build_payload_from_metrics(metrics: dict) -> dict:
    """Build an NLP payload from an ``extract_heatmap_metrics`` dictionary.

    Args:
        metrics: Metrics dictionary produced by heatmap metric extraction.

    Returns:
        Structured NLP payload.

    Raises:
        ValueError: If a required key is missing.
    """
    required_keys = {
        "status",
        "surface_pixels",
        "surface_category",
        "intensity_mean",
        "intensity_category",
        "location",
        "centroid",
    }
    missing_keys = required_keys - set(metrics.keys())

    if missing_keys:
        missing_keys_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required metric keys: {missing_keys_text}")

    return build_nlp_payload(
        status=metrics["status"],
        surface_pixels=metrics["surface_pixels"],
        surface_category=metrics["surface_category"],
        intensity_mean=metrics["intensity_mean"],
        intensity_category=metrics["intensity_category"],
        location=metrics["location"],
        centroid=metrics["centroid"],
    )


if __name__ == "__main__":
    metrics_evolution = {
        "status": "évolution",
        "surface_pixels": 1500,
        "surface_category": "massive",
        "intensity_mean": 180.5,
        "intensity_category": "haute intensité",
        "centroid": (170, 40),
        "location": "quadrant supérieur droit",
    }
    metrics_stability = {
        "status": "stabilité",
        "surface_pixels": 0,
        "surface_category": "absente",
        "intensity_mean": 0.0,
        "intensity_category": "nulle",
        "centroid": None,
        "location": "aucune localisation significative",
    }

    payload_evolution = build_payload_from_metrics(metrics_evolution)
    payload_stability = build_payload_from_metrics(metrics_stability)

    pprint(payload_evolution)
    pprint(payload_stability)

    assert type(payload_evolution) is dict
    assert payload_evolution["status"] == "évolution"
    assert payload_evolution["surface"]["pixels"] == 1500
    assert payload_stability["status"] == "stabilité"
    assert payload_stability["surface"]["pixels"] == 0
