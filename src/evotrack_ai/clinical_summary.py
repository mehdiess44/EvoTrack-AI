"""Generate cautious French clinical summaries from structured payloads."""


DEFAULT_MODEL_NAME = "google/flan-t5-small"


def validate_payload(payload: dict) -> None:
    """Validate the minimal structure expected by the summary generator.

    Args:
        payload: Structured dictionary built for NLP generation.

    Raises:
        ValueError: If the payload is invalid or missing required keys.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dictionary.")

    required_keys = {"status", "surface", "signal", "location", "safety_note"}
    missing_keys = required_keys - set(payload.keys())

    if missing_keys:
        missing_keys_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"payload is missing required keys: {missing_keys_text}")


def payload_to_prompt(payload: dict) -> str:
    """Convert a structured payload into a safe French generation prompt."""
    validate_payload(payload)

    surface = payload["surface"]
    signal = payload["signal"]
    location = payload["location"]

    return (
        "Rédige une phrase clinique descriptive en français, simple et prudente. "
        "Ne formule aucune hypothèse diagnostique, n'utilise aucun terme alarmiste, "
        "et ne produis pas de diagnostic autonome. "
        f"Statut: {payload['status']}. "
        f"Surface: {surface['category']} ({surface['pixels']} pixels). "
        f"Signal: {signal['category']} "
        f"(intensité moyenne {signal['mean_intensity']}). "
        f"Localisation: {location['description']}. "
        f"Note de sécurité: {payload['safety_note']}"
    )


def load_text_generation_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
):
    """Load a Hugging Face text2text-generation pipeline.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        A Transformers pipeline.

    Raises:
        RuntimeError: If the pipeline cannot be loaded.
    """
    try:
        from transformers import pipeline

        return pipeline("text2text-generation", model=model_name)
    except Exception as error:
        raise RuntimeError(
            f"Could not load Hugging Face model '{model_name}'."
        ) from error


def clean_generated_text(text: str) -> str:
    """Clean generated text and remove raw JSON-looking characters."""
    cleaned_text = str(text).replace("\n", " ")
    cleaned_text = cleaned_text.replace("{", "").replace("}", "")
    cleaned_text = " ".join(cleaned_text.split())

    if not cleaned_text:
        return "Résumé non disponible."

    return cleaned_text


def deterministic_summary_from_payload(payload: dict) -> str:
    """Build a safe summary without using a Transformers model."""
    validate_payload(payload)

    safety_sentence = (
        "Ce résumé est descriptif et ne constitue pas un diagnostic autonome."
    )

    if payload["status"] == "stabilité":
        return (
            "Aucune évolution significative n'est mise en évidence sur la carte "
            f"de différence. {safety_sentence}"
        )

    surface_category = payload["surface"]["category"]
    intensity_category = payload["signal"]["category"]
    location = payload["location"]["description"]

    return (
        f"Évolution {surface_category} localisée dans le {location}, "
        f"avec un signal {intensity_category}. {safety_sentence}"
    )


def generate_clinical_summary(
    payload: dict,
    generator=None,
    max_new_tokens: int = 80,
    use_fallback_on_error: bool = True,
) -> str:
    """Generate a cautious French clinical summary from a payload.

    Args:
        payload: Structured NLP payload.
        generator: Optional Hugging Face generator pipeline.
        max_new_tokens: Maximum number of generated tokens.
        use_fallback_on_error: Use deterministic summary if generation fails.

    Returns:
        A descriptive summary string without raw JSON braces.
    """
    validate_payload(payload)

    try:
        if generator is None:
            generator = load_text_generation_pipeline()

        prompt = payload_to_prompt(payload)
        result = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_text = result[0].get("generated_text", "")
        summary = clean_generated_text(generated_text)

        if len(summary) < 15:
            return deterministic_summary_from_payload(payload)

        return summary
    except Exception:
        if use_fallback_on_error:
            return deterministic_summary_from_payload(payload)

        raise


if __name__ == "__main__":
    payload_evolution = {
        "status": "évolution",
        "surface": {"pixels": 1500, "category": "massive"},
        "signal": {"mean_intensity": 180.5, "category": "haute intensité"},
        "location": {
            "description": "quadrant supérieur droit",
            "centroid": {"x": 170, "y": 40},
        },
        "safety_note": "Résumé descriptif uniquement, sans diagnostic autonome.",
    }
    payload_stability = {
        "status": "stabilité",
        "surface": {"pixels": 0, "category": "absente"},
        "signal": {"mean_intensity": 0.0, "category": "nulle"},
        "location": {
            "description": "aucune localisation significative",
            "centroid": None,
        },
        "safety_note": "Résumé descriptif uniquement, sans diagnostic autonome.",
    }

    summary_evolution = generate_clinical_summary(payload_evolution)
    summary_stability = generate_clinical_summary(payload_stability)

    print(f"Résumé évolution: {summary_evolution}")
    print(f"Résumé stabilité: {summary_stability}")

    assert type(summary_evolution) is str
    assert len(summary_evolution) > 15
    assert "{" not in summary_evolution
    assert "}" not in summary_evolution
    assert type(summary_stability) is str
    assert len(summary_stability) > 15
