"""Dermatology keyword filter for ChatDoctor patient messages."""

from __future__ import annotations

DERMATOLOGY_KEYWORDS: tuple[str, ...] = (
    "acne",
    "biopsy",
    "blister",
    "cellulitis",
    "dermatitis",
    "dermatology",
    "eczema",
    "hives",
    "itch",
    "itchy",
    "lesion",
    "lip swelling",
    "melanoma",
    "mole",
    "pigmented",
    "psoriasis",
    "rash",
    "rosacea",
    "scaly",
    "skin",
    "swelling",
    "tongue swelling",
    "urticaria",
    "wart",
    "wound",
)

MIN_MESSAGE_LENGTH = 20
MAX_MESSAGE_LENGTH = 1500


def find_matched_keywords(
    text: str, keywords: tuple[str, ...] = DERMATOLOGY_KEYWORDS
) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]


def passes_derm_filter(
    text: str, keywords: tuple[str, ...] = DERMATOLOGY_KEYWORDS
) -> bool:
    return bool(find_matched_keywords(text, keywords))


def normalize_message(text: str) -> str:
    return " ".join(text.split())


def preprocess_input(
    text: str,
    *,
    min_length: int = MIN_MESSAGE_LENGTH,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> str | None:
    cleaned = normalize_message(text)
    if len(cleaned) < min_length:
        return None
    if len(cleaned) <= max_length:
        return cleaned
    truncated = cleaned[:max_length]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated if len(truncated) >= min_length else None
