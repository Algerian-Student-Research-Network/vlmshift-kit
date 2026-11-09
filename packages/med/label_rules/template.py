"""Labeling rule of a source. One rule per source: each one codes its labels in its own way."""


def to_label(value: str) -> str:
    """1 → positive, 0 → negative. Everything else (uncertain, empty) → unknown, never negative."""
    return {"1": "positive", "0": "negative"}.get(value, "unknown")
