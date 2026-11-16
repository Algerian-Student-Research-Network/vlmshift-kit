"""Labeling rule of a source. One rule per source: each one codes its labels in its own way.

The concepts are those of configs/questions.yaml, and each rule follows the label policy the paper
gives to its source (configs/datasets.yaml of the study code):
- a source whose classes exclude each other (normal or tuberculosis) gives a positive for the class
  of the image and a negative for the others: `one_class`;
- a source that only shows positives gives no negative at all;
- anything the source does not state stays unknown. Unknown is never turned into negative.
"""


def to_label(value: str) -> str:
    """1 → positive, 0 → negative. Everything else (uncertain, empty) → unknown, never negative."""
    return {"1": "positive", "0": "negative"}.get(value, "unknown")


def one_class(concept: str | None, concepts) -> dict[str, str]:
    """An image of a source whose classes exclude each other: positive for its class, negative
    for the others. A class outside the concepts of the paper (None) states nothing: all unknown."""
    if concept is None:
        return {name: "unknown" for name in concepts}
    return {name: "positive" if name == concept else "negative" for name in concepts}
