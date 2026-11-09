"""Row and sample identifiers (sections 3 and 7.4)."""
from __future__ import annotations

import pandas as pd

from ..common import canonical, nfc, sha16

# Columns that do not count when recognizing a sample (section 3).
NOT_IN_SAMPLE = {"id", "original_id", "question", "positive_text", "negative_text", "_path"}


def text_of(row: dict, skip: set[str]) -> str:
    """A row as one text: sorted columns, NFC text, null kept."""
    items = [
        [column, None if pd.isna(value) else nfc(str(value))]
        for column, value in sorted(row.items())
        if column not in skip
    ]
    return canonical(items)


def digest(row: dict, skip: set[str]) -> str:
    """Fingerprint of a row."""
    return sha16(text_of(row, skip))


def content(table: pd.DataFrame, skip: set[str]) -> list[str]:
    """Each row as one text, to compare rows exactly."""
    return [text_of(row, skip) for row in table.to_dict("records")]


def sample_ids(table: pd.DataFrame, controlled_axes: list[str], question_columns=()) -> list[str]:
    """Two rows belong to the same sample if only the question and the controlled axes change.

    The columns of the templates are part of the question, such as {concept} translated in each
    language. Except when leaving them out would put two rows of one sample at the same levels
    of the controlled axes, as with several concepts asked about one image: there they tell the
    questions apart, and stay in the sample.
    """
    skip = NOT_IN_SAMPLE | set(controlled_axes)
    coarse = skip | set(question_columns)
    records = table.to_dict("records")
    ids = ["s_" + digest(row, coarse) for row in records]
    if not (question_columns and controlled_axes):
        return ids
    levels = table[list(controlled_axes)].assign(_sample=ids)
    unclear = set(levels.loc[levels.duplicated(keep=False), "_sample"])
    return ["s_" + digest(row, skip) if sample in unclear else sample for row, sample in zip(records, ids)]


def row_ids(table: pd.DataFrame) -> list[str]:
    """The identifier comes from the content: reordering the file does not change it."""
    return ["l_" + digest(row, {"id", "_path"}) for row in table.to_dict("records")]
