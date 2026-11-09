"""Choose the rows of a comparison, and pair the samples of the controlled axes."""
from __future__ import annotations

import pandas as pd


def pairing_keys(axis: str, experiment) -> list[str] | None:
    """What identifies a pair. Nothing for an observed axis: there are no pairs.

    The other controlled axes are part of the key: they stay fixed in the comparison
    (EN original against HI original, EN blurred against HI blurred).
    """
    if experiment.axes[axis].type == "observed":
        return None
    others = [name for name, spec in experiment.axes.items() if spec.type == "controlled" and name != axis]
    return ["sample_id", *others]


def select(cell: pd.DataFrame, axis: str, a: str, b: str, keys, statuses: list[str]):
    """Usable rows of levels a and b. Return (rows, number of excluded pairs)."""
    compared = cell[cell[axis].isin([a, b])]
    lines = compared[compared["status"].isin(statuses)]
    if keys is None:
        return lines, 0
    lines = lines.drop_duplicates([*keys, axis])
    used = lines[lines.groupby(keys)[axis].transform("nunique") == 2]
    excluded = len(compared.drop_duplicates(keys)) - len(used.drop_duplicates(keys))
    return used, excluded
