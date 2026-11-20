"""Choose the rows of a comparison: levels, cells, units, and the pairs of the controlled axes."""
from __future__ import annotations

import pandas as pd

from ..common import VLMShiftError
from ..config.schema import AxisSettings


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


def with_unit_codes(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Number the units from 0 to n − 1."""
    codes, units = pd.factorize(lines["unit"])
    return lines.assign(u=codes), len(units)


def split_cells(lines: pd.DataFrame, split_by: list[str], levels: dict) -> list[tuple]:
    """Split the rows according to split_by. Return (cell name, rows)."""
    for axis in split_by:
        lines = lines[lines[axis].isin(levels[axis])]
    if not split_by:
        return [(None, lines)]
    return [
        (", ".join(f"{axis}={value}" for axis, value in zip(split_by, values)), cell)
        for values, cell in lines.groupby(split_by)
    ]


def axis_levels(variants: pd.DataFrame, config, axes: list[str]) -> dict:
    """Levels kept for each axis. Check that they were really sent to the models.

    We look at the variants, not the results: an interrupted run keeps all its levels.
    """
    levels = {}
    for axis in axes:
        present = sorted(variants[axis].dropna().unique())
        settings = config.axes.get(axis, AxisSettings())
        wanted = [*(settings.levels or present), *([settings.reference] if settings.reference else [])]
        absent = sorted(set(wanted) - set(present))
        if absent:
            raise VLMShiftError(f"Axis {axis}: levels missing from the data: {absent}")
        levels[axis] = settings.levels or present
    return levels
