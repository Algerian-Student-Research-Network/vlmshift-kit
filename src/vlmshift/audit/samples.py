"""Variants per sample and incomplete samples."""
import pandas as pd


def combinations(lines: pd.DataFrame, axes: list[str]) -> set[tuple]:
    return set(map(tuple, lines[axes].dropna().to_numpy()))


def sample_table(table: pd.DataFrame, controlled_axes: list[str]) -> pd.DataFrame:
    """One row per sample: its number of variants and the combinations of levels it lacks.

    With two controlled axes, a sample can have each level of each axis and still lack
    one combination, such as FR with blur: then that pair is missing.
    """
    expected = combinations(table, controlled_axes)
    rows = []
    for sample, lines in table.groupby("sample_id"):
        missing = [
            ", ".join(f"{axis}={level}" for axis, level in zip(controlled_axes, combination))
            for combination in sorted(expected - combinations(lines, controlled_axes))
        ]
        rows.append({
            "sample_id": sample,
            "n_variants": len(lines),
            "missing": "; ".join(missing) or None,
        })
    return pd.DataFrame(rows)
