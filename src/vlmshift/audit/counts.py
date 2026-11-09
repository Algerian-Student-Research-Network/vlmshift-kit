"""Step 2: describe how the table is distributed, without changing it."""
from __future__ import annotations

from itertools import product

import pandas as pd

from ..common import MIN_UNITS, Issues, Step, to_levels, write_csv, write_parquet
from .samples import sample_table


def count_cells(table: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Counts of each combination of levels, including empty cells."""
    groups = dict(iter(table.groupby(columns)))
    rows = []
    for values in product(*(sorted(table[column].unique()) for column in columns)):
        lines = groups.get(values, table.iloc[0:0])
        positive, negative = (lines["label"] == "positive").sum(), (lines["label"] == "negative").sum()
        rows.append({
            "axis": " x ".join(columns),
            "level": " x ".join(values),
            "n_lines": len(lines),
            "n_samples": lines["sample_id"].nunique(),
            "n_units": lines["unit"].nunique(),
            "n_positive": int(positive),
            "n_negative": int(negative),
            "n_unknown": int((lines["label"] == "unknown").sum()),
            "positive_rate": positive / (positive + negative) if positive + negative else None,
        })
    return rows


def run_audit(config, table: Step) -> pd.DataFrame:
    """Write the audit in audit/: each axis alone, then all axes crossed."""
    axes = list(config.axes)
    view = table.df.assign(**{
        axis: to_levels(table.df[axis], spec.type == "observed") for axis, spec in config.axes.items()
    })
    groups = [[axis] for axis in axes] + ([axes] if len(axes) > 1 else [])
    counts = pd.DataFrame([row for columns in groups for row in count_cells(view, columns)])

    issues = Issues(step=2)
    for row in counts[counts["n_units"] < MIN_UNITS].itertuples():
        issues.warn("low_count", f"{row.axis} = {row.level}: {row.n_units} units (threshold {MIN_UNITS})")

    controlled = [axis for axis, spec in config.axes.items() if spec.type == "controlled"]
    folder = table.dir / "audit"
    samples = sample_table(table.df, controlled)
    if incomplete := int(samples["missing"].notna().sum()):
        issues.warn("samples_incomplete", (
            f"{incomplete} of {len(samples)} samples lack a level, or a combination of levels, of the controlled axes: "
            "the pairs that need it are missing. The rows of a sample are the same except for the question, "
            "the columns of the templates and the controlled axes: another column that differs between them splits the sample"
        ))
    write_parquet(counts, folder / "counts.parquet", table.hash)
    write_parquet(samples, folder / "samples.parquet", table.hash)
    # The same tables for a spreadsheet: the audit is meant to be read before any model runs.
    write_csv(counts, folder / "counts.csv", issues)
    write_csv(samples, folder / "samples.csv", issues)
    write_parquet(issues.frame(), folder / "issues.parquet", table.hash)
    return counts
