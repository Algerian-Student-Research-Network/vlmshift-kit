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


ONE_SIDED = 0.1  # below this share of one class, a level has too few of its cases to measure it
NOT_DATA = {"id", "image_hash", "sample_id", "unit"}  # written by the tool: never empty


def column_table(table: pd.DataFrame) -> pd.DataFrame:
    """For each column of the dataset: how many rows leave it empty, and how many values it takes."""
    rows = [{"column": column, "n_empty": int(table[column].isna().sum()),
             "share_empty": float(table[column].isna().mean()), "n_values": int(table[column].nunique())}
            for column in table.columns if column not in NOT_DATA]
    return pd.DataFrame(rows, columns=["column", "n_empty", "share_empty", "n_values"])


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

    known = counts["n_positive"] + counts["n_negative"]
    one_sided = counts[(known >= MIN_UNITS) & ((counts["positive_rate"] < ONE_SIDED) | (counts["positive_rate"] > 1 - ONE_SIDED))]
    for row in one_sided.itertuples():
        issues.warn("label_imbalance", (
            f"{row.axis} = {row.level}: {row.n_positive} positive and {row.n_negative} negative labels. "
            "Sensitivity or specificity will rest on few cases there: read the balanced accuracy with its interval"
        ))
    columns = column_table(table.df)
    for row in columns[columns["n_empty"] > 0].itertuples():
        issues.warn("column_empty", f"Column {row.column}: {row.n_empty} empty rows ({row.share_empty:.0%})")

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
    write_parquet(columns, folder / "columns.parquet", table.hash)
    # The same tables for a spreadsheet: the audit is meant to be read before any model runs.
    write_csv(counts, folder / "counts.csv", issues)
    write_csv(samples, folder / "samples.csv", issues)
    write_csv(columns, folder / "columns.csv", issues)
    write_parquet(issues.frame(), folder / "issues.parquet", table.hash)
    return counts
