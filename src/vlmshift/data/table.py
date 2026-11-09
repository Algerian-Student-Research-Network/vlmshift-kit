"""Step 1: turn the dataset into a valid VLMShift table and write it."""
from __future__ import annotations

import pandas as pd

from ..common import Issues, Step, VLMShiftError, canonical, sha16, write_parquet
from . import checks
from .ids import row_ids, sample_ids
from .io import read_table


def build_table(config) -> Step:
    issues = Issues(step=1)
    axes = list(config.axes)
    controlled = [axis for axis, spec in config.axes.items() if spec.type == "controlled"]
    paths = config.data if isinstance(config.data, list) else [config.data]

    # Reading and standard names
    table = read_table([config.path(path) for path in paths])
    table = checks.apply_roles(table, config.roles)
    checks.require_columns(table, ["image", "label"], "the required roles")
    checks.require_columns(table, axes, "the axes")

    # Identifiers, computed on all rows so that each one can be reported
    table = checks.add_image_hash(table, issues)
    computed = pd.Series(sample_ids(table, controlled, config.task.columns()), index=table.index, dtype="string")
    table["sample_id"] = table["sample_id"].fillna(computed) if "sample_id" in table else computed
    table["unit"] = table["unit"].fillna(table["sample_id"]) if "unit" in table else table["sample_id"]
    table["id"] = row_ids(table)

    # Checks
    table = checks.drop_duplicates(table, issues)
    checks.check_collisions(table, issues)
    table = checks.drop_missing_images(table, issues)
    if table.empty:  # nothing can be measured: say why now, not at a later step
        issues.error("table_empty", "No row left: no image was found. Check the image paths in the table")
    checks.check_labels(table, issues)
    checks.check_original_id(table, issues)
    checks.check_samples(table, controlled, issues)
    if not table.empty:
        checks.check_axes(table, axes, issues)

    # Writing. The full path replaces the original path, once the ids are computed:
    # moving the dataset therefore changes neither the identifiers nor the fingerprint.
    table = table.assign(image=table["_path"]).drop(columns="_path")
    table = table.sort_values("id").reset_index(drop=True)
    fingerprint = sha16(canonical(table["id"].tolist()))
    folder = config.path(config.workspace) / "tables" / fingerprint
    write_parquet(issues.frame(), folder / "issues.parquet", fingerprint)
    if errors := issues.errors():
        details = "\n- ".join(errors)
        raise VLMShiftError(f"Table rejected (details: {folder / 'issues.parquet'}):\n- {details}")
    write_parquet(table, folder / "table.parquet", fingerprint)
    return Step(table, folder, fingerprint)
