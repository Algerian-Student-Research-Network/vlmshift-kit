"""Step 5: compute each declared shift measure and write the metrics table."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import Issues, VLMShiftError, code_version, to_levels, write_parquet
from ..config.load import config_hash, load_experiment, run_path, write_yaml
from ..config.schema import AxisSettings, Measure
from ..tasks.yes_no import negated_rows
from .analyses import check_analyses, run_analyses
from .bootstrap import draw_weights, summarize
from .metrics import stat_of, support, unit_sums
from .pairing import axis_levels, pairing_keys, select, split_cells, with_unit_codes
from .probes import run_probes
from .scores import score_shifts, score_stats
from .shifts import change_sums, delta_of

NOT_PROBED = {"id", "image", "image_hash", "unit", "sample_id", "prompt", "positive_prompt", "negative_prompt"}

# Each metric is computed twice (section 5.0).
STATUSES = {"valid": ["valid"], "strict": ["valid", "INVALID"]}
QUALITY = ["sens", "spec", "ba", "acc", "f1", "macro_f1", "mcc"]
BASE = {"valid": [*QUALITY, "yes_rate"], "strict": [*QUALITY, "inv_rate"]}
CHANGES = {"valid": ["chg", "chg_yes_no", "chg_no_yes"], "strict": ["chg"]}
METRIC_COLUMNS = [
    "measure", "model", "split", "level_a", "level_b", "metric", "mode", "value",
    "ci_low", "ci_high", "ci95_low", "ci95_high", "m", "significant",
    "n_lines", "n_samples", "n_units", "n_images", "n_valid", "n_invalid", "n_error", "n_excluded",
    # Where the number comes from (section 7.4): what it is averaged over, which analysis, which code.
    "aggregation", "run", "analysis", "code_version",
]


def make_rows(stats: dict, used: pd.DataFrame, settings, m: int, is_shift: bool, labels: dict):
    """One row per statistic. `stats`: name → (statistic, units it rests on)."""
    counts = {
        "n_lines": len(used),
        "n_samples": used["sample_id"].nunique(),
        "n_images": used["image_hash"].nunique() if "image_hash" in used else None,
        "n_valid": int((used["status"] == "valid").sum()),
        "n_invalid": int((used["status"] == "INVALID").sum()),
    }
    rows = []
    for name, (stat, n_units) in stats.items():
        enough = n_units >= max(settings.min_units, 2)  # a single unit cannot be drawn at random
        weights = draw_weights(used["unit"].nunique(), settings.bootstrap, settings.seed) if enough else None
        summary = summarize(stat, weights, settings.alpha, m, is_shift)
        rows.append({**labels, **counts, "n_units": n_units, "metric": name, "m": m, **summary})
    return rows


def base_rows(cell, axis, level, settings, labels) -> list[dict]:
    """Base metrics of a cell: all its rows, without pairing."""
    at_level = cell[cell[axis] == level]
    n_error = int((at_level["status"] == "error").sum())
    rows = []
    for mode, statuses in STATUSES.items():
        used, n_units = with_unit_codes(at_level[at_level["status"].isin(statuses)])
        sums = unit_sums(used, n_units)
        stats = {name: (stat_of(name, sums), support(name, sums)) for name in BASE[mode]}
        if mode == "valid":  # a tie has a margin but no decision: the score metrics need both
            stats |= score_stats(used, n_units)
        details = {**labels, "level_a": level, "level_b": None, "mode": mode, "n_error": n_error, "n_excluded": 0}
        rows += make_rows(stats, used, settings, 1, False, details)
    return rows


def shift_rows(cell, axis, a, b, keys, settings, m, labels) -> list[dict]:
    """Shift metrics between a and b. Strict mode: the pairs stay paired on T_c."""
    compared = cell[cell[axis].isin([a, b])]
    n_error = int((compared["status"] == "error").sum())
    rows = []
    for mode, statuses in STATUSES.items():
        lines, excluded = select(cell, axis, a, b, keys, statuses)
        used, n_units = with_unit_codes(lines)
        at_a, at_b = used[used[axis] == a], used[used[axis] == b]
        sums_a, sums_b = unit_sums(at_a, n_units), unit_sums(at_b, n_units)
        stats = {f"delta_{name}": (delta_of(name, sums_a, sums_b), support(name, sums_a, sums_b)) for name in BASE[mode]}
        if mode == "valid":
            stats |= score_shifts(at_a, at_b, n_units)
        if keys:  # an answer change only exists for a controlled axis
            sums = change_sums(at_a, at_b, keys, n_units)
            stats |= {name: (stat_of(name, sums), support(name, sums)) for name in CHANGES[mode]}
        details = {**labels, "level_a": a, "level_b": b, "mode": mode, "n_error": n_error, "n_excluded": excluded}
        rows += make_rows(stats, used, settings, m, True, details)
    return rows


def measure_rows(lines, measure, experiment, config, levels: dict, n_cells: int) -> list[dict]:
    """All the rows of a measure for one model: the matrix cell by cell.

    n_cells: the cells of the variants. A model that has not reached a cell yet gets the same
    correction as at the end of the run.
    """
    axis = measure.compare
    reference = config.axes.get(axis, AxisSettings()).reference
    if reference:
        pairs = [(reference, level) for level in levels[axis] if level != reference]
    else:
        pairs = list(combinations(levels[axis], 2))
    keys = pairing_keys(axis, experiment)
    cells = split_cells(lines, measure.split_by, levels)
    m = max(len(pairs) * n_cells, 1)  # Bonferroni: all cells included

    rows = []
    for split, cell in cells:
        labels = {"measure": measure.name, "model": lines["model"].iloc[0], "split": split,
                  "aggregation": " x ".join(["model", axis, *measure.split_by])}
        for level in levels[axis]:
            rows += base_rows(cell, axis, level, config.stats, labels)
        for a, b in pairs:
            rows += shift_rows(cell, axis, a, b, keys, config.stats, m, labels)
    return rows


def check_rules(config, experiment, measures, variants) -> None:
    """Rules that need the run (section 7.1)."""
    measured = {axis for measure in measures for axis in (measure.compare, *measure.split_by)}
    if unknown := ({*config.axes} | measured) - set(experiment.axes):
        raise VLMShiftError(f"Axes missing from the run's experiment.yaml: {sorted(unknown)}")
    # levels: [A] on an axis no measure uses would not restrict the analysis: say so, never ignore it.
    if idle := sorted(axis for axis, settings in config.axes.items() if settings.levels and axis not in measured):
        raise VLMShiftError(
            f"axes: no measure compares or splits by {idle}, so their levels would change nothing. "
            "Add a measure for them, or remove these levels"
        )
    for measure in measures:
        controlled = experiment.axes[measure.compare].type == "controlled"
        if controlled and variants[measure.compare].nunique() < 2:
            levels = sorted(variants[measure.compare].dropna().unique())
            raise VLMShiftError(
                f"Measure {measure.name}: the axis {measure.compare} has a single level in the variants "
                f"{levels}, so there is nothing to compare. Add the rows of the other levels to the table"
            )
    check_analyses(config, experiment)
    if config.latent:
        if not any(model.embeddings for model in experiment.models):
            raise VLMShiftError("latent block: no model of the run has embeddings: true")
        missing = [target for target in config.latent.targets if target not in variants.columns]
        if missing:
            raise VLMShiftError(f"latent.targets: columns not found: {missing}")
        if wrong := sorted(set(config.latent.targets) & NOT_PROBED):
            raise VLMShiftError(f"latent.targets: {wrong} cannot be probed: the probe reads a property of each image")


def run_analysis(config) -> Path:
    """Compute all the measures of analysis.yaml. Return the analysis folder."""
    run_dir = run_path(config)
    experiment = load_experiment(run_dir / "experiment.yaml")
    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    tables = run_dir.parents[1] / "tables"
    variants = pd.read_parquet(tables / log["table"] / "variants" / log["variants"] / "variants.parquet")
    table = pd.read_parquet(tables / log["table"] / "table.parquet")  # the quartiles are those of the whole table
    results = pd.read_parquet(run_dir / "results.parquet")

    measures = config.measures or [Measure(name=axis, compare=axis) for axis in experiment.axes]
    check_rules(config, experiment, measures, variants)
    issues = Issues(step=5)
    for axis, spec in experiment.axes.items():
        empty = int(variants[axis].isna().sum())
        if empty:  # the audit shows them as (empty): say why they are not in the numbers
            issues.warn("axis_empty", f"Axis {axis}: {empty} variants without a value are left out of the measures")
        # An observed numeric axis is compared by quartiles, as in the audit.
        variants[axis] = to_levels(variants[axis], spec.type == "observed", table[axis]).where(variants[axis].notna())

    axis_levels(variants, config, list(config.axes))  # a reference absent from the data, even on an unused axis
    # One row per variant × model, with label and decision as numbers.
    lines = results.merge(variants, on="id")
    lines["y"] = lines["label"].map({"positive": 1.0, "negative": 0.0}).astype(float)
    lines["yhat"] = lines["decision"].map({"yes": 1.0, "no": 0.0}).astype(float)
    if experiment.task.negated:
        # A yes to a negated question denies the finding. The answer is counted for what it says
        # about the finding, so it compares with the label and with the plain question.
        negated = np.array(negated_rows(lines[experiment.task.negated]), dtype=bool)
        lines["yhat"] = lines["yhat"].where(~negated, 1 - lines["yhat"])
        lines["margin"] = lines["margin"].where(~negated, -lines["margin"])

    rows = []
    for measure in measures:
        levels = axis_levels(variants, config, [measure.compare, *measure.split_by])
        n_cells = len(split_cells(variants, measure.split_by, levels))
        for _, model_lines in lines.groupby("model"):
            rows += measure_rows(model_lines, measure, experiment, config, levels, n_cells)
    for model, model_lines in lines.groupby("model"):
        if model_lines["decision"].nunique() == 1:
            issues.warn("constant_model", f"{model} always gives the same answer")

    probes = pd.DataFrame()
    if config.latent and (run_dir / "embeddings.parquet").exists():
        embeddings = pd.read_parquet(run_dir / "embeddings.parquet")
        probes = run_probes(variants, embeddings, config.latent, config.stats, issues)
    elif config.latent:  # partial run: the model that produces vectors has not run yet
        issues.warn("probe_skipped", "Probe not computed: this run does not contain any vectors yet")

    further = run_analyses(config, experiment, lines, variants, probes, issues)

    fingerprint = config_hash(config, run_dir.name)
    folder = run_dir / "analyses" / fingerprint
    write_yaml(config.model_dump(), folder / "analysis.yaml")
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS).assign(
        run=run_dir.name, analysis=fingerprint, code_version=code_version())
    write_parquet(metrics, folder / "metrics.parquet", fingerprint)
    stamp = dict(run=run_dir.name, analysis=fingerprint, code_version=code_version())
    write_parquet(further.assign(**stamp), folder / "analyses.parquet", fingerprint)
    write_parquet(probes, folder / "probes.parquet", fingerprint)
    write_parquet(issues.frame(), folder / "issues.parquet", fingerprint)
    return folder
