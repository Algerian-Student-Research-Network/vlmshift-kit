"""The further analyses of analysis.yaml (the `analyses` block): one long table, analyses.parquet.

Each analysis has a type. The measures answer "how much does the answer shift between two
levels"; these answer the questions around it: is the shift significant on paired samples, does
it depend on another axis, does the order of the models hold, what happens in the worst group.

Every row has the same columns, whatever the analysis: its name, the model, the cell, what is
compared, a statistic and its value, then an interval or a p-value when there is one. The
analyses not yet validated on real runs are marked experimental, in the table and in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import VLMShiftError
from ..config.schema import AxisSettings
from .effects import consistency_rows, disagreement_shift_rows, information_rows, interaction_rows, taxonomy_rows
from .paired import holm, level_pairs, paired_tests
from .pairing import axis_levels, split_cells
from .regression import regression_rows, variance_rows
from .robustness import (
    heterogeneity_rows, link_rows, pareto_rows, parser_rows, ranking_rows, standardized_rows, worst_group_rows,
)

ANALYSIS_COLUMNS = [
    "name", "type", "experimental", "model", "split", "level_a", "level_b", "statistic", "value",
    "ci_low", "ci_high", "ci95_low", "ci95_high", "significant", "p", "p_holm", "n", "n_units", "note",
]
# Validated on real runs and listed as such in the design: the others stay experimental.
STABLE = {"tests", "consistency", "ranking", "worst_group", "standardized", "parser"}
ALL_MODELS = "(all models)"
# The axes an analysis compares on paired samples: they must be controlled.
PAIRED = {"tests": ["compare"], "consistency": ["compare"], "disagreement_shift": ["compare", "across"],
          "taxonomy": ["compare", "across"], "pareto": ["compare"]}
# The axes that need a reference level in analysis.yaml.
REFERENCED = {"interaction": ["axes"], "disagreement_shift": ["across"], "taxonomy": ["across"]}


def axes_of(analysis) -> list[str]:
    """Every axis an analysis names, in any of its keys."""
    single = [axis for axis in (analysis.compare, analysis.across, analysis.over) if axis]
    return list(dict.fromkeys([*single, *analysis.axes, *analysis.split_by, *analysis.by, *analysis.covariates]))


def named(analysis, key: str) -> list[str]:
    value = getattr(analysis, key)
    return value if isinstance(value, list) else [value]


def check_analyses(config, experiment) -> None:
    """What can be wrong in the analyses block is said before any computation."""
    for analysis in config.analyses:
        where = f"analyses, {analysis.label}"
        if unknown := sorted(set(axes_of(analysis)) - set(experiment.axes)):
            raise VLMShiftError(f"{where}: axes missing from the run's experiment.yaml: {unknown}")
        for key in PAIRED.get(analysis.type, []):
            if observed := [axis for axis in named(analysis, key) if experiment.axes[axis].type == "observed"]:
                raise VLMShiftError(
                    f"{where}: {key} must be a controlled axis, and {observed} is observed. "
                    "Only a controlled axis asks the same sample at each level, which is what this analysis pairs"
                )
        for key in REFERENCED.get(analysis.type, []):
            if missing := [axis for axis in named(analysis, key) if not config.axes.get(axis, AxisSettings()).reference]:
                raise VLMShiftError(f"{where}: give a reference level to {missing} in the axes block: the effect is measured against it")
        if analysis.type == "link":
            if not config.latent or analysis.target not in config.latent.targets:
                raise VLMShiftError(f"{where}: target {analysis.target!r} must be one of latent.targets")
            if analysis.of not in [other.label for other in config.analyses if other.type == "interaction"]:
                raise VLMShiftError(f"{where}: of {analysis.of!r} must name an interaction analysis")


def sample_keys(experiment, compared: list[str]) -> list[str]:
    """What identifies a sample to pair: its id, and the controlled axes that are not compared."""
    return ["sample_id", *[axis for axis, spec in experiment.axes.items()
                           if spec.type == "controlled" and axis not in compared]]


def within(lines: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """The rows at the levels kept for each axis."""
    for axis, kept in levels.items():
        lines = lines[lines[axis].isin(kept)]
    return lines


def corrected(rows: list[dict], statistics: list[str]) -> list[dict]:
    """Holm's correction over all the tests of the same kind of one model, cells included."""
    for prefix in statistics:
        tested = [row for row in rows if str(row.get("statistic", "")).startswith(prefix)]
        for row, adjusted in zip(tested, holm([row.get("p", np.nan) for row in tested])):
            row["p_holm"] = adjusted
    return rows


def per_model(lines: pd.DataFrame, split_by, levels, compute) -> list[dict]:
    """Apply `compute(cell)` to each cell of each model, and label its rows."""
    rows = []
    for model, model_lines in lines.groupby("model"):
        found = []
        for split, cell in split_cells(model_lines, split_by, levels):
            found += [{"split": split, **row} for row in compute(cell)]
        rows += [{"model": model, **row} for row in found]
    return rows


def analysis_rows(analysis, config, experiment, lines, variants, probes, earlier) -> list[dict]:
    """The rows of one analysis, for every model."""
    kind, stats = analysis.type, config.stats
    levels = axis_levels(variants, config, axes_of(analysis))
    lines = within(lines, {axis: levels[axis] for axis in axes_of(analysis) if axis not in analysis.split_by})
    reference = {axis: config.axes.get(axis, AxisSettings()).reference for axis in levels}
    n_cells = len(split_cells(variants, analysis.split_by, levels))
    compare, across = analysis.compare, analysis.across
    pairs = level_pairs(levels[compare], None if analysis.all_pairs else reference[compare]) if compare else []
    conditions = [level for level in levels[across] if level != reference[across]] if across else []

    if kind == "tests":
        keys = sample_keys(experiment, [compare])
        rows = per_model(lines, analysis.split_by, levels,
                         lambda cell: paired_tests(cell, compare, pairs, keys, stats.alpha, analysis.power))
        by_model: dict[str, list[dict]] = {}
        for row in rows:
            by_model.setdefault(row["model"], []).append(row)
        kinds = ["mcnemar_correct", "mcnemar_answer", "cochran_q_correct", "cochran_q_answer"]
        return [row for part in by_model.values() for row in corrected(part, kinds)]
    if kind == "consistency":
        keys = sample_keys(experiment, [compare])
        return per_model(lines, analysis.split_by, levels,
                         lambda cell: consistency_rows(cell, compare, levels[compare], keys, stats, n_cells))
    if kind == "information":
        return per_model(lines, analysis.split_by, levels, lambda cell: information_rows(cell, compare, levels[compare]))
    if kind == "interaction":
        axes = analysis.axes
        controlled = [axis for axis in axes if experiment.axes[axis].type == "controlled"]
        keys = sample_keys(experiment, axes)
        m = n_cells * int(np.prod([len(levels[axis]) - 1 for axis in axes]))
        return per_model(lines, analysis.split_by, levels, lambda cell: interaction_rows(
            cell, axes, reference, levels, analysis.metric, controlled, keys, stats, max(m, 1)))
    if kind in ("disagreement_shift", "taxonomy"):
        keys = sample_keys(experiment, [compare, across])
        if kind == "taxonomy":
            return per_model(lines, analysis.split_by, levels, lambda cell: taxonomy_rows(
                cell, compare, across, reference[across], conditions, keys, stats, max(n_cells * len(conditions), 1)))
        m = max(n_cells * len(pairs) * len(conditions), 1)
        return per_model(lines, analysis.split_by, levels, lambda cell: disagreement_shift_rows(
            cell, compare, across, reference[across], pairs, conditions, keys, stats, m))
    if kind == "ranking":
        return [{"model": ALL_MODELS, "split": split, **row}
                for split, cell in split_cells(lines, analysis.split_by, levels)
                for row in ranking_rows(cell, compare, pairs, analysis.metric, stats.min_units)]
    if kind == "worst_group":
        return per_model(lines, analysis.split_by, levels,
                         lambda cell: worst_group_rows(cell, analysis.by, analysis.metric, analysis.tail, stats.min_units))
    if kind == "standardized":
        return per_model(lines, analysis.split_by, levels, lambda cell: standardized_rows(cell, analysis.by, analysis.over))
    if kind == "heterogeneity":
        return per_model(lines, analysis.split_by, levels, lambda cell: heterogeneity_rows(
            cell, analysis.by, analysis.over, analysis.metric, stats.min_units))
    if kind == "pareto":
        keys = sample_keys(experiment, [compare])
        return [{"split": split, **row} for split, cell in split_cells(lines, analysis.split_by, levels)
                for row in pareto_rows(cell, analysis.by, compare, keys, analysis.metric, stats.min_units)]
    if kind == "regression":
        return per_model(lines, analysis.split_by, levels,
                         lambda cell: regression_rows(cell, analysis.axes, analysis.covariates, reference))
    if kind == "parser":
        return per_model(lines, analysis.split_by, levels, lambda cell: parser_rows(cell, analysis.by, experiment.task))
    if kind == "variance":
        return per_model(lines, analysis.split_by, levels, lambda cell: variance_rows(cell, analysis.axes))
    return link_rows(probes, earlier, analysis.target, analysis.of)


def run_analyses(config, experiment, lines, variants, probes, issues) -> pd.DataFrame:
    """Every analysis of the block, as one table. An analysis that finds nothing to compute says so."""
    rows = []
    for analysis in config.analyses:
        found = analysis_rows(analysis, config, experiment, lines, variants, probes, rows)
        if not found:
            issues.warn("analysis_empty", f"Analysis {analysis.label}: nothing could be computed, for lack of matching answers")
        rows += [{"name": analysis.label, "type": analysis.type, "experimental": analysis.type not in STABLE, **row}
                 for row in found]
    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS).astype({"experimental": "boolean", "significant": "boolean"})
