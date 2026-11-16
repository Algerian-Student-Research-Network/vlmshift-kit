"""How a model holds across groups: ranking, worst group, prevalence, spread, trade-offs.

These analyses compare numbers computed group by group (a site, a language, a concept). They
describe; they carry no interval. A group that rests on fewer units than stats.min_units is
left out, and each row says how many groups it rests on.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .effects import decisions
from .metrics import METRICS, support, unit_sums, weigh
from .pairing import with_unit_codes


def point(lines: pd.DataFrame, metric: str, min_units: int) -> float:
    """A metric on the valid answers of some rows. NaN when it rests on too few units."""
    used, n_units = with_unit_codes(lines[lines["status"] == "valid"])
    sums = unit_sums(used, n_units)
    if support(metric, sums) < min_units:
        return float("nan")
    return float(METRICS[metric](weigh(sums, np.ones((1, n_units))))[0])


def group_values(lines: pd.DataFrame, by: list[str], metric: str, min_units: int) -> pd.Series:
    """The metric of each group, named `axis=level, axis=level`. Groups not estimable are dropped."""
    values = {}
    for levels, group in lines.groupby(by):
        levels = levels if isinstance(levels, tuple) else (levels,)
        values[", ".join(f"{axis}={level}" for axis, level in zip(by, levels))] = point(group, metric, min_units)
    return pd.Series(values, dtype=float).dropna()


def ranking_rows(cell: pd.DataFrame, axis: str, pairs, metric: str, min_units: int) -> list[dict]:
    """Does the order of the models change between two levels? Kendall's τ over the models, the
    number of pairs of models whose order flips, and whether the best model stays the same."""
    table = pd.DataFrame({
        model: {level: point(group[group[axis] == level], metric, min_units) for level in group[axis].dropna().unique()}
        for model, group in cell.groupby("model")
    })
    rows = []
    for a, b in pairs:
        if a not in table.index or b not in table.index:
            continue
        both = table.loc[[a, b]].dropna(axis=1)
        if both.shape[1] < 3:  # an order of two models says nothing
            continue
        tau, p = kendalltau(both.loc[a], both.loc[b])
        flips = sum((both.loc[a, m1] - both.loc[a, m2]) * (both.loc[b, m1] - both.loc[b, m2]) < 0
                    for m1, m2 in combinations(both.columns, 2))
        best_a, best_b = both.loc[a].idxmax(), both.loc[b].idxmax()
        shared = {"level_a": a, "level_b": b, "n": both.shape[1]}
        rows += [
            {**shared, "statistic": "kendall_tau", "value": float(tau), "p": float(p), "note": f"metric {metric}, n = models"},
            {**shared, "statistic": "rank_reversals", "value": float(flips), "note": "pairs of models whose order flips"},
            {**shared, "statistic": "best_model_changes", "value": float(best_a != best_b), "note": f"best: {best_a} then {best_b}"},
        ]
    return rows


def worst_group_rows(lines: pd.DataFrame, by: list[str], metric: str, tail: float, min_units: int) -> list[dict]:
    """Mean over the groups, the worst group, and the mean of the worst share `tail` of them (CVaR)."""
    values = group_values(lines, by, metric, min_units).sort_values()
    if values.empty:
        return []
    k = max(1, int(np.ceil(len(values) * tail)))
    shared = {"n": len(values), "note": f"metric {metric}, n = groups of {' x '.join(by)}"}
    return [
        {**shared, "level_a": "(all groups)", "statistic": "mean", "value": float(values.mean())},
        {**shared, "level_a": values.index[0], "statistic": "worst_group", "value": float(values.iloc[0])},
        {**shared, "level_a": f"worst {tail:.0%}", "statistic": "cvar", "value": float(values.iloc[:k].mean())},
    ]


def standardized_rows(lines: pd.DataFrame, by: list[str], over: str) -> list[dict]:
    """Accuracy with the same weight for each level of `over` and each label, in each group.

    Two sites with different mixes of findings then compare on equal terms: what is left is a
    difference in how the model reads the cases, not in which cases it was given.
    """
    valid = lines[(lines["status"] == "valid") & lines["y"].notna()]
    valid = valid.assign(right=(valid["yhat"] == valid["y"]).astype(float))
    rows = []
    for levels, group in valid.groupby(by):
        levels = levels if isinstance(levels, tuple) else (levels,)
        cells = group.groupby([over, "y"])["right"].mean()
        rows += [
            {"level_a": ", ".join(f"{axis}={level}" for axis, level in zip(by, levels)), "statistic": name, "value": float(value),
             "n": len(cells), "n_units": group["unit"].nunique(), "note": f"n = cells of {over} x label"}
            for name, value in (("standardized_acc", cells.mean()), ("acc", group["right"].mean()))
        ]
    return rows


def heterogeneity_rows(lines: pd.DataFrame, by: list[str], over: str, metric: str, min_units: int) -> list[dict]:
    """In each group, how much the metric varies between the levels of `over`."""
    rows = []
    for levels, group in lines.groupby(by):
        levels = levels if isinstance(levels, tuple) else (levels,)
        values = group_values(group, [over], metric, min_units)
        if len(values) < 2:
            continue
        shared = {"level_a": ", ".join(f"{axis}={level}" for axis, level in zip(by, levels)), "n": len(values),
                  "note": f"metric {metric}, n = levels of {over}"}
        found = {"mean": values.mean(), "std": values.std(ddof=1), "min": values.min(), "max": values.max(),
                 "range": values.max() - values.min()}
        rows += [{**shared, "statistic": name, "value": float(value)} for name, value in found.items()]
    return rows


def stability(lines: pd.DataFrame, axis: str, keys: list[str]) -> float:
    """Share of the samples answered at every level of a controlled axis that keep their answer."""
    complete = decisions(lines[lines["status"] == "valid"], keys, [axis]).dropna()
    if complete.empty:
        return float("nan")
    return float((complete.max(axis=1) == complete.min(axis=1)).mean())


def pareto_rows(lines: pd.DataFrame, by: list[str], axis: str, keys: list[str], metric: str, min_units: int) -> list[dict]:
    """Three qualities per model: mean over the groups, worst group, stability over `axis`.
    A model is on the frontier when no other model is at least as good on the three and better on one."""
    found = {}
    for model, group in lines.groupby("model"):
        values = group_values(group, by, metric, min_units)
        if len(values):
            found[model] = [values.mean(), values.min(), stability(group, axis, keys)]
    table = pd.DataFrame(found, index=["mean", "worst_group", "stability"]).T.dropna()
    rows = []
    for model, own in table.iterrows():
        dominated = any((other >= own).all() and (other > own).any() for name, other in table.iterrows() if name != model)
        scores = {**own.to_dict(), "pareto_optimal": float(not dominated)}
        rows += [{"model": model, "level_a": "(all groups)", "statistic": name, "value": float(value), "n": len(table),
                  "note": f"metric {metric}, stability over {axis}, n = models compared"} for name, value in scores.items()]
    return rows


def link_rows(probes: pd.DataFrame, earlier: list[dict], target: str, of: str) -> list[dict]:
    """Across the models: does a model whose vectors give away `target` (the probe) also show a
    larger interaction in the analysis `of`? Spearman's ρ. It needs three models with a probe."""
    sizes = pd.DataFrame([row for row in earlier if row.get("name") == of and row.get("statistic") == "interaction"])
    if sizes.empty or probes.empty:
        return []
    size = sizes.assign(size=sizes["value"].abs()).groupby("model")["size"].mean()
    probe = probes[probes["target"] == target].set_index("model")["probe"]
    both = pd.concat([probe, size], axis=1, keys=["probe", "size"]).dropna()
    rows = [{"model": model, "level_a": f"probe {target}", "level_b": of, "statistic": "mean_abs_interaction",
             "value": float(row["size"]), "note": f"probe {row['probe']:.3f}"} for model, row in both.iterrows()]
    if len(both) >= 3 and both["probe"].nunique() > 1 and both["size"].nunique() > 1:
        rho, p = spearmanr(both["probe"], both["size"])
        rows.append({"model": "(all models)", "level_a": f"probe {target}", "level_b": of, "statistic": "spearman",
                     "value": float(rho), "p": float(p), "n": len(both), "note": "n = models"})
    return rows
