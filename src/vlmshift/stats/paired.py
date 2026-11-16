"""Paired tests on a controlled axis: McNemar, Cochran's Q, Holm's correction, detectable effect.

The same question is asked at each level, so the answers come in pairs. McNemar asks whether a
model is right more often at one level than at the other, looking only at the pairs where it is
right at one and wrong at the other. Cochran's Q asks the same over all the levels at once.

These tests count each pair as independent. With several images per unit they are too
confident: the row then says so, and the bootstrap intervals of the measures remain the reference.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm

EXACT_BELOW = 25  # discordant pairs under which the exact binomial test replaces the χ² approximation


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, str]:
    """Test that two paired 0/1 series differ as often one way as the other.

    Return (statistic, p, discordant pairs, method). Without any discordant pair there is
    nothing to test: p = 1.
    """
    gained, lost = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    discordant = gained + lost
    if discordant == 0:
        return 0.0, 1.0, 0, "no discordant pair"
    if discordant < EXACT_BELOW:
        return float(min(gained, lost)), float(binomtest(min(gained, lost), discordant, 0.5).pvalue), discordant, "exact"
    statistic = max(abs(gained - lost) - 1, 0) ** 2 / discordant  # with continuity correction
    return float(statistic), float(chi2.sf(statistic, 1)), discordant, "chi2"


def cochran_q(values: np.ndarray) -> tuple[float, float]:
    """Cochran's Q on a table samples × levels of 0/1. Needs 3 levels or more."""
    n, k = values.shape
    if n == 0 or k < 3:
        return float("nan"), float("nan")
    columns, rows, total = values.sum(axis=0), values.sum(axis=1), values.sum()
    spread = k * total - (rows ** 2).sum()
    if spread <= 0:  # every sample has the same value at every level
        return 0.0, 1.0
    q = (k - 1) * (k * (columns ** 2).sum() - total ** 2) / spread
    return float(q), float(chi2.sf(q, k - 1))


def holm(pvalues) -> np.ndarray:
    """Holm's step-down correction. Missing p-values stay missing and do not count as tests."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    tested = np.flatnonzero(np.isfinite(pvalues))
    order = tested[np.argsort(pvalues[tested])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def detectable(n_pairs: int, discordant_share: float, alpha: float, power: float) -> float:
    """Smallest difference between the two kinds of discordant pairs, as a share of the pairs,
    that McNemar's test would detect with this many pairs (normal approximation)."""
    if n_pairs == 0:
        return float("nan")
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * np.sqrt(discordant_share / n_pairs))


def wide(lines: pd.DataFrame, keys: list[str], axis: str, column: str) -> pd.DataFrame:
    """One row per sample, one column per level, for a 0/1 column. The unit of each sample comes along."""
    table = lines.dropna(subset=[column]).drop_duplicates([*keys, axis])
    values = table.pivot(index=keys, columns=axis, values=column)
    return values.assign(_unit=table.drop_duplicates(keys).set_index(keys)["unit"])


def pair_rows(values: pd.DataFrame, a: str, b: str, kind: str, alpha: float, power: float) -> list[dict]:
    """McNemar between two levels, and the difference it could detect."""
    if a not in values or b not in values:
        return []
    both = values[[a, b, "_unit"]].dropna(subset=[a, b])
    if both.empty:
        return []
    statistic, p, discordant, method = mcnemar(both[a].to_numpy(), both[b].to_numpy())
    units = both["_unit"].nunique()
    note = method + ("" if units == len(both) else "; several pairs per unit: p is too confident")
    shared = {"level_a": a, "level_b": b, "n": len(both), "n_units": units}
    return [
        {**shared, "statistic": f"mcnemar_{kind}", "value": statistic, "p": p, "note": note},
        {**shared, "statistic": f"discordant_{kind}", "value": discordant / len(both)},
        {**shared, "statistic": f"detectable_{kind}", "value": detectable(len(both), discordant / len(both), alpha, power),
         "note": f"alpha {alpha}, power {power}"},
    ]


def paired_tests(cell: pd.DataFrame, axis: str, pairs, keys: list[str], alpha: float, power: float) -> list[dict]:
    """The paired tests of one model in one cell.

    Two readings of each pair: `correct`, is the answer right (needs a known label), and
    `answer`, is it a yes (works without labels). Only valid answers are paired.
    """
    valid = cell[cell["status"] == "valid"]
    valid = valid.assign(correct=(valid["yhat"] == valid["y"]).astype(float).where(valid["y"].notna()))
    rows = []
    for kind, column in (("correct", "correct"), ("answer", "yhat")):
        values = wide(valid, keys, axis, column)
        for a, b in pairs:
            rows += pair_rows(values, a, b, kind, alpha, power)
        levels = [level for level in values.columns if level != "_unit"]
        complete = values.dropna(subset=levels)
        if len(levels) >= 3 and len(complete):
            q, p = cochran_q(complete[levels].to_numpy(dtype=float))
            rows.append({"level_a": "(all levels)", "statistic": f"cochran_q_{kind}", "value": q, "p": p,
                         "n": len(complete), "n_units": complete["_unit"].nunique(), "note": f"{len(levels)} levels"})
    return rows


def level_pairs(levels: list[str], reference: str | None) -> list[tuple[str, str]]:
    """The pairs to compare: each level against the reference, or every pair without one."""
    if reference:
        return [(reference, level) for level in levels if level != reference]
    return list(combinations(levels, 2))
