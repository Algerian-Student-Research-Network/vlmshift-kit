"""Metrics on continuous scores: AUROC and AUPRC of contrastive models.

A contrastive model gives a margin for each question, the score of the positive phrase minus
that of the negative one. These two metrics ask whether the margins rank the positive cases
above the negative ones, whatever the threshold. They only use the order of the margins: a
margin is a difference of similarities, not a probability.

Like the other metrics, each one is a function of the unit weights, so the bootstrap draws
patients, not rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import Stat

SCORE_METRICS = ("auroc", "auprc")
CELLS = 4_000_000  # numbers held at once while a statistic is computed on many draws


def by_chunks(weights: np.ndarray, n_lines: int, compute) -> np.ndarray:
    """Apply `compute` to a few draws at a time: 2000 draws of a large table do not fit at once."""
    step = max(1, CELLS // max(n_lines, 1))
    parts = [compute(weights[start:start + step]) for start in range(0, len(weights), step)]
    return np.concatenate(parts) if parts else np.empty(0)


def ties(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For sorted scores: where the group of equal scores of each row starts, and where it ends."""
    return np.searchsorted(scores, scores, side="left"), np.searchsorted(scores, scores, side="right")


def auroc_stat(units: np.ndarray, y: np.ndarray, scores: np.ndarray) -> Stat:
    """Chance that a positive case has a higher margin than a negative one. A tie counts half."""
    order = np.argsort(scores, kind="stable")
    units, positive, scores = units[order], y[order] == 1, scores[order]
    start, end = ties(scores)

    def compute(weights: np.ndarray) -> np.ndarray:
        lines = weights[:, units]
        below = np.concatenate([np.zeros((len(lines), 1)), np.cumsum(lines * ~positive, axis=1)], axis=1)
        wins = (lines * positive * (below[:, start] + below[:, end]) / 2).sum(axis=1)
        pairs = (lines * positive).sum(axis=1) * (lines * ~positive).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return wins / pairs

    return lambda weights: by_chunks(weights, len(units), compute)


def auprc_stat(units: np.ndarray, y: np.ndarray, scores: np.ndarray) -> Stat:
    """Average precision: the precision at each margin threshold, weighted by the recall gained there."""
    order = np.argsort(-scores, kind="stable")
    units, positive, scores = units[order], y[order] == 1, scores[order]
    last = np.flatnonzero(np.append(scores[1:] != scores[:-1], True))  # the end of each group of equal margins

    def compute(weights: np.ndarray) -> np.ndarray:
        lines = weights[:, units]
        hits = np.cumsum(lines * positive, axis=1)[:, last]
        named = np.cumsum(lines, axis=1)[:, last]
        gained = np.diff(hits, axis=1, prepend=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(named > 0, hits / named, 0)
            return (gained * precision).sum(axis=1) / hits[:, -1]

    return lambda weights: by_chunks(weights, len(units), compute)


BUILDERS = {"auroc": auroc_stat, "auprc": auprc_stat}


def score_stats(lines: pd.DataFrame, n_units: int) -> dict[str, tuple[Stat, int]]:
    """The score metrics of some rows: name → (statistic, units it rests on). Empty without margins.

    `lines` has the columns u (unit), y (label) and margin. Only the rows with a known label and
    a margin count. The support is the smaller of the units with a positive and with a negative.
    """
    if "margin" not in lines:
        return {}
    known = lines[lines["y"].notna() & lines["margin"].notna()]
    if known.empty:
        return {}
    units, y, margins = known["u"].to_numpy(), known["y"].to_numpy(), known["margin"].to_numpy(dtype=float)
    support = min(len(set(units[y == 1])), len(set(units[y == 0])))
    return {name: (build(units, y, margins), support) for name, build in BUILDERS.items()}


def score_shifts(at_a: pd.DataFrame, at_b: pd.DataFrame, n_units: int) -> dict[str, tuple[Stat, int]]:
    """Δ of each score metric between two levels: M(b) − M(a), on the same draws of units."""
    stats_a, stats_b = score_stats(at_a, n_units), score_stats(at_b, n_units)
    return {
        f"delta_{name}": (lambda weights, a=stats_a[name][0], b=stats_b[name][0]: b(weights) - a(weights),
                          min(stats_a[name][1], stats_b[name][1]))
        for name in stats_a if name in stats_b
    }
