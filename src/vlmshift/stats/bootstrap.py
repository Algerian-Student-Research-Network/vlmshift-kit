"""Bootstrap intervals over independent units, and significance (5.4, 5.5)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

EMPTY = dict.fromkeys(["value", "ci_low", "ci_high", "ci95_low", "ci95_high", "significant"])


@lru_cache(maxsize=2)  # same arguments → same draws: no need to redo them for each number
def draw_weights(n_units: int, repeats: int, seed: int) -> np.ndarray:
    """One row per draw: how many times each unit was drawn (with replacement)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_units, np.full(n_units, 1 / n_units), size=repeats).astype(float)


def summarize(stat, weights, alpha: float, m: int, is_shift: bool) -> dict:
    """Value, corrected interval (Bonferroni, m comparisons) and 95% interval.

    `weights` is None when there are too few units: the number is not estimable.
    """
    if weights is None:
        return EMPTY
    value = stat(np.ones((1, weights.shape[1])))[0]
    draws = stat(weights)
    draws = draws[np.isfinite(draws)]  # a draw may contain no useful case
    if not np.isfinite(value) or draws.size == 0:
        return EMPTY
    tail = 100 * alpha / (2 * m)
    low, high, low95, high95 = np.percentile(draws, [tail, 100 - tail, 2.5, 97.5])
    return {
        "value": float(value),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci95_low": float(low95),
        "ci95_high": float(high95),
        # Significant: the corrected interval does not contain 0. Only concerns shifts.
        "significant": bool(low > 0 or high < 0) if is_shift else None,
    }
