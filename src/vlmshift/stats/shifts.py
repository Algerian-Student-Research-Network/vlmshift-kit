"""Shift metrics (section 5.2): shifts between levels and answer change."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import Stat, Sums, counter, stat_of


def delta_of(name: str, sums_a: Sums, sums_b: Sums) -> Stat:
    """Δ_M(b, a) = M(b) − M(a)."""
    stat_a, stat_b = stat_of(name, sums_a), stat_of(name, sums_b)
    return lambda weights: stat_b(weights) - stat_a(weights)


def change_sums(at_a: pd.DataFrame, at_b: pd.DataFrame, keys: list[str], n_units: int) -> Sums:
    """Counts per unit over the pairs: yes → no, no → yes, with INVALID, total."""
    pairs = at_a.merge(at_b, on=keys, suffixes=("_a", "_b"))
    count = counter(pairs["u_a"].to_numpy(), n_units)
    yes_a, yes_b = pairs["yhat_a"].to_numpy(), pairs["yhat_b"].to_numpy()
    return {
        "yes_no": count((yes_a == 1) & (yes_b == 0)),
        "no_yes": count((yes_a == 0) & (yes_b == 1)),
        "invalid": count(np.isnan(yes_a) | np.isnan(yes_b)),
        "pairs": count(np.ones(len(pairs), dtype=bool)),
    }

