"""Base metrics (section 5.1), and the answer-change ratios (section 5.2).

All of them are ratios of two counts. So we keep the counts per independent
unit: recomputing a metric on a bootstrap draw amounts to weighting these
counts by the number of times each unit was drawn. It is short and very fast.

Strict mode: INVALID rows are present (empty decision). They count in the
denominators but never in the correct answers: an INVALID therefore counts as wrong.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Sums = dict[str, np.ndarray]
Stat = Callable[[np.ndarray], np.ndarray]  # unit weights → values of the statistic


def rate(top, bottom):
    """Ratio. A division by zero gives NaN (not estimable), never 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return top / bottom


def counter(units: np.ndarray, n_units: int):
    """Return a function that counts, per unit, the rows of a mask."""
    return lambda mask: np.bincount(units[mask], minlength=n_units).astype(float)


def unit_sums(lines: pd.DataFrame, n_units: int) -> Sums:
    """Counts per unit. `lines` has the columns u (unit), y (label), yhat (decision)."""
    y, yhat = lines["y"].to_numpy(), lines["yhat"].to_numpy()
    count = counter(lines["u"].to_numpy(), n_units)
    return {
        "tp": count((y == 1) & (yhat == 1)),
        "pos": count(y == 1),
        "tn": count((y == 0) & (yhat == 0)),
        "neg": count(y == 0),
        "known": count(~np.isnan(y)),
        "yes": count(yhat == 1),
        "valid": count(~np.isnan(yhat)),
        "invalid": count(np.isnan(yhat)),
        "answered": count(np.ones(len(y), dtype=bool)),
    }


def f1_of(hit, real, said):
    """F1 of one class: 2·hits / (cases of the class + answers naming it). Without any case: not estimable."""
    return np.where(real > 0, rate(2 * hit, real + said), np.nan)


def f1(s: Sums):
    """F1 of the positive class. A wrong or unreadable answer on a negative is a false positive."""
    return f1_of(s["tp"], s["pos"], s["tp"] + s["neg"] - s["tn"])


def macro_f1(s: Sums):
    """Mean of the F1 of the two classes: the two weigh the same, whatever the prevalence."""
    negative = f1_of(s["tn"], s["neg"], s["tn"] + s["pos"] - s["tp"])
    return (f1(s) + negative) / 2


def mcc(s: Sums):
    """Matthews correlation. A model that always gives the same answer has 0, as by chance;
    with only one class in the labels it is not estimable."""
    tp, tn = s["tp"], s["tn"]
    fn, fp = s["pos"] - tp, s["neg"] - tn
    spread = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    both = (s["pos"] > 0) & (s["neg"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(spread > 0, (tp * tn - fp * fn) / spread, np.where(both, 0.0, np.nan))


METRICS = {
    "sens": lambda s: rate(s["tp"], s["pos"]),
    "spec": lambda s: rate(s["tn"], s["neg"]),
    "ba": lambda s: (rate(s["tp"], s["pos"]) + rate(s["tn"], s["neg"])) / 2,
    "acc": lambda s: rate(s["tp"] + s["tn"], s["known"]),
    "f1": f1,
    "macro_f1": macro_f1,
    "mcc": mcc,
    "yes_rate": lambda s: rate(s["yes"], s["valid"]),
    "inv_rate": lambda s: rate(s["invalid"], s["answered"]),
    # Answer change (section 5.2), on the counts of change_sums. In valid mode, there is no
    # INVALID pair. In strict mode, they count as changes.
    "chg": lambda s: rate(s["yes_no"] + s["no_yes"] + s["invalid"], s["pairs"]),
    "chg_yes_no": lambda s: rate(s["yes_no"], s["pairs"]),
    "chg_no_yes": lambda s: rate(s["no_yes"], s["pairs"]),
}


# What each number divides by. It rests only on the units that have some of it:
# a sensitivity rests on the units with a positive case, whatever the size of the cell.
SUPPORT = {
    "sens": ["pos"], "spec": ["neg"], "ba": ["pos", "neg"], "yes_rate": ["valid"], "inv_rate": ["answered"],
    "acc": ["known"], "f1": ["pos"], "macro_f1": ["pos", "neg"], "mcc": ["pos", "neg"],
    "chg": ["pairs"], "chg_yes_no": ["pairs"], "chg_no_yes": ["pairs"],
}


def support(name: str, *sums: Sums) -> int:
    """Units a number rests on: those with a non-zero denominator, at each level compared."""
    return min(int((part[key] > 0).sum()) for part in sums for key in SUPPORT[name])


def weigh(sums: Sums, weights: np.ndarray) -> Sums:
    """Sum each count: one value per row of the weight matrix."""
    return {name: weights @ values for name, values in sums.items()}


def stat_of(name: str, sums: Sums) -> Stat:
    return lambda weights: METRICS[name](weigh(sums, weights))
