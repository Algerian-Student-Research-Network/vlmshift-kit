"""Effects that need more than two levels at once: interactions, consistency, grounding.

- interaction: does the shift on one axis depend on the level of another? (domain × language,
  frame × term, image condition × language × domain);
- consistency: does a sample keep its answer at every level of a controlled axis?
- information: how much the answer tells about the level it was asked at;
- disagreement shift: do two levels disagree more under another condition, on the same samples?
- taxonomy: is a sample stable across levels because the model reads the image, or ignores it?

Every rate is a mean over samples, recomputed on each draw of units like the other metrics.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .bootstrap import estimate
from .metrics import Stat, stat_of, support, unit_sums
from .pairing import with_unit_codes


def mean_stat(values: np.ndarray, units: np.ndarray, n_units: int) -> Stat:
    """The mean of one value per sample, as a function of the unit weights."""
    total = np.bincount(units, weights=values, minlength=n_units)
    count = np.bincount(units, minlength=n_units).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return lambda weights: (weights @ total) / (weights @ count)


def label_of(axes, levels) -> str:
    return ", ".join(f"{axis}={level}" for axis, level in zip(axes, levels))


def decisions(valid: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """One row per sample, one column per level (or combination of levels): the decision, 1 or 0."""
    table = valid.drop_duplicates([*keys, *columns])
    return table.pivot(index=keys, columns=columns, values="yhat")


def units_of(valid: pd.DataFrame, keys: list[str], index) -> tuple[np.ndarray, int]:
    """The unit code of each sample of `index`."""
    units = valid.drop_duplicates(keys).set_index(keys)["unit"].reindex(index)
    codes, found = pd.factorize(units)
    return codes, len(found)


# --- Interaction ---


def matched(valid: pd.DataFrame, controlled: list[str], corner_levels: dict, keys: list[str]) -> pd.DataFrame:
    """Keep the samples that have an answer at every corner of the controlled axes.

    An interaction between controlled axes compares the same samples in each cell: a sample
    missing in one cell would change which questions are compared, not only the condition.
    """
    for axis in controlled:
        valid = valid[valid[axis].isin(corner_levels[axis])]
    if not controlled:
        return valid
    corners = valid.drop_duplicates([*keys, *controlled]).groupby(keys)[controlled[0]].transform("size")
    full = valid.drop_duplicates([*keys, *controlled])[corners == 2 ** len(controlled)]
    return valid.merge(full[keys].drop_duplicates(), on=keys)


def interaction_rows(cell, axes, references, levels, metric, controlled, keys, settings, m) -> list[dict]:
    """Difference of differences of a metric over two or three axes, against the reference corner.

    Two axes: M(a, b) − M(a, 0) − M(0, b) + M(0, 0). It is 0 when the two shifts simply add up.
    Three axes: how much that interaction changes when the third axis leaves its reference.
    """
    valid = cell[cell["status"] == "valid"]
    rows, k = [], len(axes)
    others = [[level for level in levels[axis] if level != references[axis]] for axis in axes]
    for combo in product(*others):
        corner_levels = {axis: [references[axis], level] for axis, level in zip(axes, combo)}
        part = matched(valid, [axis for axis in axes if axis in controlled], corner_levels, keys)
        used, n_units = with_unit_codes(part)
        stats, supports = {}, []
        for flags in product((0, 1), repeat=k):
            mask = np.ones(len(used), dtype=bool)
            for axis, level, flag in zip(axes, combo, flags):
                mask &= (used[axis] == (level if flag else references[axis])).to_numpy()
            sums = unit_sums(used[mask], n_units)
            stats[flags] = stat_of(metric, sums)
            supports.append(support(metric, sums))
        if min(supports) == 0:  # a corner without data: this combination was not asked
            continue

        def contrast(chosen, fixed=()):
            """Signed sum over the corners of the axes in `chosen`, the others held at `fixed`."""
            free = [i for i in range(k) if i in chosen]

            def stat(weights):
                total = 0
                for bits in product((0, 1), repeat=len(free)):
                    flags = list(fixed) if fixed else [0] * k
                    for i, bit in zip(free, bits):
                        flags[i] = bit
                    total = total + (-1) ** (len(free) - sum(bits)) * stats[tuple(flags)](weights)
                return total
            return stat

        found = {"interaction": contrast(range(k))}
        if k == 2:
            found |= {f"main_{axes[0]}": contrast([0]), f"main_{axes[1]}": contrast([1]),
                      "joint": lambda weights: stats[(1, 1)](weights) - stats[(0, 0)](weights)}
        else:  # the two-axis interaction, at the reference of the third axis and away from it
            found |= {"interaction_at_reference": contrast([0, 1]),
                      "interaction_away": contrast([0, 1], fixed=(0, 0, 1))}
        shared = {"level_a": label_of(axes, [references[axis] for axis in axes]), "level_b": label_of(axes, combo),
                  "n": len(used), "n_units": min(supports), "note": f"metric {metric}"}
        for name, stat in found.items():
            rows.append({**shared, "statistic": name, **estimate(stat, n_units, min(supports), settings, m, True)})
    return rows


# --- Consistency over a controlled axis ---


def entropy(share: np.ndarray) -> np.ndarray:
    """Binary entropy in bits: 0 when every level agrees, 1 when they split in half."""
    inside = np.clip(share, 1e-12, 1 - 1e-12)
    bits = -(inside * np.log2(inside) + (1 - inside) * np.log2(1 - inside))
    return np.where((share <= 0) | (share >= 1), 0.0, bits)


def consistency_rows(cell, axis, levels, keys, settings, m) -> list[dict]:
    """Over the samples answered at every level: share that never change, entropy of the answers,
    and accuracy of the majority answer."""
    valid = cell[(cell["status"] == "valid") & cell[axis].isin(levels)]
    answers = decisions(valid, keys, [axis])
    complete = answers.dropna()
    if complete.empty or complete.shape[1] < 2:
        return []
    units, n_units = units_of(valid, keys, complete.index)
    yes_share = complete.to_numpy(dtype=float).mean(axis=1)
    label = valid.drop_duplicates(keys).set_index(keys)["y"].reindex(complete.index).to_numpy()
    known = ~np.isnan(label)
    found = {
        "invariant": mean_stat(((yes_share == 0) | (yes_share == 1)).astype(float), units, n_units),
        "entropy_bits": mean_stat(entropy(yes_share), units, n_units),
        "vote_margin": mean_stat(np.abs(2 * yes_share - 1), units, n_units),
    }
    if known.any():
        right = ((yes_share > 0.5) == (label == 1)).astype(float)
        found["majority_acc"] = mean_stat(right[known], units[known], n_units)
    shared = {"level_a": "(all levels)", "n": len(complete), "n_units": n_units,
              "note": f"{complete.shape[1]} levels, {len(answers) - len(complete)} incomplete samples left out"}
    return [{**shared, "statistic": name, **estimate(stat, n_units, n_units, settings, m, False)}
            for name, stat in found.items()]


def information_rows(cell, axis, levels) -> list[dict]:
    """Mutual information between the answer and the level, in bits, inside one cell.

    0 when the level tells nothing about the answer. Half a count is added to each case, so a
    small cell does not look informative by accident.
    """
    valid = cell[(cell["status"] == "valid") & cell[axis].isin(levels)]
    if valid[axis].nunique() < 2:
        return []
    counts = pd.crosstab(valid[axis], valid["yhat"]).reindex(columns=[0.0, 1.0], fill_value=0).to_numpy() + 0.5
    joint = counts / counts.sum()
    apart = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    bits = float((joint * np.log2(joint / apart)).sum())
    return [{"level_a": "(all levels)", "statistic": "mutual_information_bits", "value": bits,
             "n": len(valid), "n_units": valid["unit"].nunique(), "note": f"{valid[axis].nunique()} levels"}]


# --- Grounding ---


def disagreement_shift_rows(cell, compare, across, reference, pairs, conditions, keys, settings, m) -> list[dict]:
    """Do two levels of `compare` disagree more under a condition of `across` than at its reference?

    The four answers of each sample are needed: the two levels, at the reference and under the
    condition. So an INVALID answer never changes which samples are compared.
    """
    valid = cell[cell["status"] == "valid"]
    answers = decisions(valid, keys, [across, compare])
    rows = []
    for (a, b), condition in product(pairs, conditions):
        columns = [(reference, a), (reference, b), (condition, a), (condition, b)]
        if not all(column in answers.columns for column in columns):
            continue
        four = answers[columns].dropna()
        if four.empty:
            continue
        units, n_units = units_of(valid, keys, four.index)
        before = (four[(reference, a)] != four[(reference, b)]).to_numpy(dtype=float)
        under = (four[(condition, a)] != four[(condition, b)]).to_numpy(dtype=float)
        shared = {"level_a": f"{a} vs {b}", "level_b": f"{across}={condition}", "n": len(four), "n_units": n_units}
        for name, values, is_shift in (("disagreement_reference", before, False), ("disagreement", under, False),
                                       ("delta_disagreement", under - before, True)):
            stat = mean_stat(values, units, n_units)
            rows.append({**shared, "statistic": name, **estimate(stat, n_units, n_units, settings, m, is_shift)})
    return rows


def taxonomy_rows(cell, stable_axis, across, reference, conditions, keys, settings, m) -> list[dict]:
    """Sort each sample in two ways: stable (same answer at every level of `stable_axis`, on the
    reference image) and grounded (its answer changes, at some level, when the image is replaced).

    A model can be stable because it ignores the image: stable and ungrounded is not robustness.
    """
    valid = cell[cell["status"] == "valid"]
    answers = decisions(valid, keys, [across, stable_axis])
    rows = []
    for condition in conditions:
        if reference not in answers.columns.get_level_values(0) or condition not in answers.columns.get_level_values(0):
            continue
        real, control = answers[reference], answers[condition]
        common = [level for level in real.columns if level in control.columns]
        both = real[common].notna() & control[common].notna()
        kept = both.sum(axis=1) >= 2
        if not kept.any():
            continue
        real, control, both = real.loc[kept, common], control.loc[kept, common], both[kept]
        seen = real.where(both)
        stable = (seen.max(axis=1) == seen.min(axis=1)).to_numpy()
        moved = ((real != control) & both).to_numpy()
        grounded = moved.any(axis=1)
        units, n_units = units_of(valid, keys, real.index)
        shares = {
            "grounded_stable": grounded & stable, "grounded_unstable": grounded & ~stable,
            "ungrounded_stable": ~grounded & stable, "ungrounded_unstable": ~grounded & ~stable,
        }
        shared = {"level_a": reference, "level_b": f"{across}={condition}", "n": int(kept.sum()), "n_units": n_units}
        for name, flags in shares.items():
            stat = mean_stat(flags.astype(float), units, n_units)
            rows.append({**shared, "statistic": name, **estimate(stat, n_units, n_units, settings, m, False)})
        fraction = mean_stat(moved.sum(axis=1) / both.to_numpy().sum(axis=1), units, n_units)
        rows.append({**shared, "statistic": "grounded_level_share", **estimate(fraction, n_units, n_units, settings, m, False)})
    return rows
