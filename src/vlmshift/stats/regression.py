"""Models of whether an answer is right: a clustered logistic regression, and a split of the variance.

- regression: a logistic model of right/wrong on the levels of some axes and their interaction.
  The answers of one unit are not independent, so the standard errors are computed by unit
  (a GEE with an independence working correlation and a robust covariance).
- variance: how much of the variation in right/wrong goes with each axis (partial η², type II).

Both are written with numpy alone. A model that cannot be fitted says why; it never gives numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as fisher
from scipy.stats import norm

from .paired import holm

LARGE = 15  # a coefficient beyond this means a cell where the model is always right, or always wrong


def right_or_wrong(lines: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The valid answers with a known label and a value in each column, with `right` 1 or 0."""
    valid = lines[(lines["status"] == "valid") & lines["y"].notna()].dropna(subset=columns)
    return valid.assign(right=(valid["yhat"] == valid["y"]).astype(float))


def dummies(data: pd.DataFrame, axis: str, reference: str | None) -> pd.DataFrame:
    """One 0/1 column per level but the reference (or the first level), named `axis=level`."""
    levels = sorted(data[axis].unique())
    first = reference if reference in levels else levels[0]
    return pd.DataFrame({f"{axis}={level}": (data[axis] == level).astype(float) for level in levels if level != first},
                        index=data.index)


def design(data: pd.DataFrame, factors: list[str], covariates: list[str], references: dict) -> dict[str, pd.DataFrame]:
    """The columns of each term: every factor, the interaction of the first two, every covariate."""
    terms = {axis: dummies(data, axis, references.get(axis)) for axis in [*factors, *covariates]}
    if len(factors) >= 2:
        a, b = terms[factors[0]], terms[factors[1]]
        crossed = {f"{left} x {right}": a[left] * b[right] for left in a for right in b}
        terms = {**{axis: terms[axis] for axis in factors}, f"{factors[0]} x {factors[1]}": pd.DataFrame(crossed, index=data.index),
                 **{axis: terms[axis] for axis in covariates}}
    return {name: columns.loc[:, columns.any()] for name, columns in terms.items()}  # an empty cell has no column


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 60) -> tuple[np.ndarray, np.ndarray, bool]:
    """Logistic regression by Newton's method. Return (coefficients, fitted chances, converged)."""
    beta = np.zeros(x.shape[1])
    for _ in range(steps):
        chance = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        weight = chance * (1 - chance)
        step = np.linalg.pinv(x.T @ (x * weight[:, None])) @ (x.T @ (y - chance))
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            return beta, chance, True
    return beta, chance, False


def regression_rows(lines: pd.DataFrame, factors: list[str], covariates: list[str], references: dict) -> list[dict]:
    """One row per coefficient: log-odds, 95% interval and p from standard errors by unit."""
    data = right_or_wrong(lines, [*factors, *covariates])
    shared = {"n": len(data), "n_units": data["unit"].nunique() if len(data) else 0}
    if len(data) < 4 or shared["n_units"] < 2 or data["right"].nunique() < 2:
        return [{**shared, "level_a": "(model)", "statistic": "not_estimable",
                 "note": "too few answers or units, or the model is always right or always wrong"}]
    terms = design(data, factors, covariates, references)
    columns = pd.concat([pd.DataFrame({"(intercept)": 1.0}, index=data.index), *terms.values()], axis=1)
    x, y = columns.to_numpy(dtype=float), data["right"].to_numpy()
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return [{**shared, "level_a": "(model)", "statistic": "not_estimable",
                 "note": "some levels always come together, so their effects cannot be told apart"}]
    beta, chance, converged = fit_logistic(x, y)

    # Robust covariance: the errors are added up inside each unit before being squared.
    bread = np.linalg.pinv(x.T @ (x * (chance * (1 - chance))[:, None]))
    codes, units = pd.factorize(data["unit"])
    scores = np.zeros((len(units), x.shape[1]))
    np.add.at(scores, codes, x * (y - chance)[:, None])
    covariance = bread @ (scores.T @ scores) @ bread
    error = np.sqrt(np.clip(np.diag(covariance), 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * norm.sf(np.abs(beta / error))
    status = "ok" if converged and np.abs(beta).max() < LARGE else "separation: a cell is always right or always wrong"
    adjusted = holm(np.where(columns.columns == "(intercept)", np.nan, p))
    rows = []
    for name, value, se, p_value, p_holm in zip(columns.columns, beta, error, p, adjusted):
        shared_term = {**shared, "level_a": name, "note": status}
        rows.append({**shared_term, "statistic": "log_odds", "value": float(value), "ci95_low": float(value - 1.96 * se),
                     "ci95_high": float(value + 1.96 * se), "p": float(p_value), "p_holm": p_holm})
        rows.append({**shared_term, "statistic": "odds_ratio", "value": float(np.exp(np.clip(value, -50, 50)))})
    return rows


def residual(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Sum of squared errors of the least-squares fit, and the rank of the columns."""
    solution = np.linalg.lstsq(x, y, rcond=None)
    return float(((y - x @ solution[0]) ** 2).sum()), int(solution[2])


def variance_rows(lines: pd.DataFrame, factors: list[str]) -> list[dict]:
    """Share of the variation of right/wrong that goes with each axis, once the others are counted.

    Type II: a main effect is measured given the other main effects; the interaction of the
    first two axes is measured given all the main effects.
    """
    data = right_or_wrong(lines, factors)
    factors = [axis for axis in factors if data[axis].nunique() > 1]
    if not factors or data["right"].nunique() < 2:
        return [{"level_a": "(model)", "statistic": "not_estimable", "n": len(data), "note": "no variation to split"}]
    terms = design(data, factors, [], {})
    y = data["right"].to_numpy()
    one = np.ones((len(data), 1))

    def columns(names) -> np.ndarray:
        return np.hstack([one, *[terms[name].to_numpy(dtype=float) for name in names]])

    full_error, full_rank = residual(columns(terms), y)
    left = len(data) - full_rank
    mains_error, mains_rank = residual(columns(factors), y)
    rows = []
    for name in terms:
        if name in factors:  # given the other main effects
            without_error, without_rank = residual(columns([axis for axis in factors if axis != name]), y)
            sum_sq, freedom = without_error - mains_error, mains_rank - without_rank
        else:  # the interaction, given every main effect
            sum_sq, freedom = mains_error - full_error, full_rank - mains_rank
        if freedom <= 0:
            continue
        ratio = (sum_sq / freedom) / (full_error / left) if left > 0 and full_error > 0 else float("nan")
        shared = {"level_a": name, "n": len(data), "n_units": data["unit"].nunique(), "note": f"df {freedom}"}
        rows += [
            {**shared, "statistic": "partial_eta2", "value": float(sum_sq / (sum_sq + full_error)) if sum_sq + full_error > 0 else float("nan"),
             "p": float(fisher.sf(ratio, freedom, left)) if np.isfinite(ratio) else float("nan")},
            {**shared, "statistic": "sum_sq", "value": float(sum_sq)},
        ]
    return rows
