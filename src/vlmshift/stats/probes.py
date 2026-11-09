"""Latent space probe (section 5.7): can a column be read from the vectors?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common import Issues
from .bootstrap import draw_weights
from .metrics import counter, rate

SHUFFLES = 5  # number of shuffles for the noise floor


def predict_by_folds(vectors, target, units, folds: int, seed: int) -> np.ndarray:
    """Test predictions with cross-validation. A unit stays in a single fold."""
    predictions = np.empty(len(target), dtype=object)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train, test in splitter.split(vectors, target, units):
        if len(set(target[train])) < 2:  # only one class to learn
            predictions[test] = target[train][0]
        else:
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            predictions[test] = probe.fit(vectors[train], target[train]).predict(vectors[test])
    return predictions


def balanced_accuracy(target, predictions, units) -> tuple:
    """Balanced accuracy that can be weighted per unit. Return (weights → BA function, number of units)."""
    codes, unique = pd.factorize(units)
    classes = np.unique(target)
    count = counter(codes, len(unique))
    hits = np.array([count((target == c) & (predictions == c)) for c in classes]).T
    totals = np.array([count(target == c) for c in classes]).T
    return (lambda weights: np.nanmean(rate(weights @ hits, weights @ totals), axis=1)), len(unique)


def probe_scores(vectors, target, units, folds: int, stats) -> dict:
    """Probe score, noise floor and 95% interval."""
    predictions = predict_by_folds(vectors, target, units, folds, stats.seed)
    score_of, n_units = balanced_accuracy(target, predictions, units)
    score = float(score_of(np.ones((1, n_units)))[0])
    draws = score_of(draw_weights(n_units, stats.bootstrap, stats.seed))

    rng = np.random.default_rng(stats.seed)
    floors = []
    for _ in range(SHUFFLES):
        mixed = rng.permutation(target)
        floor_of, _ = balanced_accuracy(mixed, predict_by_folds(vectors, mixed, units, folds, stats.seed), units)
        floors.append(float(floor_of(np.ones((1, n_units)))[0]))
    floor = float(np.mean(floors))

    low, high = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
    return {
        "score": score, "floor": floor, "probe": score - floor,
        "ci_low": float(low), "ci_high": float(high),
    }


def run_probes(variants, embeddings, latent, stats, issues: Issues) -> pd.DataFrame:
    """One row per model and per target column."""
    images = []  # (target, one row per image): the same for every model, so checked once
    for target in latent.targets:
        data = variants[["image_hash", "unit", target]].dropna()
        if target == "label":
            data = data[data["label"] != "unknown"]
        # The probe is about the image: a target that changes on the same image makes no sense.
        if len(data) and data.groupby("image_hash")[target].nunique().max() > 1:
            issues.warn("probe_skipped", f"Probe {target}: the column varies for the same image")
        else:
            images.append((target, data.drop_duplicates("image_hash")))
    rows = []
    for model, vectors in embeddings.groupby("model"):
        for target, data in images:
            data = data.merge(vectors, on="image_hash")
            n_classes, n_units = data[target].nunique(), data["unit"].nunique()
            row = {
                "model": model, "target": target, "n_images": len(data),
                "n_units": n_units, "chance": 1 / max(n_classes, 1),
            }
            if n_classes >= 2 and n_units >= latent.folds:
                X = np.array(data["vector"].tolist())
                y, units = data[target].to_numpy(dtype=object), data["unit"].to_numpy()
                row |= probe_scores(X, y, units, latent.folds, stats)
            else:
                issues.warn("probe_not_estimable", f"Probe {target}, {model}: too little data")
            rows.append(row)
    columns = ["model", "target", "score", "floor", "probe", "ci_low", "ci_high", "chance", "n_images", "n_units"]
    return pd.DataFrame(rows, columns=columns)
