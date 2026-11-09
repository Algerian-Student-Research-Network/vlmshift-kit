"""Latent space probe (section 5.7)."""
import numpy as np
import pandas as pd
import pytest

from vlmshift.common import Issues
from vlmshift.config.schema import Latent, Stats
from vlmshift.stats.probes import balanced_accuracy, predict_by_folds, run_probes

STATS = Stats(bootstrap=200, min_units=1)


def test_site_is_found_in_the_vectors_of_the_fake_model(full_run):
    probes = pd.read_parquet(full_run.analysis / "probes.parquet")
    site = probes[probes["target"] == "site"].iloc[0]
    assert site["model"] == "fake-clip"
    # The vectors encode the site, on purpose. With 4 subjects in 2 folds, one image can still be
    # missed: it depends on the folds, which scikit-learn shuffles differently from 1.6 (Colab) to 1.8.
    assert site["score"] >= 0.75
    assert site["probe"] > 0.2 and site["probe"] == pytest.approx(site["score"] - site["floor"])
    assert site["chance"] == 0.5
    assert (site["n_images"], site["n_units"]) == (8, 4)


def test_label_probe_ignores_unknown(full_run):
    probes = pd.read_parquet(full_run.analysis / "probes.parquet")
    assert probes[probes["target"] == "label"].iloc[0]["n_images"] == 6


def test_a_unit_is_never_in_train_and_test():
    """Vector = unit number: the probe learns nothing from a unit it has not seen."""
    rng = np.random.default_rng(0)
    units = np.repeat(np.arange(40), 2).astype(str)
    target = rng.choice(["a", "b"], 40).repeat(2)
    vectors = np.eye(40).repeat(2, axis=0)
    predictions = predict_by_folds(vectors, target, units, 5, 0)
    score = balanced_accuracy(target, predictions, units)[0](np.ones((1, 40)))[0]
    assert score < 0.75  # with a leak, it would be almost 1


def test_noise_floor_stays_near_chance():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "image_hash": [f"i{i}" for i in range(80)],
        "unit": [f"u{i}" for i in range(80)],
        "site": rng.choice(["a", "b"], 80),
    })
    vectors = pd.DataFrame({
        "model": "m", "image_hash": data["image_hash"],
        "vector": [list(rng.normal(size=4)) for _ in range(80)],
    })
    row = run_probes(data, vectors, Latent(targets=["site"]), STATS, Issues(5)).iloc[0]
    assert row["floor"] == pytest.approx(0.5, abs=0.15)
    assert abs(row["probe"]) < 0.25


def test_target_that_varies_on_an_image_is_skipped():
    data = pd.DataFrame({"image_hash": ["i", "i"], "unit": ["u", "u"], "language": ["EN", "FR"]})
    vectors = pd.DataFrame({"model": ["m"], "image_hash": ["i"], "vector": [[1.0]]})
    issues = Issues(5)
    assert run_probes(data, vectors, Latent(targets=["language"]), STATS, issues).empty
    assert issues.rows[0][2] == "probe_skipped"


def test_chance_follows_the_number_of_classes():
    data = pd.DataFrame({"image_hash": list("abcdef"), "unit": list("abcdef"), "scanner": list("xyzxyz")})
    vectors = pd.DataFrame({"model": "m", "image_hash": list("abcdef"), "vector": [[float(i)] for i in range(6)]})
    row = run_probes(data, vectors, Latent(targets=["scanner"], folds=2), STATS, Issues(5)).iloc[0]
    assert row["chance"] == pytest.approx(1 / 3)


def test_too_little_data_is_not_estimable():
    data = pd.DataFrame({"image_hash": ["i"], "unit": ["u"], "site": ["a"]})
    vectors = pd.DataFrame({"model": ["m"], "image_hash": ["i"], "vector": [[1.0]]})
    issues = Issues(5)
    row = run_probes(data, vectors, Latent(targets=["site"]), STATS, issues).iloc[0]
    assert pd.isna(row["score"])
    assert issues.rows[0][2] == "probe_not_estimable"


def test_a_skipped_target_is_said_once_with_two_models():
    data = pd.DataFrame({"image_hash": ["i", "i"], "unit": ["u", "u"], "language": ["EN", "FR"]})
    vectors = pd.DataFrame({"model": ["m1", "m2"], "image_hash": ["i", "i"], "vector": [[1.0], [2.0]]})
    issues = Issues(5)
    run_probes(data, vectors, Latent(targets=["language"]), STATS, issues)
    assert [row[2] for row in issues.rows] == ["probe_skipped"]
