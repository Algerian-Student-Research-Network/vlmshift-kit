"""stats/: values computed by hand, simulated effect, degenerate cases."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from vlmshift.cli import do_analyze
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_analysis, read_yaml, write_yaml
from vlmshift.config.schema import Axis, Stats
from vlmshift.stats.bootstrap import draw_weights, summarize
from vlmshift.stats.measures import base_rows, shift_rows
from vlmshift.stats.pairing import pairing_keys, select

SETTINGS = Stats(bootstrap=200, min_units=1)


def find(metrics, **labels):
    rows = metrics
    for column, value in labels.items():
        rows = rows[rows[column].isna() if pd.isna(value) else rows[column] == value]
    assert len(rows) == 1, labels
    return rows.iloc[0]


def lines_of(rows: list[tuple]) -> pd.DataFrame:
    """Small table of rows: (unit, sample, level, label, decision, status)."""
    table = pd.DataFrame(rows, columns=["unit", "sample_id", "level", "label", "decision", "status"])
    table["y"] = table["label"].map({"positive": 1.0, "negative": 0.0}).astype(float)
    table["yhat"] = table["decision"].map({"yes": 1.0, "no": 0.0}).astype(float)
    return table


def by_metric(rows, mode="valid"):
    return {row["metric"]: row for row in rows if row["mode"] == mode}


def test_values_computed_by_hand(full_run):
    expected = pd.read_csv(FIXTURES / "metrics.csv", dtype=str, keep_default_na=False)
    for row in expected.to_dict("records"):
        value = row.pop("value")
        found = find(full_run.metrics, **{k: v if v else None for k, v in row.items()})
        if value:
            assert found["value"] == pytest.approx(float(Fraction(value)), abs=1e-12), row
        else:
            assert pd.isna(found["value"]), row


def test_interval_contains_the_value(full_run):
    computed = full_run.metrics.dropna(subset=["value", "ci_low"])
    assert len(computed) > 0
    assert (computed["ci_low"] <= computed["value"] + 1e-9).all()
    assert (computed["value"] <= computed["ci_high"] + 1e-9).all()


def test_corrected_interval_contains_the_exploratory_one(full_run):
    computed = full_run.metrics.dropna(subset=["value"])
    assert (computed["ci_low"] <= computed["ci95_low"] + 1e-9).all()
    assert (computed["ci_high"] >= computed["ci95_high"] - 1e-9).all()


def test_bonferroni_counts_every_cell(full_run):
    shifts = full_run.metrics[full_run.metrics["level_b"].notna()]
    assert shifts.groupby("measure")["m"].first().to_dict() == {
        "language": 1, "site": 1, "language_by_site": 2,
    }


def test_an_unfinished_run_is_corrected_for_every_cell(toy, edit):
    """A model that has not reached site B yet gets the m of the finished run: 1 pair × 2 sites."""
    from test_inference import only, run

    folder = run(edit(lambda raw: only(raw, "fake_generative", params={"faults": False})))
    table = pd.read_parquet(next(toy.glob("workspace/tables/*/table.parquet")))
    results = pd.read_parquet(folder / "results.parquet")
    results[~results["id"].isin(table.loc[table["site"] == "B", "id"])].to_parquet(folder / "results.parquet")
    report = analyze(toy, folder, lambda raw: raw.pop("latent"))
    metrics = pd.read_parquet(report.parent / "metrics.parquet")
    shifts = metrics[(metrics["measure"] == "language_by_site") & metrics["level_b"].notna()]
    assert set(shifts["split"]) == {"site=A"} and shifts["m"].eq(2).all()


def test_each_number_carries_its_counts(full_run):
    row = find(full_run.metrics, model="fake-gen", measure="language", level_a="EN", level_b="FR",
               metric="chg", mode="valid", split=None)
    assert (row["n_lines"], row["n_samples"], row["n_units"]) == (12, 6, 4)
    assert (row["n_valid"], row["n_invalid"], row["n_error"], row["n_excluded"]) == (12, 0, 1, 2)
    strict = find(full_run.metrics, model="fake-gen", measure="language", level_a="EN", level_b="FR",
                  metric="chg", mode="strict", split=None)
    assert (strict["n_samples"], strict["n_invalid"], strict["n_excluded"]) == (7, 1, 1)


def test_significance_rule():
    weights = draw_weights(50, 500, 0)
    assert summarize(lambda w: w[:, 0] + 1, weights, 0.05, 1, True)["significant"] is True
    centered = summarize(lambda w: (w[:, 0] - 1) / 2, weights, 0.05, 1, True)
    assert centered["significant"] is False
    assert summarize(lambda w: w.sum(axis=1), weights, 0.05, 1, False)["significant"] is None


def test_more_comparisons_widen_the_interval():
    weights = draw_weights(100, 2000, 0)
    stat = lambda w: (w[:, 0] - 1) / 3  # noqa: E731
    one, many = summarize(stat, weights, 0.05, 1, True), summarize(stat, weights, 0.05, 66, True)
    assert many["ci_high"] - many["ci_low"] > one["ci_high"] - one["ci_low"]
    assert many["ci95_low"] == one["ci95_low"]


def test_bootstrap_draws_units_not_lines():
    """Two rows of the same unit stay together in each draw."""
    weights = draw_weights(10, 100, 0)
    assert (weights.sum(axis=1) == 10).all() and weights.shape == (100, 10)
    # Duplicating the rows of a unit must not reduce the uncertainty.
    single = lines_of([(f"u{i}", f"s{i}", "x", "positive", "yes" if i % 2 else "no", "valid") for i in range(40)])
    double = pd.concat([single, single.assign(sample_id=single["sample_id"] + "b")])

    def width(table):
        row = by_metric(base_rows(table, "level", "x", Stats(bootstrap=500, min_units=1), {}))["sens"]
        return row["ci_high"] - row["ci_low"]

    assert width(double) == pytest.approx(width(single), rel=1e-9)


def test_known_effect_is_found_and_interval_covers_it():
    """True value ΔBA = −0.1. The 95% interval must contain it about 95 times out of 100."""
    rng, runs, hits, values = np.random.default_rng(0), 300, 0, []
    n = 300
    for run in range(runs):
        truth = rng.integers(0, 2, n)
        answers = {"a": rng.random(n) < 0.9, "b": rng.random(n) < 0.8}  # right or wrong
        rows = [
            (i, i, level, "positive" if truth[i] else "negative",
             "yes" if bool(truth[i]) == bool(right[i]) else "no", "valid")
            for level, right in answers.items() for i in range(n)
        ]
        found = by_metric(shift_rows(lines_of(rows), "level", "a", "b", ["sample_id"],
                                     Stats(bootstrap=300, min_units=1, seed=run), 1, {}))["delta_ba"]
        values.append(found["value"])
        hits += found["ci95_low"] <= -0.1 <= found["ci95_high"]
    assert np.mean(values) == pytest.approx(-0.1, abs=0.01)
    assert 0.90 <= hits / runs <= 0.99


def test_pairing_keeps_only_samples_present_at_both_levels():
    table = lines_of([
        ("u1", "s1", "a", "positive", "yes", "valid"),
        ("u1", "s1", "b", "positive", "yes", "valid"),
        ("u2", "s2", "a", "positive", "yes", "valid"),  # no b
        ("u3", "s3", "a", "positive", "yes", "valid"),
        ("u3", "s3", "b", "positive", None, "INVALID"),
        ("u4", "s4", "a", "positive", "yes", "valid"),
        ("u4", "s4", "b", "positive", None, "error"),
    ])
    used, excluded = select(table, "level", "a", "b", ["sample_id"], ["valid"])
    assert set(used["sample_id"]) == {"s1"} and excluded == 3
    used, excluded = select(table, "level", "a", "b", ["sample_id"], ["valid", "INVALID"])
    assert set(used["sample_id"]) == {"s1", "s3"} and excluded == 2
    used, excluded = select(table, "level", "a", "b", None, ["valid"])
    assert len(used) == 5 and excluded == 0


def test_other_controlled_axes_stay_fixed():
    table = lines_of([
        ("u1", "s1", "EN", "positive", "yes", "valid"),
        ("u1", "s1", "FR", "positive", "no", "valid"),
        ("u1", "s1", "EN", "positive", "yes", "valid"),
        ("u1", "s1", "FR", "positive", "yes", "valid"),
        ("u2", "s2", "EN", "positive", "yes", "valid"),
        ("u2", "s2", "FR", "positive", "yes", "valid"),
    ]).assign(quality=["orig", "orig", "blur", "blur", "orig", "orig"])
    rows = shift_rows(table, "level", "EN", "FR", ["sample_id", "quality"], SETTINGS, 1, {})
    assert by_metric(rows)["chg"]["value"] == pytest.approx(1 / 3)


def test_pairs_are_made_on_the_other_controlled_axes():
    axes = {"language": Axis(type="controlled"), "quality": Axis(type="controlled"), "site": Axis(type="observed")}
    experiment = SimpleNamespace(axes=axes)
    assert pairing_keys("language", experiment) == ["sample_id", "quality"]
    assert pairing_keys("site", experiment) is None  # an observed axis has no pairs


def test_an_observed_shift_needs_enough_units_at_each_level():
    """2 units at a and 4 at b: 6 in total, but a rests on 2 only."""
    table = lines_of([(f"u{i}", f"s{i}", "a" if i < 2 else "b", "positive", "yes", "valid") for i in range(6)])
    rows = shift_rows(table, "level", "a", "b", None, Stats(bootstrap=200, min_units=3), 1, {})
    assert all(row["value"] is None for row in rows)


def test_a_number_rests_on_its_own_units():
    """40 negative units and 1 positive: the sensitivity rests on 1 unit, not on 41."""
    rows = [(f"u{i}", f"s{i}", "a", "negative", "no", "valid") for i in range(40)]
    rows.append(("u40", "s40", "a", "positive", "yes", "valid"))
    found = by_metric(base_rows(lines_of(rows), "level", "a", Stats(bootstrap=200, min_units=30), {}))
    assert found["spec"]["n_units"] == 40 and found["spec"]["value"] == 1
    assert found["sens"]["n_units"] == 1 and found["sens"]["value"] is None
    assert found["ba"]["value"] is None  # it needs the sensitivity


def test_one_line_is_not_estimable():
    rows = base_rows(lines_of([("u1", "s1", "a", "positive", "yes", "valid")]), "level", "a", SETTINGS, {})
    assert all(row["value"] is None for row in rows)


def test_only_invalid_answers():
    table = lines_of([(f"u{i}", f"s{i}", "a", "positive", None, "INVALID") for i in range(5)])
    rows = base_rows(table, "level", "a", SETTINGS, {})
    assert by_metric(rows)["sens"]["value"] is None  # no valid answer
    assert by_metric(rows, "strict")["sens"]["value"] == 0  # all counted as wrong
    assert by_metric(rows, "strict")["inv_rate"]["value"] == 1


def test_only_one_class_is_not_estimable_never_zero():
    table = lines_of([(f"u{i}", f"s{i}", "a", "positive", "yes", "valid") for i in range(5)])
    rows = by_metric(base_rows(table, "level", "a", SETTINGS, {}))
    assert rows["sens"]["value"] == 1
    assert rows["spec"]["value"] is None and rows["ba"]["value"] is None


def test_too_few_units_is_not_estimable():
    table = lines_of([(f"u{i}", f"s{i}", "a", "positive", "yes", "valid") for i in range(5)])
    rows = base_rows(table, "level", "a", Stats(bootstrap=200, min_units=30), {})
    assert all(row["value"] is None for row in rows)


def test_unknown_labels_only_count_for_answer_changes(full_run):
    """Images 7 and 8 are unknown. They count for the answer change, not for accuracy."""
    labels = dict(model="fake-clean", measure="language", level_a="EN", level_b="FR", mode="valid", split=None)
    assert find(full_run.metrics, metric="chg", **labels)["n_samples"] == 8
    assert find(full_run.metrics, metric="delta_sens", **labels)["value"] == pytest.approx(-1 / 3)


# --- Full analyses: rules and edge cases ---


def analyze(toy, run, change=None):
    """Run an analysis on a changed copy of analysis.yaml: the original does not move."""
    raw = read_yaml(toy / "analysis.yaml")
    (change or (lambda _: None))(raw)
    write_yaml(raw, toy / "changed.yaml")
    return do_analyze(load_analysis(toy / "changed.yaml", str(run)))


def test_same_analysis_finds_the_same_folder(full_run):
    assert analyze(full_run.folder, full_run.run) == full_run.report


def test_default_measures_one_per_axis(full_run):
    metrics = pd.read_parquet(analyze(full_run.folder, full_run.run, lambda raw: raw.pop("measures")).parent / "metrics.parquet")
    assert set(metrics["measure"]) == {"language", "site"}


def test_levels_restrict_the_comparison(full_run):
    def change(raw):
        raw["axes"]["language"] = {"reference": "FR", "levels": ["FR"]}
        raw["measures"] = [{"name": "one", "compare": "language"}]

    metrics = pd.read_parquet(analyze(full_run.folder, full_run.run, change).parent / "metrics.parquet")
    assert set(metrics["level_a"]) == {"FR"} and metrics["level_b"].isna().all()


def test_levels_restrict_the_cells_of_split_by(full_run):
    def change(raw):
        raw["axes"]["site"] = {"levels": ["A"]}
        raw["measures"] = [{"name": "by_site", "compare": "language", "split_by": ["site"]}]

    metrics = pd.read_parquet(analyze(full_run.folder, full_run.run, change).parent / "metrics.parquet")
    assert set(metrics["split"]) == {"site=A"}


def test_controlled_axis_with_one_level_sent_is_an_error(toy, edit):
    from test_inference import only, run

    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data[data["language"] == "EN"].to_csv(toy / "toy.csv", index=False)
    folder = run(edit(lambda raw: only(raw, "fake_generative")))
    with pytest.raises(VLMShiftError, match="single level in the variants"):
        analyze(toy, folder, lambda raw: raw.pop("latent"))


@pytest.mark.parametrize("change, message", [
    (lambda raw: raw["measures"].append({"name": "x", "compare": "absent"}), "absent"),
    (lambda raw: raw["axes"].update(language={"reference": "DE"}), "DE"),
    (lambda raw: raw.update(latent={"targets": ["absent"]}), "absent"),
])
def test_analysis_rules(full_run, change, message):
    with pytest.raises(VLMShiftError, match=message):
        analyze(full_run.folder, full_run.run, change)


def test_latent_without_vectors_is_an_error(toy, edit):
    from test_inference import only, run

    folder = run(edit(lambda raw: only(raw, "fake_generative")))
    with pytest.raises(VLMShiftError, match="embeddings"):
        analyze(toy, folder)


def test_constant_model_is_signaled(toy, edit):
    from vlmshift.extensions import get_adapter, register_adapter
    from test_inference import only, run

    class AlwaysYes(get_adapter("fake_generative")):
        def generate(self, images, prompts, params):
            return ["yes"] * len(images)

    register_adapter("always_yes", AlwaysYes)
    folder = run(edit(lambda raw: only(raw, "always_yes")))
    report = analyze(toy, folder, lambda raw: raw.pop("latent"))
    assert "constant_model" in set(pd.read_parquet(report.parent / "issues.parquet")["code"])


def test_an_observed_numeric_axis_is_compared_by_quartiles(toy, edit):
    from test_inference import only, run

    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data["age"] = data["image"].str.extract(r"img(\d)")[0].astype(int) * 10  # 8 ages, the same in EN and FR
    data.loc[data["image"].str.contains("img8"), "site"] = ""
    data.to_csv(toy / "toy.csv", index=False)
    folder = run(edit(lambda raw: (only(raw, "fake_generative", params={"faults": False}),
                                   raw["axes"].update(age={"type": "observed"}))))

    def change(raw):
        raw.pop("latent")
        raw["measures"] = [{"name": "age", "compare": "age"}, {"name": "site", "compare": "site"}]

    report = analyze(toy, folder, change)
    metrics = pd.read_parquet(report.parent / "metrics.parquet")
    ages = metrics[metrics["measure"] == "age"]
    assert ages["level_a"].nunique() == 4 and ages["level_b"].dropna().nunique() == 3  # 4 quartiles, 6 pairs
    pairs = ages.dropna(subset=["level_b"])
    low = {side: pairs[side].str.extract(r"\(([-\d.]+),")[0].astype(float) for side in ("level_a", "level_b")}
    assert (low["level_a"] < low["level_b"]).all()  # each pair goes from the lower quartile to the higher, Q1 -> Q2
    issues = pd.read_parquet(report.parent / "issues.parquet")
    assert "Axis site: 2 variants without a value" in " ".join(issues["message"])
    sites = metrics[metrics["measure"] == "site"]
    assert set(pd.concat([sites["level_a"], sites["level_b"]]).dropna()) == {"A", "B"}  # an empty value is left out, never a level of its own


def test_a_column_of_the_tool_cannot_be_probed(full_run):
    with pytest.raises(VLMShiftError, match=r"\['unit'\] cannot be probed"):
        analyze(full_run.folder, full_run.run, lambda raw: raw["latent"].update(targets=["unit"]))


def test_the_yes_rate_counts_only_valid_answers():
    """Defined on valid answers: an INVALID row is neither yes nor no."""
    from vlmshift.stats.metrics import METRICS, unit_sums

    lines = pd.DataFrame({"u": [0, 0, 0, 0], "y": [1.0, 1.0, 0.0, 0.0], "yhat": [1.0, 0.0, np.nan, np.nan]})
    assert METRICS["yes_rate"](unit_sums(lines, 1))[0] == 0.5


def test_a_controlled_numeric_axis_keeps_its_levels():
    from vlmshift.common import to_levels

    blur = pd.Series(["0", "1", "2", "3", "4", "5"], dtype="string")
    assert to_levels(blur, observed=False).tolist() == blur.tolist()
    assert to_levels(blur, observed=True).nunique() == 4  # an observed axis is cut into quartiles


def test_an_axis_setting_no_measure_uses_is_an_error(full_run):
    def change(raw):
        raw["axes"]["site"] = {"levels": ["A"]}  # as if it restricted every measure to site A
        raw["measures"] = [{"name": "language", "compare": "language"}]

    with pytest.raises(VLMShiftError, match=r"no measure compares or splits by \['site'\]"):
        analyze(full_run.folder, full_run.run, change)


def test_a_reference_absent_from_the_data_is_an_error_even_unused(full_run):
    def change(raw):
        raw["axes"]["site"] = {"reference": "Z"}  # no site Z in the toy
        raw["measures"] = [{"name": "language", "compare": "language"}]

    with pytest.raises(VLMShiftError, match=r"Axis site: levels missing from the data: \['Z'\]"):
        analyze(full_run.folder, full_run.run, change)
