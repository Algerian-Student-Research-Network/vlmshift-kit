"""audit/: step 2, exact numbers on a known distribution."""
from pathlib import Path

import pandas as pd

from conftest import FIXTURES
from vlmshift.audit.counts import run_audit
from vlmshift.common import to_levels
from vlmshift.audit.samples import sample_table
from vlmshift.config.load import load_experiment
from vlmshift.data.table import build_table


def audit(toy):
    config = load_experiment(toy / "experiment.yaml")
    table = build_table(config)
    return run_audit(config, table), table


def test_counts_are_exactly_those_expected(toy):
    counts, _ = audit(toy)
    expected = pd.read_csv(FIXTURES / "counts.csv")
    pd.testing.assert_frame_equal(counts[expected.columns.tolist()], expected, check_dtype=False)
    assert counts["positive_rate"].dropna().eq(0.5).all()


def test_table_is_not_modified(toy):
    config = load_experiment(toy / "experiment.yaml")
    table = build_table(config)
    before = table.df.copy()
    run_audit(config, table)
    pd.testing.assert_frame_equal(table.df, before)


def test_low_counts_are_signaled(toy):
    _, table = audit(toy)
    issues = pd.read_parquet(table.dir / "audit" / "issues.parquet")
    assert issues["code"].eq("low_count").all() and len(issues) == 8  # all below 30 units


def test_empty_cell_is_kept_and_signaled(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data[(data["language"] == "EN") | (data["site"] == "A")].to_csv(toy / "toy.csv", index=False)
    counts, _ = audit(toy)
    assert counts.set_index("level").loc["FR x B", "n_lines"] == 0


def test_numeric_column_is_cut_in_quartiles():
    ages = pd.Series([str(age) for age in range(20, 60)], dtype="string")
    assert to_levels(ages).nunique() == 4
    assert to_levels(pd.Series(["EN", "FR"], dtype="string")).tolist() == ["EN", "FR"]


def test_quartiles_sorted_as_text_keep_the_order_of_the_numbers():
    ages = pd.Series([str(age) for age in range(5, 100, 5)], dtype="string")  # bounds 4.999, 27.5, 50.0, 72.5
    assert sorted(to_levels(ages).unique()) == [
        "Q1 (4.999, 27.5]", "Q2 (27.5, 50.0]", "Q3 (50.0, 72.5]", "Q4 (72.5, 95.0]",
    ]  # without Q1 to Q4 the text order was (27.5, (4.999, (50.0, (72.5


def test_incomplete_samples_show_their_missing_levels(toy):
    _, table = audit(toy)
    assert sample_table(table.df, ["language"])["missing"].isna().all()
    partial = table.df[~((table.df["language"] == "FR") & table.df["image"].str.endswith("img1.png"))]
    samples = sample_table(partial, ["language"])
    assert samples["missing"].dropna().tolist() == ["language=FR"]
    assert sorted(samples["n_variants"].unique()) == [1, 2]


def test_a_missing_combination_of_two_controlled_axes_is_seen():
    table = pd.DataFrame({
        "sample_id": ["s1"] * 3 + ["s2"] * 4,
        "language": ["EN", "EN", "FR", "EN", "EN", "FR", "FR"],
        "quality": ["orig", "blur", "orig", "orig", "blur", "orig", "blur"],
    })
    samples = sample_table(table, ["language", "quality"]).set_index("sample_id")
    assert samples.loc["s1", "missing"] == "language=FR, quality=blur"
    assert pd.isna(samples.loc["s2", "missing"])


def test_the_audit_explains_incomplete_samples(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(translator=data["language"].map({"EN": "ann", "FR": "bob"})).to_csv(toy / "toy.csv", index=False)
    audit(toy)
    issues = pd.read_parquet(next(toy.glob("workspace/tables/*/audit/issues.parquet")))
    assert "16 of 16 samples lack a level" in " ".join(issues["message"])


def test_a_csv_open_in_a_spreadsheet_does_not_stop_the_step(toy, monkeypatch):
    write = pd.DataFrame.to_csv

    def locked(self, path, *args, **kwargs):
        if Path(path).name == "counts.csv":
            raise PermissionError(13, "The file is used by another process")
        return write(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", locked)
    counts, table = audit(toy)
    assert len(counts)
    assert "csv_locked" in set(pd.read_parquet(table.dir / "audit" / "issues.parquet")["code"])
    assert (table.dir / "audit" / "samples.csv").exists()
