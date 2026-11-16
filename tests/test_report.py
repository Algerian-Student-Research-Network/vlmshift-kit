"""report/: step 6, each number identical to the one in the metrics table."""
from types import SimpleNamespace

import pandas as pd
import pytest

from vlmshift.cli import do_analyze, do_audit, do_prepare, do_run, main
from vlmshift.config.load import load_analysis, load_experiment, read_yaml, write_yaml
from vlmshift.extensions import register_adapter
from vlmshift.inference.runner import run_inference
from vlmshift.report.export import coverage, probe_section, show, summary_section


def text(full_run):
    return (full_run.report / "report.md").read_text(encoding="utf-8")


def test_every_number_is_identical_to_the_metrics_table(full_run):
    # Read as text, then converted: the fast pandas reader rounds the last digit.
    exported = pd.read_csv(full_run.report / "metrics.csv", dtype={"value": str, "ci_low": str})
    for column in ("value", "ci_low"):
        numbers = exported[column].astype(float)
        pd.testing.assert_series_equal(numbers, full_run.metrics[column], check_exact=True)
    assert exported["display"].tolist() == [show(row) for row in full_run.metrics.to_dict("records")]


def test_matrix_crosses_cells_and_levels(full_run):
    matrix = pd.read_csv(full_run.report / "matrix_language_by_site.csv")
    assert {"EN", "FR", "EN -> FR"} <= set(matrix.columns)
    row = matrix[(matrix["model"] == "fake-gen") & (matrix["metric"] == "chg")
                 & (matrix["mode"] == "valid") & (matrix["split"] == "site=A")].iloc[0]
    assert row["EN -> FR"].startswith("0.500")
    assert "units" in row["EN -> FR"]  # counts shown


def test_not_estimable_is_displayed(full_run):
    matrix = pd.read_csv(full_run.report / "matrix_language_by_site.csv")
    row = matrix[(matrix["model"] == "fake-gen") & (matrix["metric"] == "delta_ba")
                 & (matrix["mode"] == "valid") & (matrix["split"] == "site=B")].iloc[0]
    assert row["EN -> FR"].startswith("not estimable")


def test_probe_not_estimable_is_displayed():
    shown = probe_section(pd.DataFrame({"model": ["m"], "score": [float("nan")]}))[0]
    assert "not estimable" in shown and "NaN" not in shown


def test_sections_and_notice(full_run):
    report = text(full_run)
    for section in ("Summary", "Audit", "Results by measure", "Latent space probe",
                    "Reading the answers", "Coverage limits"):
        assert f"## {section}" in report
    assert "these numbers are only for testing the tool" in report
    assert "8 samples, from 2 to 2 variants per sample, 0 incomplete." in report


def test_summary_speaks_plainly(full_run):
    report = text(full_run)
    assert "**fake-gen**" in report
    assert "- language, EN to FR: balanced accuracy falls by 0.417 (likely between -0.500 and -0.159, 2 units)." in report
    assert "- language_by_site, EN to FR, site=A: 50% of the answers changed" in report
    assert "**Few units**" in report


def shift(metric="delta_ba", mode="valid", value=-0.2, significant=True, n_units=50):
    return {
        "measure": "m", "model": "x", "split": None, "level_a": "A", "level_b": "B", "metric": metric, "mode": mode,
        "value": value, "ci_low": value - 0.1, "ci_high": value + 0.1, "significant": significant, "n_units": n_units,
    }


def summary(*rows, covered=None):
    data = SimpleNamespace(log={"run": "r"}, analysis="a", experiment=SimpleNamespace(models=[1]), variants=[1])
    covered = pd.DataFrame(covered or [], columns=["model", "level", "expected", "valid", "invalid", "error", "missing", "coverage"])
    return "\n".join(summary_section(data, pd.DataFrame(rows), covered))


def level(name, valid, expected=4, model="x"):
    return {"model": model, "level": name, "expected": expected, "valid": valid, "invalid": expected - valid,
            "error": 0, "missing": 0, "coverage": valid / expected}


def test_summary_says_when_nothing_could_be_tested():
    written = summary(shift(significant=None))
    assert "No shift could be tested" in written and "No significant shift" not in written


def test_summary_names_the_levels_with_few_usable_answers():
    written = summary(shift(), covered=[level("(all)", 4, 10), level("language = EN", 4), level("language = FR", 0, 6)])
    assert "- x, language = FR: 0 valid answer(s) out of 6 (6 INVALID, 0 failed, 0 not asked yet)." in written
    assert "language = EN" not in written and "(all)" not in written


def test_half_of_the_answers_is_enough():
    assert "Few usable answers" not in summary(shift(), covered=[level("language = FR", 2)])
    assert "Few usable answers" in summary(shift(), covered=[level("language = FR", 1)])


def test_summary_gives_each_direction_its_words():
    assert "balanced accuracy rises by 0.200" in summary(shift(value=0.2))
    assert "says yes less often, by 0.200" in summary(shift("delta_yes_rate"))
    assert "says yes more often, by 0.200" in summary(shift("delta_yes_rate", value=0.2))
    assert "unreadable answers rise by 0.200" in summary(shift("delta_inv_rate", "strict", value=0.2))
    assert "20% of the answers changed (likely between 10% and 30%" in summary(shift("chg", value=0.2))


def test_summary_keeps_only_the_main_significant_shifts():
    assert "No significant shift" in summary(shift(significant=False))
    assert "No significant shift" in summary(shift("delta_sens"))  # a detail: it stays in metrics.csv
    assert "No significant shift" in summary(shift(mode="strict"))  # the strict version of a main measure


def test_summary_warns_only_when_units_are_few():
    assert "Few units" not in summary(shift(n_units=30))
    assert "Few units" in summary(shift(n_units=29))


def test_summary_shows_the_largest_shifts_first_and_stops():
    rows = [shift(value=-0.001 * (number + 1)) for number in range(35)]
    written = summary(*rows)
    assert written.index("falls by 0.035") < written.index("falls by 0.034")
    assert "And 5 more in `metrics.csv`." in written
    assert "falls by 0.005" not in written


def test_significant_shift_is_marked_by_a_sign(full_run):
    significant = pd.read_csv(full_run.report / "metrics.csv")
    significant = significant[significant["significant"].eq(True)]
    assert len(significant) and significant["display"].str.contains(r"\*").all()


def test_every_table_exists_in_csv_and_parquet(full_run):
    for name in ("metrics", "probes", "audit_counts", "coverage", "reader_share", "reader_validation", "issues"):
        assert (full_run.report / f"{name}.csv").exists() and (full_run.report / f"{name}.parquet").exists()


def test_coverage_by_level_shows_where_a_model_fails(full_run):
    coverage = pd.read_csv(full_run.report / "coverage.csv").set_index(["model", "level"])
    assert coverage.loc[("fake-gen", "language = FR"), ["valid", "invalid", "error"]].tolist() == [6, 1, 1]
    assert coverage.loc[("fake-gen", "language = EN"), "coverage"] == 1
    assert {"site = A", "site = B"} <= set(coverage.loc["fake-gen"].index)


def test_coverage_cuts_a_numeric_axis_into_quartiles():
    variants = pd.DataFrame({"id": list("abcdefgh"), "age": [20, 30, 40, 50, 60, 70, 80, 90]})
    results = pd.DataFrame({"id": list("abcdefgh"), "model": "m", "status": ["valid"] * 6 + ["INVALID"] * 2})
    table = coverage(variants, results, ["m"], ["age"])
    assert len(table) == 5 and table["expected"].tolist() == [8, 2, 2, 2, 2]
    assert table["valid"].tolist() == [6, 2, 2, 2, 0]


def test_coverage(full_run):
    coverage = pd.read_csv(full_run.report / "coverage.csv")
    coverage = coverage[coverage["level"] == "(all)"].set_index("model")
    assert coverage.loc["fake-gen", ["valid", "invalid", "error", "missing"]].tolist() == [14, 1, 1, 0]
    assert coverage.loc["fake-clean", "coverage"] == 1
    assert coverage.loc["fake-gen", "coverage"] == pytest.approx(14 / 16)
    assert "Partial coverage" in text(full_run)


def test_reader_share_by_model_and_level(full_run):
    share = pd.read_csv(full_run.report / "reader_share.csv")
    assert {"fake-gen", "fake-clean"} == set(share["model"])
    assert share["llm_share"].eq(0).all()  # the list read everything


def test_reader_accuracy_by_language_is_shown(full_run):
    assert "accuracy" in text(full_run)
    assert "not validated" not in text(full_run)


def test_unvalidated_reader_is_flagged(toy, edit):
    from test_inference import only, run

    def change(raw):
        only(raw, "fake_generative", params={"faults": False})
        raw["task"]["reader_llm"] = {"model": "fake_reader"}

    folder = run(edit(change))
    raw = read_yaml(toy / "analysis.yaml")
    del raw["latent"]
    write_yaml(raw, toy / "analysis.yaml")
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(folder)))
    content = (report / "report.md").read_text(encoding="utf-8")
    assert "not validated" in content
    assert "No probe computed" in content


def test_interrupted_run_still_gives_a_report(toy, edit):
    from test_inference import Interrupted, only

    register_adapter("interrupted", Interrupted)
    Interrupted.batches = 0
    config = edit(lambda raw: only(raw, "interrupted", params={"faults": False}))
    variants = do_prepare(config)
    with pytest.raises(KeyboardInterrupt):
        run_inference(config, variants)
    folder = next((toy / "workspace" / "runs").iterdir())
    raw = read_yaml(toy / "analysis.yaml")
    del raw["latent"]
    write_yaml(raw, toy / "analysis.yaml")
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(folder)))
    coverage = pd.read_csv(report / "coverage.csv").iloc[0]
    assert (coverage["valid"], coverage["missing"]) == (4, 12)
    assert "Partial coverage" in (report / "report.md").read_text(encoding="utf-8")


def test_command_line_chains_every_step(toy, capsys):
    main(["all", str(toy / "experiment.yaml"), str(toy / "analysis.yaml")])
    assert "report.md" in capsys.readouterr().out
    run = next((toy / "workspace" / "runs").iterdir())
    main(["analyze", str(toy / "analysis.yaml"), "--run", str(run)])


def test_command_line_reports_errors_clearly(toy):
    with pytest.raises(SystemExit, match="Error"):
        main(["import", str(toy / "broken.yaml")])


def test_the_summary_names_what_could_not_be_estimated(full_run):
    report = text(full_run)
    assert "**Not estimable**, for lack of units, of complete samples or of valid answers" in report
    assert "- fake-gen, site: " in report


def test_rows_left_out_before_the_models_are_counted(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[0, "image"] = "images/absent.png"
    data.to_csv(toy / "toy.csv", index=False)
    run = do_run(load_experiment(toy / "experiment.yaml"))
    report = (do_analyze(load_analysis(toy / "analysis.yaml", str(run))) / "report.md").read_text(encoding="utf-8")
    assert "**Left out before the models**: 1 row(s) (image_missing: 1)." in report


def one_clean_model(raw):
    raw["task"].pop("reader_llm")
    raw["models"] = [{"name": "clean", "adapter": "fake_generative", "params": {"faults": False}}]


def test_a_report_keeps_the_audit_of_its_own_experiment(toy):
    raw = read_yaml(toy / "experiment.yaml")
    one_clean_model(raw)
    raw["axes"] = {"language": {"type": "controlled"}}
    write_yaml(raw, toy / "one.yaml")
    raw["axes"]["site"] = {"type": "observed"}
    write_yaml(raw, toy / "two.yaml")
    analysis = read_yaml(toy / "analysis.yaml")
    analysis.pop("latent")
    analysis["measures"] = [{"name": "language", "compare": "language"}]
    write_yaml(analysis, toy / "one_analysis.yaml")
    run = do_run(load_experiment(toy / "one.yaml"))
    do_audit(load_experiment(toy / "two.yaml"))  # the same table, audited with another axis
    report = do_analyze(load_analysis(toy / "one_analysis.yaml", str(run)))
    assert set(pd.read_csv(report / "audit_counts.csv")["axis"]) == {"language"}


def test_the_quartiles_of_the_report_are_those_of_the_audit(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data["age"] = data["image"].str.extract(r"img(\d)")[0].astype(int).mul(10).add(10).astype(str)
    data.loc[data["image"].str.contains("img7|img8"), "question"] = ""  # two images are left out at step 3
    data.to_csv(toy / "toy.csv", index=False)

    def change(raw):
        one_clean_model(raw)
        raw["axes"]["age"] = {"type": "observed"}

    run = do_run(edit(change))
    analysis = read_yaml(toy / "analysis.yaml")
    analysis.pop("latent")  # no model with vectors here
    analysis["measures"].append({"name": "age", "compare": "age"})
    write_yaml(analysis, toy / "analysis.yaml")
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(run)))
    audit = pd.read_csv(report / "audit_counts.csv").query("axis == 'age'")["level"]
    covered = pd.read_csv(report / "coverage.csv")["level"]
    covered = set(covered[covered.str.startswith("age = ")])
    assert covered and covered <= {f"age = {level}" for level in audit}
    measured = pd.read_parquet(report.parent / "metrics.parquet").query("measure == 'age'")
    assert len(measured) and set(measured["level_a"]) <= set(audit)
    shares = pd.read_csv(report / "reader_share.csv").query("axis == 'age'")["level"]
    assert len(shares) and set(shares) <= set(audit)


# --- Figures, HTML page, provenance ---


def test_figures_exist_as_svg_and_pdf_and_print_their_numbers(full_run):
    for name in ("fig_language_ba", "fig_language_shifts", "fig_site_ba", "fig_language_by_site_ba"):
        svg = (full_run.report / f"{name}.svg").read_text(encoding="utf-8")
        assert svg.startswith("<svg") and "n=" in svg  # each number comes with its units
        assert (full_run.report / f"{name}.pdf").read_bytes().startswith(b"%PDF-1.4")
    heatmap = (full_run.report / "fig_language_ba.svg").read_text(encoding="utf-8")
    value = full_run.metrics.query("measure == 'language' and metric == 'ba' and mode == 'valid' and model == 'fake-clean' and level_a == 'EN'")
    assert f"{value['value'].iloc[0]:.3f}" in heatmap


def test_a_run_of_the_fake_models_is_stamped_as_synthetic(full_run):
    written = (full_run.report / "report.md").read_text(encoding="utf-8")
    assert "**Synthetic run**" in written and "Research use only" in written
    assert "SYNTHETIC DATA: NOT FOR PUBLICATION" in (full_run.report / "fig_language_ba.svg").read_text(encoding="utf-8")


def test_figures_of_a_real_model_carry_no_stamp(full_run):
    from vlmshift.report.figures import make_figures

    figures = make_figures(full_run.metrics, synthetic=False)
    assert all("SYNTHETIC" not in drawing.svg() for drawing in figures.values())


def test_the_html_page_holds_the_report_and_its_figures(full_run):
    page = (full_run.report / "report.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and '<h2 id="summary">Summary</h2>' in page
    assert page.count("<svg") == len(list(full_run.report.glob("fig_*.svg"))) and '<a href="#figures">Figures</a>' in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "") and "https://" not in page  # nothing fetched


def test_provenance_says_where_the_report_comes_from(full_run):
    import json

    found = json.loads((full_run.report / "provenance.json").read_text(encoding="utf-8"))
    assert found["run"] == full_run.run.name and found["analysis"] == full_run.analysis.name
    assert found["synthetic"] is True and found["code_version"]
    assert found["axes"]["language"] == {"type": "controlled", "levels": ["EN", "FR"]}
    assert set(found["models"]) == {"fake-gen", "fake-clean", "fake-clip"} and found["models"]["fake-clean"]["coverage"] == 1
    assert {"report.md", "report.html", "metrics.csv", "provenance.json", "fig_language_ba.pdf"} <= set(found["files"])


def test_the_report_does_not_publish_local_folders(toy, edit):
    """An image that is missing is reported with its file name, not with the folders of this machine."""
    from vlmshift.report.export import shareable

    assert shareable(r"Image not found, row excluded: C:\Users\someone\data\img9.png") == "Image not found, row excluded: .../img9.png"
    assert shareable("Image not found: /home/someone/data/img9.png") == "Image not found: .../img9.png"
    assert shareable("EN/FR: 3 units") == "EN/FR: 3 units"
    table = pd.read_csv(toy / "toy.csv")
    table.loc[0, "image"] = "images/absent.png"
    table.to_csv(toy / "toy.csv", index=False)
    run = do_run(load_experiment(toy / "experiment.yaml"))
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(run)))
    messages = pd.read_csv(report / "issues.csv")["message"]
    assert messages.str.contains("absent.png").any() and not messages.str.contains(str(toy.parent), regex=False).any()


def test_the_pdf_of_a_figure_is_well_formed(full_run):
    """Each entry of the cross-reference table points at its object, and the drawing can be read back."""
    import re
    import zlib

    data = (full_run.report / "fig_language_ba.pdf").read_bytes()
    start = int(re.search(rb"startxref\n(\d+)\n%%EOF", data)[1])
    assert data[start:start + 4] == b"xref"
    offsets = [int(entry) for entry in re.findall(rb"(\d{10}) 00000 n ", data[start:])]
    assert len(offsets) == 6
    for number, offset in enumerate(offsets, 1):
        assert data[offset:].startswith(b"%d 0 obj" % number)
    length = int(re.search(rb"/Length (\d+)", data)[1])
    begin = data.index(b"stream\n") + len(b"stream\n")
    drawn = zlib.decompress(data[begin:begin + length]).decode("latin-1")
    assert "(ba by language) Tj" in drawn and "(fake-clean) Tj" in drawn and " re f" in drawn
