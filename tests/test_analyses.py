"""The further analyses (analyses block): values worked out by hand on scripted models."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from vlmshift.cli import do_analyze, do_run
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_analysis, load_experiment, write_yaml
from vlmshift.extensions import register_adapter
from vlmshift.inference.adapters.fake import FakeGenerative, has_circle
from vlmshift.stats.paired import cochran_q, detectable, holm, mcnemar
from vlmshift.stats.regression import regression_rows, variance_rows
from vlmshift.stats.robustness import link_rows

LANGUAGES = ("EN", "FR", "DE")


class Scripted(FakeGenerative):
    """Right everywhere, except on the images listed for a language: params.wrong = {FR: [5, 6]}.
    On a blank image (the question says so), it follows params.blank: "no", or "same" as on the real one."""

    def _answer(self, number: int, prompt: str) -> str:
        language = prompt[:2]
        yes = has_circle(number) != (number in self.params.get("wrong", {}).get(language, []))
        if "[blank]" in prompt and self.params.get("blank", "no") == "no":
            yes = False
        return "yes" if yes else "no"


register_adapter("fake_scripted", Scripted)

MODELS = [
    {"name": "site-fr", "adapter": "fake_scripted", "params": {"wrong": {"FR": [5, 6, 7, 8]}}},  # French fails at site B
    {"name": "first-en", "adapter": "fake_scripted", "params": {"wrong": {"FR": [1, 2, 3], "DE": [1]}}},
    {"name": "second", "adapter": "fake_scripted", "params": {"wrong": {"EN": [1], "FR": [1], "DE": [2]}}},
    {"name": "first-fr", "adapter": "fake_scripted", "params": {"wrong": {"EN": [1, 2], "DE": [3]}}},
]


def write_dataset(folder, conditions=("real",)):
    rows = [{"image": f"images/img{n}.png", "label": "positive" if has_circle(n) else "negative",
             "subject": f"s{n}", "site": "A" if n <= 4 else "B", "language": language, "condition": condition,
             "question": f"{language} Is there a circle?" + (" [blank]" if condition == "blank" else "")}
            for n in range(1, 9) for language in LANGUAGES for condition in conditions]
    pd.DataFrame(rows).to_csv(folder / "scripted.csv", index=False)


def analyze(folder, analyses, models=MODELS, axes=None, conditions=("real",), settings=None, latent=None):
    """Run the scripted models and one analyses block. Return analyses.parquet."""
    write_dataset(folder, conditions)
    controlled = {"language": {"type": "controlled"}, **({"condition": {"type": "controlled"}} if len(conditions) > 1 else {})}
    write_yaml({
        "schema_version": 1, "data": "scripted.csv", "roles": {"unit": "subject"},
        "axes": {**controlled, "site": {"type": "observed"}},
        "task": {"type": "yes_no"}, "models": models,
    }, folder / "scripted.yaml")
    write_yaml({
        "schema_version": 1, "run": "x", "measures": [{"name": "language", "compare": "language"}],
        "axes": axes or {"language": {"reference": "EN"}, "site": {"reference": "A"}},
        "stats": {"bootstrap": 100, "min_units": 2}, "analyses": analyses, **({"latent": latent} if latent else {}),
    }, folder / "scripted_analysis.yaml")
    run = do_run(load_experiment(folder / "scripted.yaml"))
    report = do_analyze(load_analysis(folder / "scripted_analysis.yaml", str(run)))
    return pd.read_parquet(report.parent / "analyses.parquet")


def pick(table, **labels):
    rows = table
    for column, value in labels.items():
        rows = rows[rows[column].isna() if value is None else rows[column] == value]
    assert len(rows) == 1, (labels, len(rows))
    return rows.iloc[0]


# --- Paired tests ---


def test_mcnemar_and_cochran_q_on_paired_languages(toy):
    """site-fr: right in English and German, wrong in French on the 4 images of site B."""
    found = analyze(toy, [{"type": "tests", "compare": "language"}]).query("model == 'site-fr'")
    french = pick(found, statistic="mcnemar_correct", level_a="EN", level_b="FR")
    assert french["p"] == pytest.approx(2 * 0.5 ** 4) and french["note"] == "exact"  # 4 discordant pairs, all one way
    assert (french["n"], french["n_units"]) == (8, 8)
    assert pick(found, statistic="discordant_correct", level_b="FR")["value"] == pytest.approx(0.5)
    german = pick(found, statistic="mcnemar_correct", level_a="EN", level_b="DE")
    assert german["p"] == 1 and german["note"] == "no discordant pair"
    q = pick(found, statistic="cochran_q_correct")
    assert q["value"] == pytest.approx(8) and q["p"] == pytest.approx(chi2.sf(8, 2))
    assert french["p_holm"] == pytest.approx(2 * french["p"])  # two McNemar tests for this model
    assert not found["experimental"].any()


def test_paired_tests_without_a_reference_compare_every_pair(toy):
    found = analyze(toy, [{"type": "tests", "compare": "language"}], axes={"site": {"reference": "A"}})
    pairs = found[found["statistic"] == "mcnemar_correct"].query("model == 'site-fr'")
    assert set(zip(pairs["level_a"], pairs["level_b"])) == {("DE", "EN"), ("DE", "FR"), ("EN", "FR")}


def test_test_formulas():
    assert mcnemar(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]))[:3] == (0.0, 1.0, 0)
    one_way = mcnemar(np.ones(30), np.zeros(30))  # 30 discordant pairs: the χ² form
    assert one_way[0] == pytest.approx(29 ** 2 / 30) and one_way[3] == "chi2"
    assert cochran_q(np.ones((5, 3)))[1] == 1.0 and math.isnan(cochran_q(np.ones((5, 2)))[0])
    assert holm([0.01, 0.04, 0.03, float("nan")])[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(holm([0.01, float("nan")])[1])
    assert detectable(100, 0.25, 0.05, 0.8) == pytest.approx((1.959964 + 0.841621) * 0.05, rel=1e-4)


# --- Interaction, consistency, information ---


def test_interaction_between_site_and_language(toy):
    """Accuracy 1 everywhere but 0 for French at site B: the two shifts do not add up."""
    found = analyze(toy, [{"type": "interaction", "axes": ["site", "language"]}]).query("model == 'site-fr'")
    french = found[found["level_b"] == "site=B, language=FR"].set_index("statistic")["value"]
    assert french.to_dict() == pytest.approx({"interaction": -1, "main_site": 0, "main_language": 0, "joint": -1})
    german = found[found["level_b"] == "site=B, language=DE"].set_index("statistic")["value"]
    assert german["interaction"] == 0
    row = pick(found, statistic="interaction", level_b="site=B, language=FR")
    assert row["level_a"] == "site=A, language=EN" and row["ci_low"] <= -1 <= row["ci_high"]
    assert found["experimental"].all()


def test_interaction_of_three_axes(toy):
    """site-fr with a blank image answering no: French at site B is wrong on the real image only for
    the questions it flips, so the interaction of language and site changes with the image."""
    found = analyze(
        toy, [{"type": "interaction", "axes": ["site", "language", "condition"]}], conditions=("real", "blank"),
        axes={"language": {"reference": "EN"}, "site": {"reference": "A"}, "condition": {"reference": "real"}},
    ).query("model == 'site-fr'")
    french = found[found["level_b"] == "site=B, language=FR, condition=blank"].set_index("statistic")["value"]
    # Real image: 0 − 1 − 1 + 1 = −1. Blank image: every answer is no, accuracy 0.5 in each cell: 0.
    assert french["interaction_at_reference"] == pytest.approx(-1)
    assert french["interaction_away"] == pytest.approx(0)
    assert french["interaction"] == pytest.approx(1)


def test_consistency_across_languages(toy):
    found = analyze(toy, [{"type": "consistency", "compare": "language"}]).query("model == 'site-fr'")
    values = found.set_index("statistic")["value"]
    assert values["invariant"] == pytest.approx(0.5)  # the 4 samples of site B change in French
    assert values["majority_acc"] == 1  # two languages out of three are still right
    assert values["entropy_bits"] == pytest.approx(0.5 * -(1 / 3 * math.log2(1 / 3) + 2 / 3 * math.log2(2 / 3)))
    assert values["vote_margin"] == pytest.approx(0.5 * 1 + 0.5 * 1 / 3)
    assert found["n"].eq(8).all()


def test_information_is_zero_when_the_language_tells_nothing(toy):
    found = analyze(toy, [{"type": "information", "compare": "language"}])
    assert found.query("model == 'site-fr'")["value"].iloc[0] == pytest.approx(0, abs=1e-9)  # half yes in each language


# --- Robustness ---


def test_worst_group_and_cvar(toy):
    found = analyze(toy, [{"type": "worst_group", "by": ["site", "language"], "tail": 0.34}]).query("model == 'site-fr'")
    assert pick(found, statistic="mean")["value"] == pytest.approx(5 / 6)
    worst = pick(found, statistic="worst_group")
    assert worst["value"] == 0 and worst["level_a"] == "site=B, language=FR"
    cvar = pick(found, statistic="cvar")  # the worst 34% of 6 groups: 3 groups, one at 0
    assert cvar["value"] == pytest.approx((0 + 1 + 1) / 3)


def test_standardized_accuracy_and_spread_between_languages(toy):
    found = analyze(toy, [{"type": "standardized", "by": ["site"], "over": "language"},
                          {"type": "heterogeneity", "by": ["site"], "over": "language"}]).query("model == 'site-fr'")
    assert pick(found, statistic="standardized_acc", level_a="site=B")["value"] == pytest.approx(4 / 6)
    assert pick(found, statistic="standardized_acc", level_a="site=A")["value"] == 1
    assert pick(found, name="heterogeneity", statistic="range", level_a="site=B")["value"] == 1
    assert pick(found, name="heterogeneity", statistic="std", level_a="site=B")["value"] == pytest.approx(math.sqrt(1 / 3))


def test_ranking_sees_the_order_of_the_models_reverse(toy):
    """In English first-en > second > first-fr (accuracy 1, 7/8, 6/8); in French it is the opposite."""
    three = MODELS[1:]
    found = analyze(toy, [{"type": "ranking", "compare": "language"}], models=three)
    french = found[found["level_b"] == "FR"].set_index("statistic")
    assert french.loc["kendall_tau", "value"] == pytest.approx(-1)
    assert french.loc["rank_reversals", "value"] == 3 and french.loc["best_model_changes", "value"] == 1
    assert french.loc["best_model_changes", "note"] == "best: first-en then first-fr"
    assert set(found["model"]) == {"(all models)"}


def test_pareto_frontier(toy):
    """perfect is right and stable everywhere: it dominates site-fr, which is alone below it."""
    models = [MODELS[0], {"name": "perfect", "adapter": "fake_scripted"}]
    found = analyze(toy, [{"type": "pareto", "by": ["site"], "compare": "language"}], models=models)
    optimal = found[found["statistic"] == "pareto_optimal"].set_index("model")["value"]
    assert optimal.to_dict() == {"perfect": 1, "site-fr": 0}
    assert pick(found, model="site-fr", statistic="stability")["value"] == pytest.approx(0.5)
    assert pick(found, model="site-fr", statistic="worst_group")["value"] == pytest.approx(2 / 3)


# --- Grounding ---

GROUNDING = dict(
    conditions=("real", "blank"),
    axes={"language": {"reference": "EN"}, "site": {"reference": "A"}, "condition": {"reference": "real"}},
    models=[MODELS[0], {"name": "blind", "adapter": "fake_scripted", "params": {"blank": "same"}}],
)


def test_disagreement_between_languages_under_a_blank_image(toy):
    found = analyze(toy, [{"type": "disagreement_shift", "compare": "language", "across": "condition"}], **GROUNDING)
    french = found[(found["model"] == "site-fr") & (found["level_a"] == "EN vs FR")].set_index("statistic")["value"]
    # Real image: French disagrees with English on the 4 samples of site B. Blank image: every answer is no.
    assert french.to_dict() == pytest.approx({"disagreement_reference": 0.5, "disagreement": 0, "delta_disagreement": -0.5})
    blind = found[(found["model"] == "blind") & (found["level_a"] == "EN vs FR")].set_index("statistic")["value"]
    assert blind["delta_disagreement"] == 0  # a model that does not look at the image keeps its answers


def test_taxonomy_tells_stable_from_grounded(toy):
    found = analyze(toy, [{"type": "taxonomy", "compare": "language", "across": "condition"}], **GROUNDING)
    blind = found[found["model"] == "blind"].set_index("statistic")["value"]
    assert blind["ungrounded_stable"] == 1 and blind["grounded_level_share"] == 0  # stable, because it ignores the image
    seeing = found[found["model"] == "site-fr"].set_index("statistic")["value"]
    # Site A: the 2 positives change on a blank image (grounded), the 2 negatives do not; all stable.
    # Site B: unstable in French; each sample changes at some language on a blank image.
    assert seeing.to_dict() == pytest.approx({
        "grounded_stable": 0.25, "ungrounded_stable": 0.25, "grounded_unstable": 0.5, "ungrounded_unstable": 0,
        "grounded_level_share": (2 * 1 + 2 * 0 + 2 * 2 / 3 + 2 * 1 / 3) / 8,
    })


# --- Regression and variance ---


def answers(rows) -> pd.DataFrame:
    """Rows (unit, group, other, right) → valid answers with a known label."""
    table = pd.DataFrame(rows, columns=["unit", "group", "other", "right"])
    return table.assign(status="valid", y=1.0, yhat=table["right"].astype(float))


def test_regression_gives_the_log_odds_of_a_group():
    """Group a: 3 right of 4. Group b: 1 right of 4. The coefficient is the difference of their log-odds."""
    rows = [(f"u{i}", group, "x", right) for i, (group, right) in enumerate(
        [("a", 1), ("a", 1), ("a", 1), ("a", 0), ("b", 1), ("b", 0), ("b", 0), ("b", 0)])]
    found = pd.DataFrame(regression_rows(answers(rows), ["group"], [], {"group": "a"}))
    slope = pick(found, statistic="log_odds", level_a="group=b")
    assert slope["value"] == pytest.approx(math.log(1 / 3) - math.log(3))
    assert pick(found, statistic="odds_ratio", level_a="group=b")["value"] == pytest.approx(1 / 9)
    assert pick(found, statistic="log_odds", level_a="(intercept)")["value"] == pytest.approx(math.log(3))
    assert slope["note"] == "ok" and slope["n_units"] == 8


def test_regression_errors_follow_the_units_not_the_rows():
    """Writing each answer twice for the same unit adds no information: same interval."""
    rows = [(f"u{i}", group, "x", right) for i, (group, right) in enumerate(
        [("a", 1), ("a", 1), ("a", 1), ("a", 0), ("b", 1), ("b", 0), ("b", 0), ("b", 0)])]
    once = pick(pd.DataFrame(regression_rows(answers(rows), ["group"], [], {})), statistic="log_odds", level_a="group=b")
    twice = pick(pd.DataFrame(regression_rows(answers(rows * 2), ["group"], [], {})), statistic="log_odds", level_a="group=b")
    assert twice["ci95_low"] == pytest.approx(once["ci95_low"]) and twice["p"] == pytest.approx(once["p"])


def test_regression_says_when_it_cannot_be_fitted():
    always = [(f"u{i}", group, "x", 1) for i, group in enumerate("aabb")]
    assert regression_rows(answers(always), ["group"], [], {})[0]["statistic"] == "not_estimable"
    split = [(f"u{i}", group, "x", right) for i, (group, right) in enumerate([("a", 1)] * 4 + [("b", 0)] * 4)]
    assert "separation" in regression_rows(answers(split), ["group"], [], {})[0]["note"]


def test_variance_goes_with_the_axis_that_matters():
    """Right in group a, wrong in group b, whatever the other axis."""
    rng = np.random.default_rng(0)
    rows = [(f"u{i}", group, other, int(rng.random() < (0.9 if group == "a" else 0.1)))
            for i, (group, other) in enumerate([(g, o) for g in "ab" for o in "xy"] * 50)]
    found = pd.DataFrame(variance_rows(answers(rows), ["group", "other"]))
    share = found[found["statistic"] == "partial_eta2"].set_index("level_a")
    assert share.loc["group", "value"] > 0.5 and share.loc["other", "value"] < 0.02
    assert share.loc["group", "p"] < 0.001 and share.loc["other", "p"] > 0.05
    assert "group x other" in share.index


def test_link_between_probe_and_interaction():
    probes = pd.DataFrame({"model": ["a", "b", "c"], "target": "site", "probe": [0.1, 0.2, 0.4]})
    earlier = [{"name": "site_language", "statistic": "interaction", "model": model, "value": value}
               for model, value in (("a", -0.1), ("b", 0.2), ("c", -0.5))]
    found = pd.DataFrame(link_rows(probes, earlier, "site", "site_language"))
    assert pick(found, statistic="spearman")["value"] == pytest.approx(1)  # the size of the interaction, not its sign
    assert link_rows(probes.head(2), earlier, "site", "site_language")[-1]["statistic"] == "mean_abs_interaction"


# --- The block itself ---


@pytest.mark.parametrize("analysis, message", [
    ({"type": "tests"}, "type tests needs"),
    ({"type": "interaction", "axes": ["language"]}, "between 2 or 3 axes"),
    ({"type": "tests", "compare": "language", "split_by": ["language"]}, "named twice"),
    ({"type": "tests", "compare": "site"}, "must be a controlled axis"),
    ({"type": "tests", "compare": "scanner"}, "axes missing from the run"),
    ({"type": "worst_group", "by": ["site"], "metric": "chg"}, "metric"),
    ({"type": "link", "target": "site", "of": "nothing"}, "must be one of latent.targets"),
])
def test_a_wrong_analysis_is_refused_with_its_reason(toy, analysis, message):
    with pytest.raises(VLMShiftError, match=message):
        analyze(toy, [analysis])


def test_an_interaction_needs_its_references(toy):
    with pytest.raises(VLMShiftError, match="give a reference level to .'site'."):
        analyze(toy, [{"type": "interaction", "axes": ["site", "language"]}], axes={"language": {"reference": "EN"}})


def test_two_analyses_of_one_type_need_names(toy):
    with pytest.raises(VLMShiftError, match="give each a name"):
        analyze(toy, [{"type": "tests", "compare": "language"}, {"type": "tests", "compare": "language"}])


def test_every_row_says_which_analysis_and_code_gave_it(toy):
    found = analyze(toy, [{"type": "tests", "compare": "language", "name": "by_site", "split_by": ["site"]},
                          {"type": "information", "compare": "language"}])
    assert set(found["name"]) == {"by_site", "information"}
    assert set(found.query("name == 'by_site'")["split"]) == {"site=A", "site=B"}
    assert found["run"].notna().all() and found["analysis"].notna().all() and found["code_version"].notna().all()
    assert found.groupby("name")["experimental"].first().to_dict() == {"by_site": False, "information": True}


def test_the_report_lists_the_analyses_and_marks_the_experimental_ones(toy):
    analyze(toy, [{"type": "tests", "compare": "language"}, {"type": "interaction", "axes": ["site", "language"]}])
    report = next(toy.glob("workspace/runs/*/analyses/*/report"))
    written = (report / "report.md").read_text(encoding="utf-8")
    assert "## Further analyses" in written
    assert "- `analysis_tests.csv`: tests," in written and "tests significant after Holm's correction (p < 0.05)." in written
    assert "`analysis_interaction.csv`: interaction" in written and "**Experimental**: not yet validated" in written
    exported = pd.read_csv(report / "analysis_tests.csv")
    assert {"statistic", "value", "p", "p_holm", "n"} <= set(exported.columns) and (report / "analyses.parquet").exists()


def test_without_the_block_the_report_says_so(full_run):
    written = (full_run.report / "report.md").read_text(encoding="utf-8")
    assert "No further analysis declared" in written
    assert pd.read_parquet(full_run.analysis / "analyses.parquet").empty
