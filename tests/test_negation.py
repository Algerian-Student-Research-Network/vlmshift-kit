"""Negated questions: the answer is counted for what it says about the finding."""
import pandas as pd
import pytest

from vlmshift.cli import do_analyze, do_prepare, do_run
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_analysis, load_experiment, write_yaml
from vlmshift.extensions import register_adapter
from vlmshift.inference.adapters.fake import FakeContrastive, FakeGenerative, has_circle, image_number


class Logical(FakeGenerative):
    """Understands the negation: says yes to "no circle" when there is none."""

    def _answer(self, number: int, prompt: str) -> str:
        return "yes" if has_circle(number) != ("no circle" in prompt) else "no"


class Deaf(FakeGenerative):
    """Does not hear the negation: answers as if the plain question had been asked."""

    def _answer(self, number: int, prompt: str) -> str:
        return "yes" if has_circle(number) else "no"


class LogicalPair(FakeContrastive):
    """Contrastive: the positive phrase of a negated row states the negated question."""

    def _scores(self, number: int, positive: str) -> tuple[float, float]:
        return (1.0, 0.0) if has_circle(number) != ("no circle" in positive) else (0.0, 1.0)


for name, cls in (("fake_logical", Logical), ("fake_deaf", Deaf), ("fake_logical_pair", LogicalPair)):
    register_adapter(name, cls)


@pytest.fixture
def negation(toy):
    plain = dict(question="Is there a circle?", positive_text="a circle", negative_text="without a circle", negated="0")
    reverse = dict(question="Is there no circle?", positive_text="there is no circle", negative_text="a circle", negated="1")
    rows = [{"image": f"images/img{n}.png", "label": "positive" if has_circle(n) else "negative", "subject": f"s{n}", **texts}
            for n in range(1, 9) for texts in (plain, reverse)]
    pd.DataFrame(rows).to_csv(toy / "negation.csv", index=False)
    write_yaml({
        "schema_version": 1, "data": "negation.csv",
        "roles": {"unit": "subject", "positive_text": "positive_text", "negative_text": "negative_text"},
        "axes": {"negated": {"type": "controlled"}},
        "task": {"type": "yes_no", "negated": "negated"},
        "models": [{"name": "logical", "adapter": "fake_logical"}, {"name": "deaf", "adapter": "fake_deaf"},
                   {"name": "pair", "adapter": "fake_logical_pair"}],
    }, toy / "negation.yaml")
    write_yaml({"schema_version": 1, "run": "x", "axes": {"negated": {"reference": "0"}},
                "stats": {"bootstrap": 100, "min_units": 2}}, toy / "negation_analysis.yaml")
    return toy


def metrics_of(folder) -> pd.DataFrame:
    run = do_run(load_experiment(folder / "negation.yaml"))
    report = do_analyze(load_analysis(folder / "negation_analysis.yaml", str(run)))
    metrics = pd.read_parquet(report.parent / "metrics.parquet")
    return metrics[metrics["mode"] == "valid"].set_index(["model", "metric", "level_a"])["value"]


def test_a_model_that_understands_the_negation_keeps_its_answers(negation):
    values = metrics_of(negation)
    for model in ("logical", "pair"):  # generative and contrastive
        assert values[model, "chg", "0"] == 0
        assert values[model, "ba", "1"] == 1 and values[model, "ba", "0"] == 1


def test_a_model_deaf_to_the_negation_contradicts_itself(negation):
    values = metrics_of(negation)
    assert values["deaf", "chg", "0"] == 1
    assert values["deaf", "ba", "0"] == 1 and values["deaf", "ba", "1"] == 0
    assert values["deaf", "delta_ba", "0"] == -1


def test_the_raw_decision_is_kept_as_the_model_said_it(negation):
    """The results table keeps the literal answer: only the analysis turns it around."""
    run = do_run(load_experiment(negation / "negation.yaml"))
    results = pd.read_parquet(run / "results.parquet").query("model == 'deaf'")
    table = pd.read_parquet(next(negation.glob("workspace/tables/*/table.parquet")))
    lines = results.merge(table, on="id")
    assert (lines["decision"].eq("yes") == lines["label"].eq("positive")).all()


def test_an_unclear_negation_value_stops_before_the_models(negation):
    table = pd.read_csv(negation / "negation.csv", dtype=str)
    table.loc[0, "negated"] = "maybe"
    table.to_csv(negation / "negation.csv", index=False)
    with pytest.raises(VLMShiftError, match="task.negated: the column negated must say"):
        do_prepare(load_experiment(negation / "negation.yaml"))
