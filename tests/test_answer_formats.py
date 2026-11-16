"""Several answer formats in one experiment: each row is read with the list of its format."""
import pandas as pd
import pytest

from vlmshift.cli import do_prepare
from vlmshift.common import VLMShiftError, canonical, sha16
from vlmshift.config.load import config_hash, load_experiment, read_yaml, write_yaml
from vlmshift.extensions import register_adapter
from vlmshift.inference.adapters.fake import FakeGenerative, has_circle, image_number
from vlmshift.inference.runner import run_inference

QUESTIONS = {
    "yes_no": "Is there a circle? Answer YES or NO.",
    "numeric": "Is there a circle? Answer 1 or 0.",
    "letter": "Is there a circle? Answer A or B.",
}
SETS = {
    "yes_no": {"yes": ["yes"], "no": ["no"]},
    "numeric": {"yes": ["1"], "no": ["0"]},
    "letter": {"yes": ["A"], "no": ["B"]},
}


class Formats(FakeGenerative):
    """Answers in the format the question asks for."""

    def _answer(self, number: int, prompt: str) -> str:
        yes = has_circle(number)
        if "1 or 0" in prompt:
            return "1" if yes else "0"
        if "A or B" in prompt:
            return "A." if yes else "b"
        return "Yes" if yes else "No"


class AlwaysOne(FakeGenerative):
    def generate(self, images, prompts, params):
        return ["1" for _ in images]


register_adapter("fake_formats", Formats)
register_adapter("fake_always_one", AlwaysOne)


@pytest.fixture
def formats(toy):
    """The toy images, each asked in three answer formats."""
    rows = [{"image": f"images/img{n}.png", "label": "positive" if has_circle(n) else "negative",
             "answer_format": name, "question": question, "subject": f"s{n}"}
            for n in range(1, 5) for name, question in QUESTIONS.items()]
    pd.DataFrame(rows).to_csv(toy / "formats.csv", index=False)

    def build(adapter="fake_formats", **task):
        write_yaml({
            "schema_version": 1, "data": "formats.csv", "roles": {"unit": "subject"},
            "axes": {"answer_format": {"type": "controlled"}},
            "task": {"type": "yes_no", "answer_sets": SETS, "answers_by": "answer_format", **task},
            "models": [{"name": "m", "adapter": adapter}],
        }, toy / "formats.yaml")
        return load_experiment(toy / "formats.yaml")

    return build


def results_of(config) -> pd.DataFrame:
    variants = do_prepare(config)
    folder = run_inference(config, variants)
    return pd.read_parquet(folder / "results.parquet").merge(variants.df, on="id")


def test_each_format_is_read_with_its_own_list(formats):
    lines = results_of(formats())
    assert lines["status"].eq("valid").all()
    right = lines["decision"].eq("yes") == lines["label"].eq("positive")
    assert right.all() and set(lines["answer_format"]) == set(QUESTIONS)


def test_an_answer_of_another_format_is_invalid_never_a_no(formats):
    """A model that says 1 to everything is only understood where 1 was asked for."""
    lines = results_of(formats("fake_always_one"))
    assert lines.groupby("answer_format")["status"].agg(set).to_dict() == {
        "yes_no": {"INVALID"}, "letter": {"INVALID"}, "numeric": {"valid"},
    }
    assert lines["decision"].dropna().eq("yes").all()


def test_a_format_without_a_list_stops_before_the_models(formats, toy):
    table = pd.read_csv(toy / "formats.csv")
    table.loc[0, "answer_format"] = "emoji"
    table.to_csv(toy / "formats.csv", index=False)
    with pytest.raises(VLMShiftError, match=r"no list for these values of the column answer_format: \['emoji'\]"):
        do_prepare(formats())


def test_lists_and_column_go_together(formats):
    with pytest.raises(VLMShiftError, match="answer_sets and answers_by go together"):
        formats(answers_by=None)


def test_keys_added_later_do_not_move_an_older_fingerprint(toy):
    """An experiment written before answer_sets existed keeps its run folder."""
    config = load_experiment(toy / "experiment.yaml")
    before = config.model_dump(exclude={"data", "workspace", "run"})
    for key in ("answer_sets", "answers_by"):
        assert before["task"].pop(key) is None
    assert config_hash(config) == sha16(canonical([before]))
