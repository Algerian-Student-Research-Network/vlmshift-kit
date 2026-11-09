"""variants/: step 3, exact texts sent to the models."""
import pandas as pd
import pytest

from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_experiment
from vlmshift.data.table import build_table
from vlmshift.tasks import get_task
from vlmshift.variants.plan import prepare_variants


def prepare(config):
    return prepare_variants(config, build_table(config))


def use_template(raw, template="Is there {concept}? Answer yes or no."):
    raw["task"]["template"] = template


def add_concept(toy, value="a circle"):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).assign(concept=value)
    data.to_csv(toy / "toy.csv", index=False)


def test_template_is_filled():
    task = get_task("yes_no")
    template = "Is there {concept} in this image? Answer yes or no."
    assert task.build_prompt({"concept": "a circle"}, template) == (
        "Is there a circle in this image? Answer yes or no."
    )
    assert task.build_prompt({"question": "Q?"}, None) == "Q?"
    assert task.build_contrastive({"c": "x"}, "with {c}", "no {c}") == ("with x", "no x")


def test_same_configuration_gives_same_variants(toy):
    config = load_experiment(toy / "experiment.yaml")
    first, second = prepare(config), prepare(config)
    assert first.hash == second.hash
    pd.testing.assert_frame_equal(first.df, second.df)


def test_prompts_are_stored(toy):
    variants = prepare(load_experiment(toy / "experiment.yaml")).df
    assert variants["prompt"].tolist() == variants["question"].tolist()
    assert variants["positive_prompt"].tolist() == variants["positive_text"].tolist()


def test_contrastive_templates_are_used_without_columns(toy, edit):
    def change(raw):
        del raw["roles"]["positive_text"], raw["roles"]["negative_text"]
        raw["task"]["contrastive"] = {"positive_template": "a {shape}", "negative_template": "no {shape}"}

    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.drop(columns=["positive_text", "negative_text"]).assign(shape="circle").to_csv(toy / "toy.csv", index=False)
    variants = prepare(edit(change)).df
    assert set(variants["positive_prompt"]) == {"a circle"}


def test_columns_come_before_contrastive_templates(toy, edit):
    templates = {"positive_template": "with {question}", "negative_template": "without {question}"}
    variants = prepare(edit(lambda raw: raw["task"].update(contrastive=templates))).df
    assert variants["positive_prompt"].tolist() == variants["positive_text"].tolist()


def test_template_change_gives_new_variants_but_answers_do_not(toy, edit):
    add_concept(toy)
    first = prepare(edit(use_template)).hash
    assert prepare(edit(lambda raw: raw["task"]["answers"]["yes"].append("sure"))).hash == first
    assert prepare(edit(lambda raw: use_template(raw, "Any {concept}?"))).hash != first


def test_template_hole_without_column_is_an_error(edit):
    with pytest.raises(VLMShiftError, match="concept"):
        prepare(edit(use_template))


def test_empty_value_excludes_the_line(toy, edit):
    add_concept(toy)
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[0, "concept"] = ""
    data.to_csv(toy / "toy.csv", index=False)
    variants = prepare(edit(use_template))
    assert len(variants.df) == 15
    issues = pd.read_parquet(variants.dir / "issues.parquet")
    assert issues["code"].tolist() == ["empty_template_value"]


def test_axis_without_column_is_an_error(toy, edit):
    config = load_experiment(toy / "experiment.yaml")
    table = build_table(config)
    with pytest.raises(VLMShiftError, match="absent"):
        prepare_variants(edit(lambda raw: raw["axes"].update(absent={"type": "observed"})), table)


def test_a_template_word_translated_in_each_language_keeps_the_pairs(toy, edit):
    """The documented pattern: {concept} changes with the language, like the question."""
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(concept=data["language"].map({"EN": "a circle", "FR": "un cercle"})).to_csv(toy / "toy.csv", index=False)
    variants = prepare(edit(use_template))
    assert variants.df["sample_id"].nunique() == 8
    assert variants.df.groupby("sample_id")["language"].nunique().eq(2).all()


def test_the_contrastive_template_columns_keep_the_pairs(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(concept=data["language"].map({"EN": "circle", "FR": "cercle"})).to_csv(toy / "toy.csv", index=False)

    def change(raw):
        raw["task"]["contrastive"] = {"positive_template": "a {concept}", "negative_template": "no {concept}"}

    assert build_table(edit(change)).df["sample_id"].nunique() == 8


def test_no_row_left_for_the_models_stops_at_step_3(toy, edit):
    add_concept(toy, "")
    with pytest.raises(VLMShiftError, match=r"No variant left.*\(concept\)"):
        prepare(edit(use_template))


def test_a_template_names_a_role_column_by_its_role(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).rename(columns={"question": "texte"})
    data.to_csv(toy / "toy.csv", index=False)

    def change(raw):
        raw["roles"]["question"] = "texte"
        use_template(raw, "{texte}")

    with pytest.raises(VLMShiftError, match=r"write \{question\} instead of \{texte\}"):
        prepare(edit(change))


def test_no_variant_left_names_the_empty_column_of_the_templates_used(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).assign(positive_text="")
    data.to_csv(toy / "toy.csv", index=False)

    def change(raw):
        raw["models"] = [{"name": "clip", "adapter": "fake_contrastive"}]

    with pytest.raises(VLMShiftError, match=r"No variant left.*\(positive_text\)"):
        prepare(edit(change))
