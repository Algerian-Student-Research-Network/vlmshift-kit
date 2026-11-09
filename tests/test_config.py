"""config/: reading, validation rules, fingerprints."""
import pytest

from vlmshift.common import VLMShiftError
from vlmshift.config.load import config_hash, load_analysis, load_experiment, read_yaml, write_yaml
from vlmshift.config.schema import export_json_schemas
from vlmshift.variants.prompts import templates_for


def expect_error(edit, change, name="experiment.yaml"):
    with pytest.raises(VLMShiftError):
        edit(change, name)


def test_yes_and_no_stay_text(toy):
    assert load_experiment(toy / "experiment.yaml").task.answers.yes == ["yes", "oui"]
    (toy / "t.yaml").write_text("a: yes\nb: no\n", encoding="utf-8")
    assert read_yaml(toy / "t.yaml") == {"a": "yes", "b": "no"}


@pytest.mark.parametrize("text, message", [
    ("measures:\n  - {name: a, compare: b}\nstats: {seed: 1}\n  - {name: c}\n", "not valid YAML"),
    ("schema_version: 1\nschema_version: 1\n", "not valid YAML"),  # the same key twice
    ("just a sentence\n", "keys and values"),
    ("- a\n- b\n", "keys and values"),
])
def test_broken_yaml_gives_a_clear_message(toy, text, message):
    (toy / "experiment.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(VLMShiftError, match=message):
        load_experiment(toy / "experiment.yaml")


def test_task_that_is_not_a_block_gives_a_clear_message(edit):
    expect_error(edit, lambda raw: raw.update(task="yes_no"))


def test_unknown_key_is_refused(edit):
    expect_error(edit, lambda raw: raw.update(modelz=[]))


def test_wrong_value_is_refused(edit):
    expect_error(edit, lambda raw: raw.update(inference={"retries": -1}))
    expect_error(edit, lambda raw: raw.update(schema_version=2))


def test_model_names_are_unique(edit):
    expect_error(edit, lambda raw: raw["models"].append({"name": "fake-gen", "adapter": "fake_generative"}))


def test_answer_cannot_be_in_both_lists(edit):
    expect_error(edit, lambda raw: raw["task"]["answers"]["no"].append("YES"))


def test_contrastive_model_needs_phrases(edit):
    def change(raw):
        del raw["roles"]["positive_text"]

    expect_error(edit, change)


def test_unknown_adapter_only_stops_at_step_3(toy, edit):
    """Re-reading a run to analyze it must not ask for its adapters."""
    config = edit(lambda raw: raw["models"].append({"name": "x", "adapter": "absent"}))
    with pytest.raises(VLMShiftError, match="Unknown adapter"):
        templates_for(config)


def test_answers_can_be_in_a_file(toy, edit):
    write_yaml({"yes": ["oui"], "no": ["non"]}, toy / "answers.yaml")
    config = edit(lambda raw: raw["task"].update(answers="answers.yaml"))
    assert config.task.answers.yes == ["oui"]


def test_reference_must_be_in_levels(edit):
    expect_error(edit, lambda raw: raw["axes"]["language"].update(levels=["FR"]), "analysis.yaml")


def test_compare_and_split_by_share_no_axis(edit):
    change = lambda raw: raw["measures"].append({"name": "x", "compare": "site", "split_by": ["site"]})  # noqa: E731
    expect_error(edit, change, "analysis.yaml")


def test_measure_names_are_unique(edit):
    change = lambda raw: raw["measures"].append({"name": "site", "compare": "site"})  # noqa: E731
    expect_error(edit, change, "analysis.yaml")


def test_unknown_key_is_refused_in_analysis(edit):
    expect_error(edit, lambda raw: raw["stats"].update(boostrap=500), "analysis.yaml")


def test_comments_order_and_defaults_do_not_change_the_hash(toy):
    plain = "schema_version: 1\ndata: a.csv\naxes: {x: {type: observed}}\ntask: {type: yes_no}\n" \
            "models: [{name: m, adapter: fake_generative}]\n"
    verbose = "# comment\ninference: {temperature: 0, retries: 2}\nmodels: [{adapter: fake_generative, name: m}]\n" \
              "task: {type: yes_no}\naxes: {x: {type: observed}}\ndata: other.csv\nworkspace: elsewhere\nschema_version: 1\n"
    (toy / "a.yaml").write_text(plain, encoding="utf-8")
    (toy / "b.yaml").write_text(verbose, encoding="utf-8")
    assert config_hash(load_experiment(toy / "a.yaml")) == config_hash(load_experiment(toy / "b.yaml"))


def test_a_value_changes_the_hash(edit, toy):
    before = config_hash(load_experiment(toy / "experiment.yaml"))
    after = config_hash(edit(lambda raw: raw["task"].update(template="{question} ?")))
    assert before != after


def test_json_schemas_are_exported(tmp_path):
    export_json_schemas(tmp_path)
    assert (tmp_path / "experiment.schema.json").exists() and (tmp_path / "analysis.schema.json").exists()


def test_analysis_run_can_be_overridden(toy):
    assert load_analysis(toy / "analysis.yaml", run="other").run == "other"


@pytest.mark.parametrize("template", ["Is there a {shape?", "Is there {}?", "Is there {0}?"])
def test_a_brace_without_a_column_name_is_refused(edit, template):
    expect_error(edit, lambda raw: raw["task"].update(template=template))
    expect_error(edit, lambda raw: raw["task"].update(contrastive={"positive_template": template, "negative_template": "no"}))


def test_a_brace_can_be_written_twice_to_stay_text(toy, edit):
    assert edit(lambda raw: raw["task"].update(template="{{note}} {question}")).task.template == "{{note}} {question}"


def test_answers_are_compared_as_the_reader_reads_them(edit):
    expect_error(edit, lambda raw: raw["task"]["answers"]["no"].append("Yes."))


def test_a_level_written_as_a_number_is_read_as_text(toy, edit):
    config = edit(lambda raw: raw["axes"].update(quality={"reference": 1, "levels": [1, 2]}), "analysis.yaml")
    assert config.axes["quality"].reference == "1" and config.axes["quality"].levels == ["1", "2"]


def test_run_typed_in_the_terminal_is_read_from_the_current_folder(toy, monkeypatch):
    (toy / "workspace" / "runs" / "r1").mkdir(parents=True)
    monkeypatch.chdir(toy.parent)
    config = load_analysis(toy / "analysis.yaml", run="toy/workspace/runs/r1")
    assert config.run == str(toy / "workspace" / "runs" / "r1")


def test_a_problem_is_one_clear_line(edit):
    with pytest.raises(VLMShiftError) as caught:
        edit(lambda raw: raw["task"].update(answers={"yes": ["yes"], "no": ["Yes."]}))
    assert "- task.answers: answers in both yes and no: ['yes']" in str(caught.value)
    assert "valid string" not in str(caught.value)  # the form not used (a file name) is not mentioned
    with pytest.raises(VLMShiftError, match=r"- inference.retries: .* \(you wrote: -1\)"):
        edit(lambda raw: raw.update(inference={"retries": -1}))


def test_a_misspelled_key_gets_a_suggestion(edit):
    with pytest.raises(VLMShiftError, match=r"- inferance: unknown key: did you mean inference\?"):
        edit(lambda raw: raw.update(inferance={"retries": 1}))


def test_a_word_where_a_block_is_expected(edit):
    with pytest.raises(VLMShiftError, match=r"- task: should be a block of keys and values.*\(you wrote: 'yes_no'\)"):
        edit(lambda raw: raw.update(task="yes_no"))
