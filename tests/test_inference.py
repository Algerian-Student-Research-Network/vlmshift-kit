"""inference/: step 4, with the fake models."""
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import FIXTURES
from vlmshift.cli import do_prepare, main
from vlmshift.common import JsonlWriter, VLMShiftError, append_jsonl, read_jsonl
from vlmshift.config.load import load_experiment
from vlmshift.extensions import register_adapter
from vlmshift.inference.adapters.fake import FakeContrastive, FakeGenerative
from vlmshift.inference.base import ModelError
from vlmshift.inference.runner import run_inference


class Counting(FakeGenerative):
    """Counts the variants received."""

    calls = 0

    def generate(self, images, prompts, params):
        Counting.calls += len(images)
        return super().generate(images, prompts, params)


class Interrupted(FakeGenerative):
    """Stops at the second batch, like a power cut."""

    batch_size = 4
    batches = unloads = 0

    def generate(self, images, prompts, params):
        Interrupted.batches += 1
        if Interrupted.batches == 2:
            raise KeyboardInterrupt
        return super().generate(images, prompts, params)

    def unload(self):
        Interrupted.unloads += 1
        super().unload()


class Flaky(FakeGenerative):
    """Fails once, then succeeds."""

    failed = False

    def generate(self, images, prompts, params):
        if not Flaky.failed:
            Flaky.failed = True
            raise ModelError("temporary failure")
        return super().generate(images, prompts, params)


def only(raw, adapter, **entry):
    """Keep only one model in the configuration."""
    raw["models"] = [{"name": "m", "adapter": adapter, **entry}]
    raw["task"].pop("reader_llm", None)


def run(config):
    return run_inference(config, do_prepare(config))


def run_that_fails(toy, config) -> pd.DataFrame:
    """A run where no variant gets an answer: the command stops, but its results are written."""
    with pytest.raises(VLMShiftError, match="No answer from"):
        run(config)
    return pd.read_parquet(next((toy / "workspace" / "runs").glob("*/results.parquet")))


def test_results_are_those_expected(full_run):
    results = pd.read_parquet(full_run.run / "results.parquet")
    table = pd.read_parquet(next(full_run.folder.glob("workspace/tables/*/table.parquet")))
    merged = results.merge(table[["id", "image", "language"]], on="id")
    merged["image"] = [Path(path).stem for path in merged["image"]]
    expected = pd.read_csv(FIXTURES / "results.csv")
    columns = ["model", "image", "language"]
    merged = merged.set_index(columns).sort_index()
    expected = expected.set_index(columns).sort_index()
    assert merged.index.tolist() == expected.index.tolist()
    assert merged["status"].tolist() == expected["status"].tolist()
    assert merged["decision"].fillna("").tolist() == expected["decision"].fillna("").tolist()


def test_outputs_are_kept(full_run):
    results = pd.read_parquet(full_run.run / "results.parquet")
    generative = results[results["model"] == "fake-gen"]
    assert generative["decision_source"].dropna().eq("list").all()
    assert generative.loc[generative["status"] == "INVALID", "response"].tolist() == ["maybe"]
    assert generative.loc[generative["status"] == "INVALID", "reader_response"].tolist() == ["unclear"]
    assert generative.loc[generative["status"] == "error", "message"].tolist() == ["simulated failure"]
    contrastive = results[results["model"] == "fake-clip"]
    assert contrastive["margin"].equals(contrastive["score_pos"] - contrastive["score_neg"])
    assert contrastive["decision_source"].isna().all()


def test_each_image_gets_one_vector(full_run):
    vectors = pd.read_parquet(full_run.run / "embeddings.parquet")
    assert len(vectors) == 8 and vectors["image_hash"].is_unique
    assert set(vectors["model"]) == {"fake-clip"}


def test_run_log(full_run):
    log = json.loads((full_run.run / "run_log.json").read_text(encoding="utf-8"))
    assert log["run"] == full_run.run.name
    assert set(log["models"]) == {"fake-gen", "fake-clean", "fake-clip"}
    assert log["reader_llm"]["model"] == "fake_reader"
    assert "pandas" in log["libraries"] and log["started_at"] <= log["finished_at"]


def test_issues_of_the_run(full_run):
    issues = pd.read_parquet(full_run.run / "issues.parquet")
    assert issues["code"].tolist() == ["model_error"]


def test_reader_validation_is_written(full_run):
    assert len(pd.read_parquet(full_run.run / "reader_validation.parquet")) == 3


def test_same_experiment_finds_the_same_run_without_inference(toy, edit):
    register_adapter("counting", Counting)
    config = edit(lambda raw: only(raw, "counting", params={"faults": False}))
    first = run(config)
    assert Counting.calls == 16
    assert run(config) == first
    assert Counting.calls == 16


def test_new_model_keeps_the_cache_of_the_others(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    config = edit(lambda raw: only(raw, "counting", params={"faults": False}))
    first = run(config)
    second = run(edit(lambda raw: raw["models"].append({"name": "other", "adapter": "fake_generative"})))
    assert second != first
    assert Counting.calls == 16


def test_cache_lines_are_in_the_file_before_it_is_closed(tmp_path):
    cache = JsonlWriter(tmp_path / "cache" / "outputs.jsonl")
    cache.write([{"id": "a"}])
    assert read_jsonl(tmp_path / "cache" / "outputs.jsonl") == [{"id": "a"}]  # a stopped run keeps it
    cache.write([{"id": "b"}])
    cache.close()
    assert len(read_jsonl(tmp_path / "cache" / "outputs.jsonl")) == 2
    JsonlWriter(tmp_path / "unused.jsonl").close()
    assert not (tmp_path / "unused.jsonl").exists()


def test_more_retries_does_not_rerun_the_model(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    run(edit(lambda raw: only(raw, "counting", params={"faults": False})))
    run(edit(lambda raw: raw.update(inference={"retries": 5})))
    assert Counting.calls == 16


def test_changing_the_reader_does_not_rerun_the_model(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    run(edit(lambda raw: only(raw, "counting", params={"faults": False})))
    folder = run(edit(lambda raw: raw["task"].update(answers={"yes": ["oui"], "no": ["non"]})))
    assert Counting.calls == 16
    results = pd.read_parquet(folder / "results.parquet")
    assert (results["status"] == "INVALID").sum() == 8  # "Yes." and "No." are no longer read


def test_error_lines_are_retried_at_the_next_launch(toy, edit):
    config = edit(lambda raw: only(raw, "fake_generative"))
    variants = do_prepare(config)
    run_inference(config, variants)
    cache = next((toy / "workspace").glob("cache/*/outputs.jsonl"))
    assert len(read_jsonl(cache)) == 15  # the failure does not go into the cache


def test_interruption_then_resume_leaves_no_duplicate_and_no_hole(toy, edit):
    register_adapter("interrupted", Interrupted)
    config = edit(lambda raw: only(raw, "interrupted", params={"faults": False}))
    variants = do_prepare(config)
    with pytest.raises(KeyboardInterrupt):
        run_inference(config, variants)
    assert Interrupted.unloads == 1  # the memory of the card is freed even when the run stops
    folder = variants.dir.parents[1].parent.parent / "runs"
    partial = pd.read_parquet(next(folder.glob("*/results.parquet")))
    assert len(partial) == 4  # the first batch is saved

    run_inference(config, variants)
    results = pd.read_parquet(next(folder.glob("*/results.parquet")))
    assert len(results) == 16 and results["id"].is_unique
    assert Interrupted.batches == 2 + 3  # 12 variants left, so 3 batches of 4


def test_model_error_is_retried(toy, edit):
    register_adapter("flaky", Flaky)
    folder = run(edit(lambda raw: only(raw, "flaky", params={"faults": False})))
    assert pd.read_parquet(folder / "results.parquet")["status"].eq("valid").all()


def test_retries_exhausted_gives_error_only_on_the_faulty_line(toy, edit):
    folder = run(edit(lambda raw: (only(raw, "fake_generative"), raw.update(inference={"retries": 0}))))
    status = pd.read_parquet(folder / "results.parquet")["status"].value_counts()
    assert status["error"] == 1 and status["INVALID"] == 1 and status["valid"] == 14


def test_load_failure_gives_error_status(toy, edit):
    class Broken(FakeGenerative):
        def load(self):
            raise ModelError("weights not found")

    register_adapter("broken", Broken)
    assert run_that_fails(toy, edit(lambda raw: only(raw, "broken")))["status"].eq("error").all()


def test_embeddings_without_embed_method_is_an_error(toy, edit):
    with pytest.raises(VLMShiftError, match="embed"):
        run(edit(lambda raw: only(raw, "fake_generative", embeddings=True)))


def test_unloadable_reader_stops_before_inference(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0

    def change(raw):
        only(raw, "counting")
        raw["task"]["reader_llm"] = {"model": "absent/absent"}

    with pytest.raises(VLMShiftError, match="LLM"):
        run(edit(change))
    assert Counting.calls == 0


def test_reader_validation_file_is_checked_before_inference(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0

    def change(raw):
        only(raw, "counting")
        raw["task"]["reader_llm"] = {"model": "fake_reader", "validation": "absent.csv"}

    with pytest.raises(VLMShiftError, match="validation"):
        run(edit(change))
    assert Counting.calls == 0


def test_llm_alone_reads_everything(toy, edit):
    def change(raw):
        only(raw, "fake_generative", params={"faults": False})
        del raw["task"]["answers"]
        raw["task"]["reader_llm"] = {"model": "fake_reader"}

    results = pd.read_parquet(run(edit(change)) / "results.parquet")
    assert results["decision_source"].eq("llm").all()


def test_no_reader_uses_the_default_list(toy, edit):
    def change(raw):
        only(raw, "fake_generative", params={"faults": False})
        del raw["task"]["answers"]

    results = pd.read_parquet(run(edit(change)) / "results.parquet")
    assert (results["status"] == "valid").sum() == 8  # "oui" and "non" are not in the default list
    assert (results["status"] == "INVALID").sum() == 8


def test_list_then_llm(toy, edit):
    def change(raw):
        only(raw, "fake_generative", params={"faults": False})
        raw["task"]["answers"] = {"yes": ["yes"], "no": ["no"]}
        raw["task"]["reader_llm"] = {"model": "fake_reader"}

    results = pd.read_parquet(run(edit(change)) / "results.parquet")
    assert results["decision_source"].value_counts().to_dict() == {"list": 8, "llm": 8}


def two_models(raw):
    raw["models"] = [
        {"name": "a", "adapter": "counting", "params": {"faults": False}},
        {"name": "b", "adapter": "counting", "params": {"faults": False, "other": True}},
    ]
    raw["task"].pop("reader_llm", None)


def test_only_runs_the_chosen_model_and_reports_the_other(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    config = edit(two_models)
    folder = run_inference(config, do_prepare(config), ["a"])
    assert Counting.calls == 16
    assert set(pd.read_parquet(folder / "results.parquet")["model"]) == {"a"}
    issues = pd.read_parquet(folder / "issues.parquet")
    assert issues["code"].tolist() == ["model_pending"]
    assert issues["message"].tolist() == ["b: 16 variants not yet processed"]


def test_two_only_launches_give_one_complete_run(toy, edit):
    """Each model runs on its own, as in two environments."""
    register_adapter("counting", Counting)
    config = edit(two_models)
    variants = do_prepare(config)
    first = run_inference(config, variants, ["a"])
    Counting.calls = 0
    second = run_inference(config, variants, ["b"])
    assert second == first
    assert Counting.calls == 16  # a is not run again
    results = pd.read_parquet(second / "results.parquet")
    assert results.groupby("model").size().to_dict() == {"a": 16, "b": 16}
    assert pd.read_parquet(second / "issues.parquet").empty


def test_each_model_keeps_the_trace_of_its_own_environment(toy, edit, monkeypatch):
    register_adapter("counting", Counting)
    config = edit(two_models)
    variants = do_prepare(config)
    monkeypatch.setattr("vlmshift.inference.runner.library_versions", lambda: {"transformers": "4.33"})
    run_inference(config, variants, ["a"])
    monkeypatch.setattr("vlmshift.inference.runner.library_versions", lambda: {"transformers": "4.57"})
    folder = run_inference(config, variants, ["b"])
    models = json.loads((folder / "run_log.json").read_text(encoding="utf-8"))["models"]
    assert [run["libraries"] for run in models["a"]["sessions"]] == [{"transformers": "4.33"}]
    assert [run["libraries"] for run in models["b"]["sessions"]] == [{"transformers": "4.57"}]


def test_only_with_an_unknown_model_is_an_error(toy, edit):
    config = edit(lambda raw: only(raw, "fake_generative"))
    with pytest.raises(VLMShiftError, match="unknown"):
        run_inference(config, do_prepare(config), ["absent"])


def test_a_partial_run_can_be_analysed_and_says_what_is_missing(toy):
    main(["run", str(toy / "experiment.yaml"), "--only", "fake-clean"])
    run_dir = next((toy / "workspace" / "runs").iterdir())
    main(["analyze", str(toy / "analysis.yaml"), "--run", str(run_dir)])
    report = next(run_dir.glob("analyses/*/report/report.md")).read_text(encoding="utf-8")
    assert "fake-gen: 16 variants not yet processed" in report
    assert "fake-clip: 16 variants not yet processed" in report
    assert "Probe not computed" in report
    assert "Partial run" in report


def test_a_complete_run_is_not_called_partial(full_run):
    assert "Partial run" not in (full_run.report / "report.md").read_text(encoding="utf-8")


def test_notes_of_a_model_reach_the_issues(toy, edit):
    class Noted(FakeGenerative):
        def load(self):
            self.notes = ["temperature: refused by this model, so not applied"]

    register_adapter("noted", Noted)
    folder = run(edit(lambda raw: only(raw, "noted", params={"faults": False})))
    issues = pd.read_parquet(folder / "issues.parquet")
    assert issues["code"].tolist() == ["model_note"]
    assert issues["message"].iloc[0].startswith("m: temperature")


def test_an_answer_that_is_not_text_is_a_failure(toy, edit):
    class Silent(FakeGenerative):
        def generate(self, images, prompts, params):
            return [None] * len(images)

    register_adapter("silent", Silent)
    assert run_that_fails(toy, edit(lambda raw: only(raw, "silent")))["status"].eq("error").all()


def test_a_model_of_another_kind_keeps_the_answers_of_the_others(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    run(edit(lambda raw: only(raw, "counting", params={"faults": False})))
    run(edit(lambda raw: raw["models"].append({"name": "clip", "adapter": "fake_contrastive"})))
    assert Counting.calls == 16


def test_two_experiments_on_one_table_keep_their_own_variants(toy, edit):
    generative = do_prepare(edit(lambda raw: only(raw, "fake_generative")))
    contrastive = do_prepare(edit(lambda raw: only(raw, "fake_contrastive")))
    assert generative.dir != contrastive.dir
    assert pd.read_parquet(generative.dir / "variants.parquet")["positive_prompt"].isna().all()  # not overwritten


def test_failures_of_a_model_stay_when_another_one_runs_alone(toy, edit):
    config = load_experiment(toy / "experiment.yaml")
    run_inference(config, do_prepare(config), only=["fake-gen"])
    folder = run_inference(config, do_prepare(config), only=["fake-clean"])
    results = pd.read_parquet(folder / "results.parquet")
    assert (results.loc[results["model"] == "fake-gen", "status"] == "error").sum() == 1
    assert "model_error" in set(pd.read_parquet(folder / "issues.parquet")["code"])


def test_a_line_cut_by_a_stop_does_not_take_the_next_one(tmp_path):
    path = tmp_path / "outputs.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b", "resp', encoding="utf-8")
    cache = JsonlWriter(path)
    cache.write([{"id": "c"}])
    cache.close()
    assert [row["id"] for row in read_jsonl(path)] == ["a", "c"]


class NeverAnswers(FakeGenerative):
    """Always fails, and counts its calls."""

    calls = 0

    def generate(self, images, prompts, params):
        NeverAnswers.calls += 1
        raise ModelError("no card")


def test_a_model_that_never_answers_stops_the_command(toy, edit):
    register_adapter("never", NeverAnswers)
    config = edit(lambda raw: only(raw, "never", params={"retries": 0}))
    with pytest.raises(VLMShiftError, match=r"No answer from \['m'\]: every variant failed. First error: no card"):
        run(config)
    assert NeverAnswers.calls == 2 * (1 + 8)  # params.retries replaces inference.retries: one try per batch, then per row
    assert len(list((toy / "workspace" / "runs").glob("*/results.parquet"))) == 1  # the run folder is written


def test_a_batch_size_that_cannot_work_stops_before_any_model(toy, edit):
    with pytest.raises(VLMShiftError, match="batch_size must be a whole number of at least 1, not 0"):
        run(edit(lambda raw: only(raw, "fake_generative", params={"batch_size": 0})))


def test_more_retries_in_params_does_not_rerun_the_model(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    run(edit(lambda raw: only(raw, "counting", params={"faults": False})))
    run(edit(lambda raw: raw["models"][0]["params"].update(retries=5)))
    assert Counting.calls == 16


class FailsOnce(FakeGenerative):
    """Fails at its first launch only, like a server that was down."""

    down = True

    def generate(self, images, prompts, params):
        if FailsOnce.down:
            raise ModelError("server down")
        return super().generate(images, prompts, params)


def test_old_failures_are_forgotten_when_the_model_runs_again(toy, edit):
    register_adapter("fails_once", FailsOnce)
    FailsOnce.down = True
    config = edit(lambda raw: (only(raw, "fails_once", params={"faults": False}), raw.update(inference={"retries": 0})))
    run_that_fails(toy, config)
    FailsOnce.down = False
    folder = run(config)
    assert not pd.read_parquet(folder / "results.parquet")["status"].eq("error").any()
    assert "model_error" not in set(pd.read_parquet(folder / "issues.parquet")["code"])
    assert next(toy.glob("workspace/cache/*/errors.jsonl")).read_text(encoding="utf-8") == ""


class ById(FakeGenerative):
    """Answers according to its id, as a catalogue loads the model its id names."""

    def generate(self, images, prompts, params):
        return ["yes" if self.id == "always-yes" else "no" for _ in images]


class StopInEmbed(FakeContrastive):
    """Stopped while computing the vectors, after every score."""

    batch_size = 4
    calls = 0

    def embed(self, images):
        StopInEmbed.calls += 1
        if StopInEmbed.calls == 2:
            raise KeyboardInterrupt
        return super().embed(images)


def test_two_models_without_id_keep_their_own_answers(toy, edit):
    register_adapter("by_id", ById)

    def change(raw):
        raw["task"].pop("reader_llm", None)
        raw["models"] = [{"name": "always-yes", "adapter": "by_id"}, {"name": "always-no", "adapter": "by_id"}]

    results = pd.read_parquet(run(edit(change)) / "results.parquet")
    decisions = results.groupby("model")["decision"].unique().map(sorted).to_dict()
    assert decisions == {"always-no": ["no"], "always-yes": ["yes"]}


def test_a_corrected_label_or_another_axis_asks_no_question_again(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    config = edit(lambda raw: only(raw, "counting", params={"faults": False}))
    run(config)
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[data["image"].str.endswith("img1.png"), "label"] = "unknown"
    data.to_csv(toy / "toy.csv", index=False)
    run(config)  # another table: the questions are the same
    run(edit(lambda raw: raw["axes"].update(site={"type": "controlled"})))
    assert Counting.calls == 16
    assert len(list((toy / "workspace" / "cache").iterdir())) == 1


def test_an_answer_with_a_unicode_line_separator_is_kept(tmp_path):
    path = tmp_path / "outputs.jsonl"
    append_jsonl(path, [{"q": "a", "response": "yes\u2028no"}, {"q": "b", "response": "no"}])
    assert [row["q"] for row in read_jsonl(path)] == ["a", "b"]


def test_a_line_cut_inside_a_character_loses_only_that_line(tmp_path):
    path = tmp_path / "outputs.jsonl"
    append_jsonl(path, [{"q": "a", "response": "yes"}])
    with open(path, "ab") as file:
        file.write('{"q": "x", "response": "हाँ"}'.encode("utf-8")[:-4])  # a stop inside the last character
    append_jsonl(path, [{"q": "b", "response": "no"}])
    assert [row["q"] for row in read_jsonl(path)] == ["a", "b"]


def test_missing_vectors_make_a_partial_run_until_they_are_computed(toy, edit):
    register_adapter("stop_in_embed", StopInEmbed)
    StopInEmbed.calls = 0
    config = edit(lambda raw: raw.update(models=[{"name": "clip", "adapter": "stop_in_embed", "embeddings": True}]))
    with pytest.raises(KeyboardInterrupt):
        run(config)
    folder = next((toy / "workspace" / "runs").iterdir())
    assert "vectors_pending" in set(pd.read_parquet(folder / "issues.parquet")["code"])
    folder = run(config)
    assert "vectors_pending" not in set(pd.read_parquet(folder / "issues.parquet")["code"])
    assert len(pd.read_parquet(folder / "embeddings.parquet")) == 8


def test_another_revision_asks_the_model_again(toy, edit):
    register_adapter("counting", Counting)
    Counting.calls = 0
    run(edit(lambda raw: only(raw, "counting", revision="first", params={"faults": False})))
    run(edit(lambda raw: only(raw, "counting", revision="second", params={"faults": False})))
    assert Counting.calls == 32  # other weights: nothing comes from the cache of the first


def test_a_load_that_fails_half_way_frees_what_it_holds(toy, edit):
    class HalfLoaded(FakeGenerative):
        unloads = 0

        def load(self):
            self.model = "weights on the card"  # the weights are there, then the processor fails
            raise ModelError("processor not found")

        def unload(self):
            HalfLoaded.unloads += 1
            super().unload()

    register_adapter("half", HalfLoaded)
    run_that_fails(toy, edit(lambda raw: only(raw, "half")))
    assert HalfLoaded.unloads == 1


def test_the_first_error_shown_is_that_of_the_first_variant(toy, edit):
    class NamesTheImage(FakeGenerative):
        def generate(self, images, prompts, params):
            raise ModelError(f"cannot read {Path(images[0]).name}")

    register_adapter("names", NamesTheImage)
    config = edit(lambda raw: only(raw, "names", params={"retries": 0}))
    first = Path(do_prepare(config).df["image"].iloc[0]).name
    with pytest.raises(VLMShiftError, match=f"First error: cannot read {first}"):
        run(config)
