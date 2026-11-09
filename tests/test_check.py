"""inference/check.py and the progress lines: what you see before and during a long run."""
import pandas as pd
import pytest

from vlmshift.cli import do_prepare, main
from vlmshift.common import Step, VLMShiftError
from vlmshift.extensions import get_adapter, register_adapter
from vlmshift.inference.adapters.fake import FakeGenerative
from vlmshift.inference import check as check_module
from vlmshift.inference import runner
from vlmshift.inference.base import ModelError
from vlmshift.inference.check import check_models, sample_rows
from vlmshift.inference.runner import Progress, duration, run_inference


def only_clean(raw):
    raw["task"].pop("reader_llm")
    raw["models"] = [{"name": "clean", "adapter": "fake_generative", "params": {"faults": False}}]


def test_the_check_shows_each_level_with_the_answer(toy, edit, capsys):
    config = edit(only_clean)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "Is there a circle" in output and "cercle" in output
    assert "Answer:   'Yes.'" in output and "Read as:  yes" in output and "NOT UNDERSTOOD" not in output


def test_the_check_says_how_long_the_run_would_take(toy, edit, capsys, monkeypatch):
    config = edit(only_clean)
    variants = do_prepare(config)
    capsys.readouterr()
    monkeypatch.setattr(check_module, "perf_counter", iter(range(0, 1000, 10)).__next__)  # each call takes 10 s
    check_models(config, variants)
    assert f"All {len(variants.df)} variants: about 20 s" in capsys.readouterr().out.replace("Loaded", "")  # batches of 8


def test_the_estimate_uses_the_last_call(toy, edit, capsys, monkeypatch):
    def slow_first():
        yield from (0, 0, 0, 100)  # the load, then a first question of 100 s
        for second in range(100, 200):  # then 1 s per question
            yield from (second, second + 1)

    config = edit(only_clean)
    variants = do_prepare(config)
    capsys.readouterr()
    clock = slow_first()
    monkeypatch.setattr(check_module, "perf_counter", lambda: next(clock))
    check_models(config, variants)
    assert "variants: about 2 s" in capsys.readouterr().out  # 1 s x 16 variants / 8 per batch


def test_the_check_keeps_no_answer(toy, edit):
    config = edit(only_clean)
    variants = do_prepare(config)
    check_models(config, variants)
    assert not (toy / "workspace" / "cache").exists()
    assert not (toy / "workspace" / "runs").exists()


def test_the_check_then_the_run_give_the_same_answers(toy, edit):
    config = edit(only_clean)
    variants = do_prepare(config)
    check_models(config, variants)
    assert len(list((run_inference(config, variants) / "results.parquet").parent.iterdir())) > 0
    cache = next((toy / "workspace" / "cache").iterdir())
    assert len((cache / "outputs.jsonl").read_text(encoding="utf-8").splitlines()) == len(variants.df)


def test_an_answer_not_understood_is_announced(toy, edit, capsys):
    def change(raw):
        only_clean(raw)
        raw["task"]["answers"] = {"yes": ["certainly"], "no": ["never"]}

    config = edit(change)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "NOT UNDERSTOOD" in output and "would be INVALID" in output


def test_one_answer_not_understood_is_enough_to_warn(toy, edit, capsys):
    def change(raw):
        only_clean(raw)
        raw["task"]["answers"] = {"yes": ["yes"], "no": ["no"]}  # the French answers are not listed

    config = edit(change)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "Read as:  yes" in output and "would be INVALID" in output


def test_a_model_that_cannot_load_is_reported_and_the_others_are_tried(toy, edit, capsys, monkeypatch):
    broken = get_adapter("fake_generative")

    def load(self):
        if self.name == "broken":
            raise ModelError("CUDA is not available")

    monkeypatch.setattr(broken, "load", load)

    def change(raw):
        only_clean(raw)
        raw["models"].insert(0, {"name": "broken", "adapter": "fake_generative"})

    config = edit(change)
    with pytest.raises(VLMShiftError, match=r"No answer from \['broken'\]"):
        check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "FAILED: CUDA is not available" in output and "Answer:   'Yes.'" in output


def test_the_memory_is_freed_after_a_failure(toy, edit, monkeypatch):
    adapter, freed = get_adapter("fake_generative"), []
    monkeypatch.setattr(adapter, "generate", lambda *_: (_ for _ in ()).throw(ModelError("broken")))
    monkeypatch.setattr(adapter, "unload", lambda self: freed.append(self.name))
    config = edit(only_clean)
    with pytest.raises(VLMShiftError):
        check_models(config, do_prepare(config))
    assert freed == ["clean"]


def test_a_long_answer_is_cut_on_screen(toy, edit, capsys, monkeypatch):
    monkeypatch.setattr(get_adapter("fake_generative"), "generate", lambda *_: ["word " * 500])
    config = edit(only_clean)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "word " * 39 in output and "word " * 41 not in output


def test_a_failure_on_a_question_is_reported(toy, edit, capsys):
    def change(raw):
        only_clean(raw)
        raw["models"][0]["params"] = {"faults": True}

    config = edit(change)
    variants = do_prepare(config)
    variants.df["image"] = "elsewhere.png"
    with pytest.raises(VLMShiftError, match="No answer"):
        check_models(config, variants)
    assert "FAILED: unknown image" in capsys.readouterr().out


def test_the_notes_of_a_model_are_shown(toy, edit, capsys, monkeypatch):
    adapter = get_adapter("fake_generative")

    def load(self):
        self.notes = ["temperature: refused by this model, so not applied"]

    monkeypatch.setattr(adapter, "load", load)
    config = edit(only_clean)
    check_models(config, do_prepare(config))
    assert "! model_note: temperature: refused" in capsys.readouterr().out


def test_contrastive_models_show_their_scores(toy, edit, capsys):
    def change(raw):
        only_clean(raw)
        raw["models"] = [{"name": "clip", "adapter": "fake_contrastive"}]

    config = edit(change)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "Phrases:" in output and "1.000" in output and "Read as:  yes" in output


def test_the_small_llm_is_used_when_the_list_finds_nothing(toy, edit, capsys):
    def change(raw):
        raw["models"] = [{"name": "clean", "adapter": "fake_generative", "params": {"faults": False}}]
        raw["task"]["answers"] = {"yes": ["certainly"], "no": ["never"]}  # the toy reader stays

    config = edit(change)
    check_models(config, do_prepare(config))
    output = capsys.readouterr().out
    assert "NOT UNDERSTOOD" not in output and "Read as:  yes" in output


def test_only_chooses_the_models(toy, edit, capsys):
    config = edit(lambda raw: None)
    check_models(config, do_prepare(config), ["fake-clip"])
    output = capsys.readouterr().out
    assert "fake-clip (" in output and "fake-gen (" not in output


def test_an_unknown_model_is_refused(toy, edit):
    config = edit(only_clean)
    with pytest.raises(VLMShiftError, match="unknown model"):
        check_models(config, do_prepare(config), ["absent"])


def test_no_variant_is_an_error(toy, edit):
    config = edit(only_clean)
    variants = do_prepare(config)
    with pytest.raises(VLMShiftError, match="No variant"):
        check_models(config, Step(variants.df.iloc[0:0], variants.dir, variants.hash))


def test_the_machine_is_described(toy, edit, capsys, monkeypatch):
    monkeypatch.setattr(check_module, "graphics_cards", lambda: ["NVIDIA A100 (40 GB)"])
    config = edit(only_clean)
    check_models(config, do_prepare(config))
    assert "Graphics cards: NVIDIA A100 (40 GB)" in capsys.readouterr().out


def test_no_card_is_said_plainly(toy, edit, capsys, monkeypatch):
    monkeypatch.setattr(check_module, "graphics_cards", lambda: [])
    config = edit(only_clean)
    check_models(config, do_prepare(config))
    assert "Graphics cards: none seen by torch" in capsys.readouterr().out


def test_the_command_exists_and_stops_on_a_failure(toy, capsys, monkeypatch):
    monkeypatch.setattr(get_adapter("fake_generative"), "generate", lambda *_: (_ for _ in ()).throw(ModelError("broken")))
    with pytest.raises(SystemExit) as stop:
        main(["check", str(toy / "experiment.yaml"), "--only", "fake-gen"])
    assert "No answer from" in str(stop.value)


def test_the_command_says_what_comes_next(toy, capsys):
    main(["check", str(toy / "experiment.yaml"), "--only", "fake-clean"])
    output = capsys.readouterr().out
    assert "Next: read the answers above" in output and "fake-gen (" not in output


def test_rows_differ_on_the_axes_and_are_few(toy, edit):
    config = edit(only_clean)
    variants = do_prepare(config).df
    rows = sample_rows(variants, ["language"])
    assert rows["language"].nunique() == variants["language"].nunique() == len(rows)
    assert len(sample_rows(variants, ["image", "language"])) == check_module.QUESTIONS


@pytest.mark.parametrize("seconds, text", [
    (0, "0 s"), (89, "89 s"), (90, "2 min"), (5399, "90 min"), (5400, "1.5 h"), (36000, "10.0 h"),
])
def test_durations_are_easy_to_read(seconds, text):
    assert duration(seconds) == text


def test_progress_is_silent_for_a_short_run(toy, capsys):
    main(["run", str(toy / "experiment.yaml")])
    assert " left" not in capsys.readouterr().out


def test_progress_shows_where_a_long_run_stands(capsys, monkeypatch):
    clock = iter([0, 30, 60, 70, 130]).__next__
    monkeypatch.setattr(runner, "perf_counter", clock)
    errors = {}
    progress = Progress("smolvlm", 100, errors)
    errors["a"] = "failed"
    progress.add(10)  # 30 s: too soon
    assert capsys.readouterr().out == ""
    progress.add(10)  # 60 s
    assert capsys.readouterr().out == "  smolvlm: 20 of 100, 1 errors, about 4 min left\n"
    progress.add(10)  # 70 s: too soon after the last line
    assert capsys.readouterr().out == ""
    progress.add(10)  # 130 s
    assert "40 of 100" in capsys.readouterr().out


def test_a_real_run_prints_its_progress(toy, edit, capsys, monkeypatch):
    monkeypatch.setattr(runner, "PROGRESS_SECONDS", 0)
    config = edit(only_clean)
    run_inference(config, do_prepare(config))
    assert "clean: 16 of 16, 0 errors" in capsys.readouterr().out


def test_progress_counts_the_vectors_too(toy, edit, capsys, monkeypatch):
    monkeypatch.setattr(runner, "PROGRESS_SECONDS", 0)

    def change(raw):
        only_clean(raw)
        raw["models"] = [{"name": "clip", "adapter": "fake_contrastive", "embeddings": True}]

    config = edit(change)
    run_inference(config, do_prepare(config))
    assert "clip: 24 of 24" in capsys.readouterr().out  # 16 variants and 8 images


def test_raw_answers_can_be_read_in_a_spreadsheet(toy, edit):
    config = edit(only_clean)
    folder = run_inference(config, do_prepare(config))
    written = pd.read_csv(folder / "results.csv")
    assert written["response"].tolist() == pd.read_parquet(folder / "results.parquet")["response"].tolist()
    assert written["response"].str.contains("oui|non").any()


class FailsInFrench(FakeGenerative):
    """Always fails on the French questions."""

    def generate(self, images, prompts, params):
        if any("cercle" in prompt for prompt in prompts):
            raise ModelError("out of memory")
        return super().generate(images, prompts, params)


class FailsOnce(FakeGenerative):
    """A short failure, such as a busy card."""

    failed = False

    def generate(self, images, prompts, params):
        if not FailsOnce.failed:
            FailsOnce.failed = True
            raise ModelError("busy")
        return super().generate(images, prompts, params)


def with_adapter(adapter):
    def change(raw):
        only_clean(raw)
        raw["models"][0]["adapter"] = adapter

    return change


def test_one_failed_question_does_not_stop_the_check(toy, edit, capsys):
    register_adapter("fails_in_french", FailsInFrench)
    config = edit(with_adapter("fails_in_french"))
    check_models(config, do_prepare(config))  # some answers came: the check passes
    output = capsys.readouterr().out
    assert "Some questions failed" in output
    assert "Is there a circle" in output  # the other questions were asked


def test_the_check_tries_a_question_again_as_the_run_does(toy, edit, capsys):
    register_adapter("fails_once", FailsOnce)
    FailsOnce.failed = False
    config = edit(with_adapter("fails_once"))
    check_models(config, do_prepare(config))
    assert "FAILED" not in capsys.readouterr().out


def test_a_torch_that_cannot_load_its_libraries_shows_no_card(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "torch":
            raise OSError("[WinError 126] The specified module could not be found: c10.dll")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    assert check_module.graphics_cards() == []
