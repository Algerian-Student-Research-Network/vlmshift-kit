"""Dataset and model registry: JSONL tables, empty columns, one-sided labels, model cards, answer reading."""
import json

import pandas as pd
import pytest

from vlmshift.cli import do_analyze, do_audit, do_import, do_run
from vlmshift.common import VLMShiftError
from vlmshift.config.load import config_hash, load_analysis, load_experiment, read_yaml, write_yaml
from vlmshift.extensions import register_adapter
from vlmshift.inference.adapters.fake import FakeGenerative, has_circle
from vlmshift.tasks.yes_no import first_word


def test_a_jsonl_table_gives_the_same_table_as_the_csv(toy, edit):
    from_csv = do_import(load_experiment(toy / "experiment.yaml"))
    rows = pd.read_csv(toy / "toy.csv", dtype=str).to_dict("records")
    (toy / "toy.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    from_jsonl = do_import(edit(lambda raw: raw.update(data="toy.jsonl")))
    assert from_jsonl.hash == from_csv.hash and from_jsonl.df["id"].tolist() == from_csv.df["id"].tolist()


def test_a_jsonl_file_that_is_not_a_table_is_refused(toy, edit):
    (toy / "toy.jsonl").write_text('["a", "b"]\n', encoding="utf-8")
    with pytest.raises(VLMShiftError, match="one row per line"):
        do_import(edit(lambda raw: raw.update(data="toy.jsonl")))


def test_the_audit_counts_the_empty_rows_of_each_column(toy):
    table = pd.read_csv(toy / "toy.csv", dtype=str)
    table["scanner"] = ["CR", None] * (len(table) // 2)
    table.to_csv(toy / "toy.csv", index=False)
    step = do_audit(load_experiment(toy / "experiment.yaml"))
    columns = pd.read_parquet(step.dir / "audit" / "columns.parquet").set_index("column")
    assert columns.loc["scanner", ["n_empty", "n_values"]].tolist() == [len(table) // 2, 1]
    assert columns.loc["scanner", "share_empty"] == 0.5 and columns.loc["site", "n_empty"] == 0
    issues = pd.read_parquet(step.dir / "audit" / "issues.parquet")
    assert "Column scanner: 8 empty rows (50%)" in issues.loc[issues["code"] == "column_empty", "message"].tolist()
    assert {"id", "image_hash"}.isdisjoint(columns.index)


def test_the_audit_warns_when_a_level_has_almost_one_class(tmp_path, toy):
    """40 images at site A, 2 of them positive: sensitivity there would rest on 2 cases."""
    rows = [{"image": f"images/img{1 + n % 8}.png", "question": f"q{n}", "label": "positive" if n < 2 else "negative",
             "site": "A", "subject": f"s{n}"} for n in range(40)]
    pd.DataFrame(rows).to_csv(toy / "many.csv", index=False)
    write_yaml({"schema_version": 1, "data": "many.csv", "roles": {"unit": "subject"}, "axes": {"site": {"type": "observed"}},
                "task": {"type": "yes_no"}, "models": [{"name": "m", "adapter": "fake_generative"}]}, toy / "many.yaml")
    step = do_audit(load_experiment(toy / "many.yaml"))
    issues = pd.read_parquet(step.dir / "audit" / "issues.parquet")
    message = issues.loc[issues["code"] == "label_imbalance", "message"].iloc[0]
    assert message.startswith("site = A: 2 positive and 38 negative labels")


def test_a_model_card_goes_to_the_log_and_the_report_without_moving_the_run(toy, edit):
    before = load_experiment(toy / "experiment.yaml")

    def add_card(raw):
        raw["models"][0] |= {"family": "toy generative", "license": "MIT", "gpu": "none", "validated": "laptop, 2026-11-16"}

    config = edit(add_card)
    run = do_run(config)
    log = json.loads((run / "run_log.json").read_text(encoding="utf-8"))["models"]
    assert log["fake-gen"]["license"] == "MIT" and log["fake-gen"]["supports_generation"] is True
    assert log["fake-clip"]["supports_scores"] is True and log["fake-clip"]["supports_embed"] is True
    assert log["fake-clip"]["license"] is None
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(run)))
    written = (report / "report.md").read_text(encoding="utf-8")
    assert "## Models" in written and "toy generative" in written and "laptop, 2026-11-16" in written
    assert "**Not stated as validated** on the machine of this run: fake-clean, fake-clip." in written
    assert "**Not pinned**: fake-gen, fake-clean, fake-clip." in written
    # A card is a description: the answers come from the same cache, in a run of its own.
    assert config_hash(config) != config_hash(before)
    assert log["fake-gen"]["cache"] == json.loads(
        (do_run(before) / "run_log.json").read_text(encoding="utf-8"))["models"]["fake-gen"]["cache"]


class Chatty(FakeGenerative):
    """Answers in a sentence in French, by one word in English."""

    def _answer(self, number: int, prompt: str) -> str:
        yes = has_circle(number)
        if "cercle" in prompt:
            return "Oui, il y a un cercle." if yes else "Non."
        return "yes" if yes else "no"


register_adapter("fake_chatty", Chatty)


def test_parser_analysis_tells_a_format_problem_from_a_wrong_answer(toy, edit):
    def one_model(raw):
        raw["models"] = [{"name": "chatty", "adapter": "fake_chatty"}]
        raw["task"].pop("reader_llm")

    run = do_run(edit(one_model))
    raw = read_yaml(toy / "analysis.yaml")
    raw.pop("latent", None)
    raw["analyses"] = [{"type": "parser", "by": ["language"]}]
    write_yaml(raw, toy / "analysis.yaml")
    report = do_analyze(load_analysis(toy / "analysis.yaml", str(run)))
    found = pd.read_parquet(report.parent / "analyses.parquet").set_index(["level_a", "statistic"])["value"]
    # French: the 4 yes are sentences, INVALID read whole, a yes by their first word.
    assert found["language=FR", "strict_invalid_rate"] == 0.5 and found["language=FR", "lenient_invalid_rate"] == 0
    assert found["language=FR", "rescued_rate"] == 0.5 and found["language=FR", "reading_disagreement"] == 0
    assert found["language=EN", "strict_invalid_rate"] == 0 and found["language=EN", "rescued_rate"] == 0


def test_first_word():
    assert first_word("Yes, a nodule is visible.") == "Yes," and first_word("  ") == "" and first_word("non") == "non"
