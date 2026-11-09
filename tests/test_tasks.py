"""tasks/: answer readers, tested language by language on the labeled set."""
import pandas as pd
import pytest

from vlmshift.common import VLMShiftError
from vlmshift.config.schema import ReaderLLM
from vlmshift.tasks import get_task
from vlmshift.tasks.llm_reader import LLMReader, validate_reader
from vlmshift.tasks.yes_no import clean

ANSWERS = {"yes": ["yes", "oui", "हाँ"], "no": ["no", "non", "नहीं"]}


def fake_llm(tmp_path):
    return LLMReader(ReaderLLM(model="fake_reader"), tmp_path)


def test_clean_keeps_letters_and_marks():
    assert clean("  Yes. ") == "yes"
    assert clean("हाँ।") == "हाँ"  # the final Hindi stroke is punctuation, the vowel is not
    assert clean("...") == ""


def test_list_reader_on_annotated_answers(toy):
    task, table = get_task("yes_no"), pd.read_csv(toy / "annotated_answers.csv", keep_default_na=False)
    for row in table.itertuples():
        found = task.read(row.response, ANSWERS)
        # The list only recognizes short answers; the rest must stay without a decision.
        assert found in (row.expected, None)
        if clean(row.response) in ANSWERS["yes"] + ANSWERS["no"]:
            assert found == row.expected


def test_is_correct():
    task = get_task("yes_no")
    assert task.is_correct("yes", "positive") and task.is_correct("no", "negative")
    assert task.is_correct("yes", "negative") is False
    assert task.is_correct("yes", "unknown") is None


def test_llm_reader_decides_and_keeps_its_raw_answer(tmp_path):
    reader = fake_llm(tmp_path)
    assert reader.read("Yes, there is one.") == ("yes", "yes")
    assert reader.read("maybe") == (None, "unclear")


def test_llm_answers_are_cached(tmp_path):
    reader = fake_llm(tmp_path)
    reader.read("oui")
    reader.ask = lambda prompt: pytest.fail("the cache should have answered")
    assert reader.read("oui")[0] == "yes"
    other = fake_llm(tmp_path)  # the cache survives a new launch
    other.ask = reader.ask
    assert other.read("oui")[0] == "yes"


def test_llm_accuracy_is_measured_language_by_language(tmp_path, toy):
    table = validate_reader(fake_llm(tmp_path), toy / "annotated_answers.csv").set_index("language")
    assert table.loc["EN", "accuracy"] == 1 and table.loc["FR", "accuracy"] == 1
    assert table.loc["HI", "accuracy"] == pytest.approx(1 / 5)  # the fake LLM ignores Hindi
    assert table["n_responses"].sum() == 22


def test_llm_reader_on_a_tiny_local_model(tmp_path):
    """A tiny random LLM, created on the spot: the real loading, without downloading."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("torch")
    words = {"[UNK]": 0, "yes": 1, "no": 2, "unclear": 3}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(words, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[UNK]")
    tokenizer.save_pretrained(tmp_path / "llm")
    config = transformers.GPT2Config(vocab_size=4, n_embd=8, n_layer=1, n_head=2, n_positions=64)
    transformers.GPT2LMHeadModel(config).save_pretrained(tmp_path / "llm")

    reader = LLMReader(ReaderLLM(model=str(tmp_path / "llm")), tmp_path)
    decision, raw = reader.read("Yes.")
    assert decision in ("yes", "no", None) and isinstance(raw, str)


def test_unloadable_llm_is_a_clear_error(tmp_path):
    with pytest.raises(VLMShiftError, match="Cannot load"):
        LLMReader(ReaderLLM(model=str(tmp_path / "absent")), tmp_path)


def test_validation_file_must_exist_and_have_its_columns(tmp_path):
    with pytest.raises(VLMShiftError, match="not found"):
        validate_reader(fake_llm(tmp_path), tmp_path / "absent.csv")
    (tmp_path / "wrong.csv").write_text("response\nyes\n", encoding="utf-8")
    with pytest.raises(VLMShiftError, match="language"):
        validate_reader(fake_llm(tmp_path), tmp_path / "wrong.csv")


def test_formatting_marks_are_not_part_of_the_answer():
    assert clean("`yes`") == "yes"
    assert clean("~~no~~") == "no"


def test_validation_expects_yes_no_or_unclear(tmp_path):
    (tmp_path / "answers.csv").write_text("response,language,expected\nyes,EN,Yes\nno,EN,unknown\n", encoding="utf-8")
    with pytest.raises(VLMShiftError, match="unknown"):
        validate_reader(fake_llm(tmp_path), tmp_path / "answers.csv")


def test_validation_reads_a_file_saved_by_a_spreadsheet(tmp_path):
    text = "response;language;expected\nOui, c'est sûr;FR;Yes\nNon;FR;no\n"
    (tmp_path / "answers.csv").write_bytes(text.encode("cp1252"))
    table = validate_reader(fake_llm(tmp_path), tmp_path / "answers.csv")
    assert table["language"].tolist() == ["FR"] and table["n_responses"].tolist() == [2]
