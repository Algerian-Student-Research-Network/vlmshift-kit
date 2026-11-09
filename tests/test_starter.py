"""data/starter.py and what the terminal shows: from a folder of images to a first look."""
import io
import shutil
import sys

import pandas as pd
import pytest

from conftest import FIXTURES
from vlmshift.cli import main, show_issues
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_analysis
from vlmshift.data.starter import parse_questions, write_starter

SOURCE = FIXTURES / "toy" / "images"
QUESTIONS = ["--question", "EN=Is there a circle?", "--question", "FR=Y a-t-il un cercle ?"]


@pytest.fixture
def folder(tmp_path):
    """Two images with a label and the same file name, and one with no label."""
    for name, number in (("images/positive/a.png", 1), ("images/negative/a.png", 2), ("images/b.png", 3)):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SOURCE / f"img{number}.png", target)
    return tmp_path


def start(folder, *options):
    main(["init", str(folder / "images"), *QUESTIONS, *options])
    return pd.read_csv(folder / "table.csv")


def test_one_row_per_image_and_level(folder):
    table = start(folder)
    assert len(table) == 6
    assert table.groupby("image")["language"].apply(set).eq({"EN", "FR"}).all()
    assert table.loc[table["language"] == "FR", "question"].eq("Y a-t-il un cercle ?").all()


def test_the_terminal_says_what_was_found_in_a_fixed_order(folder, capsys):
    start(folder)
    assert "6 rows, 3 images (1 positive, 1 negative, 1 unknown)" in capsys.readouterr().out


def test_subfolders_give_the_labels(folder):
    labels = start(folder).drop_duplicates("image").set_index("image")["label"]
    assert labels.to_dict() == {
        "images/b.png": "unknown", "images/negative/a.png": "negative", "images/positive/a.png": "positive",
    }


def test_images_with_the_same_name_stay_different_subjects(folder):
    assert start(folder)["subject"].nunique() == 3


def test_the_files_work_with_the_next_steps(folder, capsys):
    start(folder)
    main(["prepare", str(folder / "experiment.yaml")])
    output = capsys.readouterr().out
    assert "Variants: 6 rows" in output and "image_missing" not in output
    assert load_analysis(folder / "analysis.yaml", "any").axes["language"].reference == "EN"


def test_the_axis_can_have_another_name(folder):
    assert "condition" in start(folder, "--axis", "condition").columns
    assert "condition: {type: controlled}" in (folder / "experiment.yaml").read_text(encoding="utf-8")


def test_an_existing_file_is_never_replaced(folder):
    (folder / "experiment.yaml").write_text("mine", encoding="utf-8")
    with pytest.raises(VLMShiftError, match="Nothing was changed"):
        write_starter(folder / "images", "language", {"EN": "a", "FR": "b"})
    assert (folder / "experiment.yaml").read_text(encoding="utf-8") == "mine"
    assert not (folder / "table.csv").exists()


def test_a_folder_with_no_image_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(VLMShiftError, match="No image"):
        write_starter(tmp_path / "empty", "language", {"EN": "a", "FR": "b"})
    with pytest.raises(VLMShiftError, match="not found"):
        write_starter(tmp_path / "absent", "language", {"EN": "a", "FR": "b"})


@pytest.mark.parametrize("items, message", [
    (["EN=a"], "at least two"),
    (["EN=a", "EN=b"], "twice"),
    (["no separator", "FR=b"], "LEVEL=text"),
    (["=a", "FR=b"], "LEVEL=text"),
    (["EN=", "FR=b"], "LEVEL=text"),
])
def test_questions_must_be_well_formed(items, message):
    with pytest.raises(VLMShiftError, match=message):
        parse_questions(items)


def test_a_question_can_contain_an_equal_sign():
    assert parse_questions(["EN=Is a=b?", "FR=c"])["EN"] == "Is a=b?"


def test_the_terminal_announces_a_missing_image(toy, capsys):
    (toy / "images" / "img1.png").unlink()
    main(["prepare", str(toy / "experiment.yaml")])
    assert "! image_missing (2): Image not found" in capsys.readouterr().out


def test_the_terminal_shows_the_issues_of_a_run_once(toy, capsys):
    main(["run", str(toy / "experiment.yaml")])
    assert capsys.readouterr().out.count("model_error") == 1


def test_the_terminal_shows_the_issues_of_the_audit(toy, capsys):
    main(["prepare", str(toy / "experiment.yaml")])
    assert "! low_count (8)" in capsys.readouterr().out


def test_an_issue_of_the_variants_is_announced_once_even_by_a_run(toy, edit, capsys):
    table = pd.read_csv(toy / "toy.csv")
    table["shape"] = ["a circle"] * (len(table) - 1) + [None]  # the last row has nothing to fill the template
    table.to_csv(toy / "toy.csv", index=False)
    edit(lambda raw: raw["task"].update(template="Is there {shape}?"))
    main(["run", str(toy / "experiment.yaml")])
    assert capsys.readouterr().out.count("! empty_template_value (1)") == 1


def test_the_terminal_shows_the_issues_of_the_analysis(toy, capsys):
    main(["run", str(toy / "experiment.yaml"), "--only", "fake-clean"])  # no vectors yet: the probe cannot run
    run = next((toy / "workspace" / "runs").iterdir())
    main(["analyze", str(toy / "analysis.yaml"), "--run", str(run)])
    assert "! probe_skipped (1)" in capsys.readouterr().out


def test_only_the_issues_of_the_step_are_shown(tmp_path, capsys):
    rows = [[3, "warning", "from_step_3", None, "a"], [4, "warning", "from_step_4", None, "b"]]
    pd.DataFrame(rows, columns=["step", "severity", "code", "id", "message"]).to_parquet(tmp_path / "issues.parquet")
    show_issues(tmp_path / "issues.parquet", step=4)
    output = capsys.readouterr().out
    assert "from_step_4" in output and "from_step_3" not in output


def test_the_audit_command_shows_the_counts(toy, capsys):
    main(["audit", str(toy / "experiment.yaml")])
    output = capsys.readouterr().out
    assert "n_units" in output and "language x site" not in output


def test_other_commands_do_not_repeat_the_counts(toy, capsys):
    main(["prepare", str(toy / "experiment.yaml")])
    assert "n_units" not in capsys.readouterr().out


def test_audit_and_exact_texts_can_be_read_in_a_spreadsheet(toy):
    main(["prepare", str(toy / "experiment.yaml")])
    table = next((toy / "workspace" / "tables").iterdir())
    for name in ("counts", "samples"):
        assert len(pd.read_csv(table / "audit" / f"{name}.csv")) == len(pd.read_parquet(table / "audit" / f"{name}.parquet"))
    variants = next((table / "variants").iterdir())
    texts = pd.read_csv(variants / "variants.csv")["prompt"]
    assert texts.tolist() == pd.read_parquet(variants / "variants.parquet")["prompt"].tolist()
    assert texts.str.contains("Répondez").any()


@pytest.mark.parametrize("axis", ["label", "question", "image", "subject", "id", "a: b", "2nd", "_root", "_path"])
def test_the_axis_must_be_a_free_simple_name(folder, axis):
    with pytest.raises(VLMShiftError, match="--axis"):
        write_starter(folder / "images", axis, parse_questions(QUESTIONS[1::2]))
    assert not (folder / "table.csv").exists()


def test_label_folders_can_be_written_in_capitals(tmp_path):
    for name in ("images/Positive/a.png", "images/NEGATIVE/b.png", "images/Positive/a.jpg"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SOURCE / "img1.png", tmp_path / name)
    table = start(tmp_path)
    assert sorted(table.drop_duplicates("image")["label"]) == ["negative", "positive", "positive"]
    assert table["subject"].nunique() == 3  # a.png and a.jpg are two files: two subjects


@pytest.mark.parametrize("axis", ["langue_é", "भाषा", "_x2"])
def test_an_axis_name_can_use_any_alphabet(folder, axis):
    assert axis in write_starter(folder / "images", axis, parse_questions(QUESTIONS[1::2])).columns


def test_init_says_where_the_files_are(folder, monkeypatch, capsys):
    monkeypatch.chdir(folder.parent)
    main(["init", f"{folder.name}/images", *QUESTIONS])
    advice = capsys.readouterr().out.strip().splitlines()[-1]
    assert (folder.parent / advice.rsplit(" ", 1)[1]).is_file()  # the command it advises can run from here


def test_the_commands_run_in_a_notebook(toy, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())  # a notebook output has no reconfigure
    main(["import", str(toy / "experiment.yaml")])
    assert "Table:" in sys.stdout.getvalue()
