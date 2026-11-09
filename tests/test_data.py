"""data/: step 1, import and edge cases."""
import shutil
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_experiment
from vlmshift.data import table as table_module
from vlmshift.data.table import build_table


def read_issues(folder):
    return pd.concat(pd.read_parquet(path) for path in folder.glob("workspace/tables/*/issues.parquet"))


def test_toy_table(toy):
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert len(table) == 16
    assert table["sample_id"].nunique() == 8
    assert table.groupby("sample_id")["language"].nunique().eq(2).all()  # EN and FR together
    assert table["id"].str.fullmatch(r"l_[0-9a-f]{16}").all()
    assert table["sample_id"].str.fullmatch(r"s_[0-9a-f]{16}").all()
    assert table["image_hash"].str.len().eq(64).all()
    assert table["unit"].nunique() == 4


def test_parquet_keeps_schema_version_and_fingerprint(toy):
    step = build_table(load_experiment(toy / "experiment.yaml"))
    metadata = pq.read_schema(step.dir / "table.parquet").metadata
    assert metadata[b"vlmshift.schema"] == b"1"
    assert metadata[b"vlmshift.fingerprint"] == step.hash.encode()


def test_absent_values_are_null(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).assign(note="")
    data.to_csv(toy / "toy.csv", index=False)
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert table["note"].isna().all()


def test_reimport_with_shuffled_lines_gives_same_ids_and_hash(toy):
    config = load_experiment(toy / "experiment.yaml")
    first = build_table(config)
    pd.read_csv(toy / "toy.csv", dtype=str).sample(frac=1, random_state=1).to_csv(toy / "toy.csv", index=False)
    second = build_table(config)
    assert first.hash == second.hash
    assert first.df["id"].tolist() == second.df["id"].tolist()


def test_moved_dataset_keeps_ids(toy, tmp_path):
    first = build_table(load_experiment(toy / "experiment.yaml")).hash
    moved = toy.rename(tmp_path / "elsewhere")
    assert build_table(load_experiment(moved / "experiment.yaml")).hash == first


def test_one_changed_value_changes_the_hash(toy):
    config = load_experiment(toy / "experiment.yaml")
    first = build_table(config).hash
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[0, "question"] = "Another question?"
    data.to_csv(toy / "toy.csv", index=False)
    assert build_table(config).hash != first


def test_changed_image_changes_the_hash(toy):
    config = load_experiment(toy / "experiment.yaml")
    first = build_table(config).hash
    (toy / "images" / "img1.png").write_bytes(b"other content")
    assert build_table(config).hash != first


def test_text_is_normalized_before_hashing(toy):
    config = load_experiment(toy / "experiment.yaml")
    first = build_table(config).hash
    text = (toy / "toy.csv").read_text(encoding="utf-8")
    (toy / "toy.csv").write_text(unicodedata.normalize("NFD", text), encoding="utf-8")
    assert build_table(config).hash == first


def test_broken_table_signals_every_case(toy):
    with pytest.raises(VLMShiftError):
        build_table(load_experiment(toy / "broken.yaml"))
    issues = read_issues(toy)
    severity = dict(zip(issues["code"], issues["severity"]))
    assert severity == {
        "duplicate_line": "warning",
        "image_missing": "warning",
        "label_invalid": "error",
        "original_id_invalid": "warning",
        "sample_many_units": "error",
        "sample_many_labels": "error",
        "same_levels": "error",  # two questions of one sample in English: their pairs would be random
        "axis_single_level": "warning",
    }
    assert (issues["code"] == "label_invalid").sum() == 2  # the empty label and "maybe"


def test_missing_role_column_stops_with_a_clear_message(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str).drop(columns="label")
    data.to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match="label"):
        build_table(load_experiment(toy / "experiment.yaml"))


def test_missing_axis_column_stops(toy, edit):
    config = edit(lambda raw: raw["axes"].update(absent={"type": "observed"}))
    with pytest.raises(VLMShiftError, match="absent"):
        build_table(config)


def test_roles_get_their_standard_name(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).rename(columns={"label": "outcome"})
    data.to_csv(toy / "toy.csv", index=False)
    table = build_table(edit(lambda raw: raw["roles"].update(label="outcome"))).df
    assert "label" in table.columns and "outcome" not in table.columns


def test_free_column_cannot_use_a_reserved_name(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).assign(status="x")
    data.to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match="status"):
        build_table(load_experiment(toy / "experiment.yaml"))


def test_id_collision_is_an_error(toy, monkeypatch):
    monkeypatch.setattr(table_module, "row_ids", lambda table: ["l_same"] * len(table))
    with pytest.raises(VLMShiftError):
        build_table(load_experiment(toy / "experiment.yaml"))
    assert "id_collision" in set(read_issues(toy)["code"])


def test_parquet_input_and_several_files(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.iloc[:8].to_parquet(toy / "a.parquet")
    data.iloc[8:].to_csv(toy / "b.csv", index=False)
    table = build_table(edit(lambda raw: raw.update(data=["a.parquet", "b.csv"]))).df
    assert len(table) == 16


def test_row_with_a_missing_image_is_left_out(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[0, "image"] = "images/absent.png"
    data.to_csv(toy / "toy.csv", index=False)
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert len(table) == 15 and table["image_hash"].notna().all()


def test_dataset_id_is_kept_as_original_id(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(id=[f"case{i}" for i in range(len(data))]).to_csv(toy / "toy.csv", index=False)
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert sorted(table["original_id"]) == sorted(f"case{i}" for i in range(16))
    assert table["id"].str.startswith("l_").all()


def test_unit_defaults_to_sample(toy, edit):
    table = build_table(edit(lambda raw: raw["roles"].pop("unit"))).df
    assert table["unit"].equals(table["sample_id"])


def test_a_table_with_no_image_found_stops_at_once(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(image="wrong/" + data["image"]).to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match="No row left: no image was found"):
        build_table(load_experiment(toy / "experiment.yaml"))


def test_a_table_saved_with_semicolons_is_read(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.to_csv(toy / "toy.csv", index=False, sep=";")
    assert len(build_table(load_experiment(toy / "experiment.yaml")).df) == 16


def test_data_files_must_have_the_same_columns(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.iloc[:8].to_csv(toy / "a.csv", index=False)
    data.iloc[8:].drop(columns="subject").to_csv(toy / "b.csv", index=False)
    with pytest.raises(VLMShiftError, match=r"same columns. Different: \['subject'\]"):
        build_table(edit(lambda raw: raw.update(data=["a.csv", "b.csv"])))


def test_an_invalid_label_says_which_image(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.loc[0, "label"] = "maybe"
    data.to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match=r"Invalid label \('maybe'\) for .*img1.png"):
        build_table(load_experiment(toy / "experiment.yaml"))


def test_a_spreadsheet_csv_in_windows_1252_with_semicolons_is_read(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data["question"] = data["question"].str.replace("Y a-t-il", "Regardez, y a-t-il", regex=False)
    data.to_csv(toy / "toy.csv", index=False, sep=";", encoding="cp1252")  # Excel set to French
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert len(table) == 16
    assert table["question"].str.contains("Regardez, y a-t-il", regex=False).sum() == 8


def test_one_column_can_serve_two_roles(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(case=data["image"].str.extract(r"(img\d)")[0]).to_csv(toy / "toy.csv", index=False)
    table = build_table(edit(lambda raw: raw["roles"].update(unit="case", original_id="case"))).df
    cases = table["image"].str.extract(r"(img\d)")[0].tolist()
    assert table["unit"].tolist() == cases and table["original_id"].tolist() == cases  # each role gets the column


def test_a_column_named_after_a_role_takes_it(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    data.assign(sample_id=data["image"].str.extract(r"(img\d)")[0]).to_csv(toy / "toy.csv", index=False)
    table = build_table(load_experiment(toy / "experiment.yaml")).df
    assert sorted(table["sample_id"].unique()) == [f"img{number}" for number in range(1, 9)]


def test_an_id_column_next_to_an_original_id_is_a_clear_error(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    numbers = range(len(data))
    data.assign(id=[f"row{i}" for i in numbers], original_id=[f"src{i}" for i in numbers]).to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match="VLMShift writes its own id"):
        build_table(load_experiment(toy / "experiment.yaml"))


def test_the_same_row_in_two_folders_is_a_duplicate_not_a_collision(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    for name, part in (("a", data.iloc[:9]), ("b", data.iloc[8:])):  # row 8 is in both files
        shutil.copytree(toy / "images", toy / name / "images")
        part.to_csv(toy / name / "part.csv", index=False)
    table = build_table(edit(lambda raw: raw.update(data=["a/part.csv", "b/part.csv"]))).df
    assert len(table) == 16
    assert "duplicate_line" in set(read_issues(toy)["code"])


def test_two_copies_that_differ_only_by_unicode_form_are_duplicates(toy):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    copy = data.loc[data["language"] == "FR"].iloc[[0]]
    copy = copy.assign(question=[unicodedata.normalize("NFD", text) for text in copy["question"]])
    assert copy["question"].iloc[0] != data.loc[copy.index[0], "question"]  # the copy is really written otherwise
    pd.concat([data, copy]).to_csv(toy / "toy.csv", index=False)
    assert len(build_table(load_experiment(toy / "experiment.yaml")).df) == 16


def test_several_questions_per_image_get_their_own_samples(toy, edit):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False)
    pd.concat([data.assign(concept="circle"), data.assign(concept="square")]).to_csv(toy / "toy.csv", index=False)
    table = build_table(edit(lambda raw: raw["task"].update(template="Is there a {concept}? Answer yes or no."))).df
    assert table["sample_id"].nunique() == 16  # one per image and concept, not one per image
    assert table.groupby("sample_id")["language"].nunique().eq(2).all()


def test_a_16_bit_image_is_flagged(toy):
    pixels = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
    Image.fromarray(pixels).save(toy / "images" / "img1.png")
    build_table(load_experiment(toy / "experiment.yaml"))
    assert "image_16bit" in set(read_issues(toy)["code"])


def test_an_empty_csv_gives_a_clear_message(toy):
    (toy / "toy.csv").write_text("", encoding="utf-8")
    with pytest.raises(VLMShiftError, match="cannot be read as a table"):
        build_table(load_experiment(toy / "experiment.yaml"))


@pytest.mark.parametrize("name", ["_root", "_path"])
def test_a_free_column_cannot_use_an_internal_name(toy, name):
    data = pd.read_csv(toy / "toy.csv", dtype=str, keep_default_na=False).assign(**{name: "x"})
    data.to_csv(toy / "toy.csv", index=False)
    with pytest.raises(VLMShiftError, match=f"Reserved names.*{name}"):
        build_table(load_experiment(toy / "experiment.yaml"))
