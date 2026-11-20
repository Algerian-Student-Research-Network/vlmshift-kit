"""Image conditions: the same questions on changed images, paired with the original."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmshift.cli import do_prepare, main
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_experiment, read_yaml, write_yaml
from vlmshift.data.conditions import DEGRADATIONS, changed, parse_conditions, write_conditions


def conditions(toy, add, **options):
    table, issues = write_conditions(load_experiment(toy / "experiment.yaml"), parse_conditions(add), toy / "conditions", **options)
    return table, issues.frame()


def pixels(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def test_conditions_are_spelled_out():
    assert parse_conditions(["blank", "blur:3", "mask:marker"]) == ["blank", "blur_s3", "marker_masked"]
    assert parse_conditions(["noise"]) == [f"noise_s{severity}" for severity in range(1, 6)]
    for wrong in (["fog"], ["blur:6"], ["blank", "blank"], ["mask:"]):
        with pytest.raises(VLMShiftError, match="(?i)condition"):
            parse_conditions(wrong)


def test_each_row_comes_back_once_per_condition(toy):
    original = pd.read_csv(toy / "toy.csv")
    table, _ = conditions(toy, ["blank", "blur:2"])
    assert table["image_condition"].value_counts().to_dict() == {"real": len(original), "blank": len(original), "blur_s2": len(original)}
    assert set(original.columns) < set(table.columns)
    for path in table["image"]:
        assert (toy / "conditions" / path).is_file()
    assert (table.loc[table["image_condition"] == "real", "image"] == table.loc[table["image_condition"] == "real", "image_condition_source"]).all()


def test_the_changed_images_stay_in_the_sample_of_the_original(toy):
    """Core pairs each condition with the original image: one sample, one row per language and condition."""
    conditions(toy, ["blank", "noise:1"])
    raw = read_yaml(toy / "experiment.yaml")
    raw["data"] = "conditions/table.csv"
    raw["axes"]["image_condition"] = {"type": "controlled"}
    write_yaml(raw, toy / "experiment.yaml")
    variants = do_prepare(load_experiment(toy / "experiment.yaml")).df
    for _, rows in variants.groupby("sample_id"):
        assert set(rows["image_condition"]) == {"real", "blank", "noise_s1"} and set(rows["language"]) == {"EN", "FR"}
    before = pd.read_csv(toy / "toy.csv")
    assert variants["sample_id"].nunique() == before["image"].nunique()  # one sample per image, as before


def test_a_blank_image_keeps_the_size_and_loses_the_content(toy):
    table, _ = conditions(toy, ["blank"])
    blank = table[table["image_condition"] == "blank"].iloc[0]
    changed_pixels = pixels(toy / "conditions" / blank["image"])
    assert changed_pixels.shape == pixels(toy / "conditions" / blank["image_condition_source"]).shape
    assert (changed_pixels == 127).all()


def test_degradations_grow_with_the_severity(toy):
    image = Image.open(toy / "images" / "img1.png").convert("L")
    clean = np.asarray(image, dtype=float)
    for kind in DEGRADATIONS:
        distance = [np.abs(np.asarray(changed(image, f"{kind}_s{severity}", seed=0), dtype=float) - clean).mean()
                    for severity in (1, 3, 5)]
        assert distance[0] <= distance[1] <= distance[2] and distance[2] > 0, kind
    assert changed(image, "border_cropped", 0).size == image.size
    assert np.asarray(changed(image, "contrast_standardized", 0)).max() == 255


def test_the_same_seed_gives_the_same_files(toy, tmp_path):
    config = load_experiment(toy / "experiment.yaml")
    first, _ = write_conditions(config, ["noise_s3", "shuffled"], tmp_path / "a", seed=4)
    second, _ = write_conditions(config, ["noise_s3", "shuffled"], tmp_path / "b", seed=4)
    assert first.equals(second)
    name = first.loc[first["image_condition"] == "noise_s3", "image"].iloc[0]
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    other, _ = write_conditions(config, ["noise_s3"], tmp_path / "c", seed=5)
    assert (tmp_path / "c" / name).read_bytes() != (tmp_path / "a" / name).read_bytes()


def test_shuffled_gives_another_image_of_the_same_group(toy):
    table, _ = conditions(toy, ["shuffled"], within=["site"])
    mixed = table[table["image_condition"] == "shuffled"]
    real = table[table["image_condition"] == "real"].set_index(["image", "language"])
    assert len(mixed) == len(real)
    site_of = table[table["image_condition"] == "real"].drop_duplicates("image").set_index("image")["site"]
    assert (site_of.reindex(mixed["image"]).to_numpy() == mixed["site"].to_numpy()).all()  # stays inside the site
    own = real.reset_index()["image"].to_numpy()
    assert (mixed["image"].to_numpy() != own).all()  # never its own image


def test_same_label_swap_changes_the_site_not_the_label(toy):
    table, issues = conditions(toy, ["same_label_swap"], across="site")
    swapped = table[table["image_condition"] == "same_label_swap"]
    real = table[table["image_condition"] == "real"].drop_duplicates("image").set_index("image")
    assert len(swapped) and (real["site"].reindex(swapped["image"]).to_numpy() != swapped["site"].to_numpy()).all()
    assert (real["label"].reindex(swapped["image"]).to_numpy() == swapped["label"].to_numpy()).all()
    with pytest.raises(VLMShiftError, match="same_label_swap needs --across"):
        conditions(toy, ["same_label_swap"])


def test_a_mask_needs_boxes_and_never_invents_them(toy):
    pd.DataFrame([{"image": "images/img1.png", "box_type": "marker", "x1": 0, "y1": 0, "x2": 9, "y2": 9}]).to_csv(
        toy / "boxes.csv", index=False)
    table, issues = conditions(toy, ["mask:marker"], boxes=toy / "boxes.csv")
    masked = table[table["image_condition"] == "marker_masked"]
    assert set(masked["image_condition_source"]) == {"../images/img1.png"}  # the only image with a box
    assert "have no image under this condition" in issues.loc[issues["code"] == "condition_left_out", "message"].iloc[0]
    corner = pixels(toy / "conditions" / masked["image"].iloc[0])[:10, :10]
    assert corner.min() == corner.max()  # one flat gray
    pd.DataFrame([{"image": "elsewhere.png", "box_type": "marker", "x1": 0, "y1": 0, "x2": 9, "y2": 9}]).to_csv(
        toy / "boxes.csv", index=False)
    with pytest.raises(VLMShiftError, match="images that are not in the table"):
        conditions(toy, ["mask:marker"], boxes=toy / "boxes.csv")


def test_the_command_writes_the_table_and_says_what_to_do_next(toy, capsys):
    main(["conditions", str(toy / "experiment.yaml"), "--add", "blank", "--axis", "view"])
    printed = capsys.readouterr().out
    assert "16 real, 16 blank" in printed and "add the axis view: {type: controlled}" in printed
    assert Path(toy / "conditions" / "table.csv").is_file()
    with pytest.raises(SystemExit, match="already has a column site"):
        main(["conditions", str(toy / "experiment.yaml"), "--add", "blank", "--axis", "site"])
