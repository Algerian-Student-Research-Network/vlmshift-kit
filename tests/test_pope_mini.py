"""POPE mini: the fetch script and the comparison with the paper, without network or model."""
import importlib.util
import io

import pandas as pd
import pytest

from conftest import ROOT
from vlmshift.config.load import load_analysis, load_experiment, write_yaml

EXAMPLE = ROOT / "examples" / "pope_mini"


def script(name: str):
    spec = importlib.util.spec_from_file_location(f"pope_{name}", EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch, compare = script("fetch"), script("compare")


def pope_line(image: int, text: str, label: str) -> dict:
    return {"question_id": 1, "image": f"COCO_val2014_{image:012d}.jpg", "text": text, "label": label}


QUESTIONS = {
    "random": [pope_line(9, "Is there a dog in the image?", "yes"), pope_line(9, "Is there an apple in the image?", "no"),
               pope_line(3, "Is there a dog in the image?", "yes"), pope_line(5, "Is there a cat in the image?", "yes")],
    "popular": [pope_line(9, "Is there a dog in the image?", "yes"), pope_line(9, "Is there a chair in the image?", "no"),
                pope_line(3, "Is there a dog in the image?", "yes")],
    "adversarial": [pope_line(9, "Is there a dog in the image?", "yes"), pope_line(9, "Is there a bone in the image?", "no"),
                    pope_line(3, "Is there a dog in the image?", "yes")],
}


def test_the_same_images_are_chosen_every_time():
    # Image 5 is asked about in one mode only: it cannot be compared across modes.
    assert fetch.chosen_images(QUESTIONS, 1) == ["COCO_val2014_000000000003.jpg"]
    assert fetch.chosen_images(QUESTIONS, 10) == ["COCO_val2014_000000000003.jpg", "COCO_val2014_000000000009.jpg"]


def test_the_table_has_what_the_experiment_reads():
    rows = pd.DataFrame(fetch.table_rows(QUESTIONS, ["COCO_val2014_000000000009.jpg"]))
    assert len(rows) == 6 and set(rows["mode"]) == set(fetch.MODES)
    apple = rows[rows["object"].eq("apple")].iloc[0]
    assert (apple["phrase"], apple["label"], apple["image_id"]) == ("an apple", "negative", "000000000009")
    assert apple["image"] == "images/COCO_val2014_000000000009.jpg"
    for name in ("experiment.yaml", "paper.yaml"):
        config = load_experiment(EXAMPLE / name)
        roles = {config.roles.image, config.roles.question, config.roles.label, config.roles.unit}
        assert roles | config.task.columns() <= set(rows.columns), name
        assert set(config.axes) <= set(rows.columns), name


def test_an_unexpected_question_stops_the_script():
    odd = {"random": [pope_line(9, "Is the dog asleep?", "yes")]}
    with pytest.raises(ValueError, match="Unexpected POPE question"):
        fetch.table_rows(odd, ["COCO_val2014_000000000009.jpg"])


def test_a_file_is_downloaded_once_and_never_left_half_written(tmp_path, monkeypatch):
    calls = []

    def answer(url, timeout):
        calls.append(url)
        return io.BytesIO(b"image bytes")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", answer)
    target = tmp_path / "images" / "a.jpg"
    fetch.fetch("http://example.test/a.jpg", target)
    fetch.fetch("http://example.test/a.jpg", target)
    assert calls == ["http://example.test/a.jpg"] and target.read_bytes() == b"image bytes"
    assert [path.name for path in target.parent.iterdir()] == ["a.jpg"]


def test_the_analysis_compares_the_modes():
    config = load_analysis(EXAMPLE / "analysis.yaml")
    assert [(m.name, m.compare, m.split_by) for m in config.measures] == [("mode", "mode", [])]
    assert config.axes["mode"].reference == "random"


def metric_rows(model: str, accuracy: dict, recall: float) -> list[dict]:
    """What VLMShift writes in metrics.parquet for one model, in the form compare.py reads."""
    rows = []
    for mode, value in accuracy.items():
        spec = 2 * value - recall
        yes = (recall + 1 - spec) / 2  # as many present objects as absent ones
        for metric, number in {"ba": value, "sens": recall, "spec": spec, "yes_rate": yes}.items():
            rows.append({"measure": "mode", "model": model, "split": None, "level_a": mode, "level_b": None,
                         "metric": metric, "mode": "valid", "value": number, "ci95_low": number - 0.02, "ci95_high": number + 0.02})
        if mode != "random":
            drop = value - accuracy["random"]
            rows.append({"measure": "mode", "model": model, "split": None, "level_a": "random", "level_b": mode,
                         "metric": "delta_ba", "mode": "valid", "value": drop, "ci95_low": drop - 0.03, "ci95_high": drop + 0.01})
    # Same numbers in strict mode: never read twice.
    rows += [{**row, "mode": "strict"} for row in rows]
    return rows


@pytest.fixture
def analysis(tmp_path):
    """A run folder laid out as VLMShift writes it, with one model of the paper and one other."""
    run = tmp_path / "runs" / "r1"
    folder = run / "analyses" / "a1"
    folder.mkdir(parents=True)
    write_yaml({"models": [
        {"name": "iblip", "adapter": "huggingface", "id": "Salesforce/instructblip-vicuna-13b"},
        {"name": "clip", "adapter": "contrastive", "id": "openai/clip-vit-base-patch32"},
    ]}, run / "experiment.yaml")
    rows = metric_rows("iblip", {"random": 0.8873, "popular": 0.78, "adversarial": 0.7437}, recall=0.9393)
    rows += metric_rows("clip", {"random": 0.77, "popular": 0.74, "adversarial": 0.67}, recall=0.64)
    pd.DataFrame(rows).to_parquet(folder / "metrics.parquet")
    (folder / "report").mkdir()
    pd.DataFrame(rows).to_parquet(folder / "report" / "metrics.parquet")  # the report keeps a copy
    (folder / "report" / "report.md").write_text("# VLMShift report", encoding="utf-8")
    return folder


def test_the_measures_sit_next_to_the_published_numbers(analysis, capsys):
    assert compare.main([str(analysis)]) == 0
    table = pd.read_csv(analysis / "report" / "pope_comparison.csv")
    assert set(table["model"]) == {"iblip"}  # CLIP is not in the paper
    assert len(table) == 12
    random = table[table["mode"].eq("random")].set_index("metric")
    # Specificity is not in the paper: it follows from accuracy and recall.
    assert random.loc["specificity", "published"] == pytest.approx(2 * 88.73 - 93.93)
    assert random["inside"].all()
    popular = table[table["mode"].eq("popular")].set_index("metric")
    assert popular.loc["accuracy", "measured"] == 78.0 and popular.loc["accuracy", "published"] == 81.37
    assert not popular.loc["accuracy", "inside"]  # 81.37 is outside 76 to 80
    assert "inside the interval" in capsys.readouterr().out


def test_the_drops_have_the_sign_of_a_loss(analysis):
    assert compare.main([str(analysis / "report" / "report.md")]) == 0  # the path vlmshift prints
    drops = pd.read_csv(analysis / "report" / "pope_drops.csv")
    clip = drops[drops["model"].eq("clip")].set_index("mode")
    assert clip.loc["adversarial", ["drop", "low", "high"]].tolist() == pytest.approx([10.0, 9.0, 13.0])
    paper = drops[drops["source"].eq("paper") & drops["model"].eq("InstructBLIP")].set_index("mode")
    assert paper.loc["adversarial", "drop"] == pytest.approx(88.73 - 74.37, abs=0.05)
    assert len(drops[drops["source"].eq("paper")]) == 10  # 5 models, 2 modes


def test_a_wrong_folder_explains_the_use(tmp_path, capsys):
    assert compare.main([str(tmp_path)]) == 2
    assert "compare.py <path of report.md>" in capsys.readouterr().err


def test_a_model_of_the_paper_is_known_by_its_id_for_its_adapter():
    paper = compare.published(pd.read_csv(compare.REFERENCE))
    experiment = {"models": [
        {"name": "a", "adapter": "huggingface", "id": "Salesforce/instructblip-vicuna-13b"},
        {"name": "b", "adapter": "vlmevalkit", "id": "MiniGPT-4-v1-13B"},
        {"name": "c", "adapter": "vlmevalkit", "id": "Salesforce/instructblip-vicuna-13b"},  # not a VLMEvalKit name
        {"name": "d", "adapter": "contrastive", "id": "openai/clip-vit-base-patch32"},
        {"name": "instructblip_13b", "adapter": "vlmevalkit", "id": None},  # no id: the name is what is loaded
    ]}
    assert compare.paper_names(experiment, paper) == {"a": "InstructBLIP", "b": "MiniGPT-4", "instructblip_13b": "InstructBLIP"}


def test_a_run_with_no_estimable_number_still_compares(analysis, capsys):
    # Too few units everywhere: every number is empty, and parquet keeps these columns as text.
    metrics = pd.read_parquet(analysis / "metrics.parquet")
    empty = metrics.assign(value=None, ci95_low=None, ci95_high=None)
    empty.to_parquet(analysis / "metrics.parquet")
    assert pd.read_parquet(analysis / "metrics.parquet")["value"].dtype == object
    assert compare.main([str(analysis)]) == 0
    drops = pd.read_csv(analysis / "report" / "pope_drops.csv")
    assert drops.loc[drops["source"].eq("measured"), "drop"].isna().all()


def test_paper_yaml_runs_the_model_of_the_paper():
    config = load_experiment(EXAMPLE / "paper.yaml")
    paper = compare.published(pd.read_csv(compare.REFERENCE))
    names = compare.paper_names({"models": [model.model_dump() for model in config.models]}, paper)
    assert set(names.values()) == {"InstructBLIP"}
    assert all(model.revision for model in config.models)  # pinned weights, for a result to compare


def test_the_colab_notebook_matches_its_generator():
    nbformat = pytest.importorskip("nbformat")
    make = importlib.util.spec_from_file_location("make_colab", ROOT / "tools" / "make_colab.py")
    module = importlib.util.module_from_spec(make)
    make.loader.exec_module(module)
    assert (EXAMPLE / "colab.ipynb").read_text(encoding="utf-8") == nbformat.writes(module.build()) + "\n", (
        "colab.ipynb is out of date: run python tools/make_colab.py"
    )
