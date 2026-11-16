"""Med: the experiments of the paper, from its plans (plans.py) and its recorded answers (replay.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import FIXTURES, ROOT
from vlmshift.common import VLMShiftError
from vlmshift.config.load import load_analysis, load_experiment, read_yaml

MED = ROOT / "packages" / "med"
sys.path.insert(0, str(MED))

import audits  # noqa: E402
import bank as med_bank  # noqa: E402
import plans  # noqa: E402
import replay  # noqa: E402

BANK, LATIN = med_bank.load_bank(), med_bank.load_romanized()
ARMS = read_yaml(MED / "configs" / "arms.yaml")
SITES = {"NIH-ChestXray14": "USA", "PadChest-GR": "Spain"}


def test_every_prompt_of_the_paper_is_rebuilt_from_the_bank():
    """The 1,493 distinct questions the October 2026 runs sent, word for word."""
    sent = pd.read_csv(FIXTURES / "med" / "paper_prompts.csv", dtype=str)
    assert len(sent) == 1493 and set(sent["language"]) == set(med_bank.LANGUAGES)
    assert set(sent["shown_concept"]) == set(BANK["concepts"])
    for row in sent.to_dict("records"):
        assert plans.texts_of(row, BANK, LATIN) == (row["prompt"], row["positive_text"], row["negative_text"]), row


def plan_row(sample, patient, site, concept, y, experiment, arm="main", language="EN", frame=None, term=None,
             variant="canonical", answer_format="english_yes_no", condition="real", image=None, **more) -> dict:
    """One row of an evaluation plan, with the texts the study would have written."""
    row = {
        "sample_id": sample, "patient_id": patient, "dataset": site, "country": SITES[site], "scanner_type": None,
        "image_path": f"{sample}.png", "resolved_image_path": image or f"/content/images/{site}/{sample}.png",
        "pathology": concept, "y_true": str(y), "image_condition": condition, "arm": arm, "experiment": experiment,
        "language": language, "frame_language": frame or language, "term_language": term or language,
        "prompt_variant": variant, "answer_format": answer_format, "negated": "1" if variant == "negated" else "0",
        "script_variant": "native", "term_condition": "true", "shown_concept": concept, **more,
    }
    row["prompt"], row["positive_text"], row["negative_text"] = plans.texts_of(row, BANK, LATIN)
    return row


def small_plans() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four patients at two sites, asked as in every arm of the paper."""
    samples = [("s1", "p1", "NIH-ChestXray14", "nodule", 1), ("s2", "p2", "NIH-ChestXray14", "nodule", 0),
               ("s3", "p3", "PadChest-GR", "normal", 1), ("s4", "p4", "PadChest-GR", "normal", 0)]
    core, supp = [], []
    for sample in samples:
        for language in ("EN", "FR", "HI"):
            core.append(plan_row(*sample, "core", language=language))
            core.append(plan_row(*sample, "core", arm="factorial", language=language, answer_format="numeric_10"))
            core.append(plan_row(*sample, "core", arm="factorial", language=language, answer_format="label_ab"))
            # The plan gives the truth of the question as asked: the negated one has the opposite truth.
            flipped = (*sample[:4], 1 - sample[4])
            core.append(plan_row(*flipped, "negation", language=language, variant="negated", answer_format="numeric_10"))
            for condition in ("real", "blank"):
                image = None if condition == "real" else f"/content/drive/MyDrive/ASRN/controls/visual/{sample[0]}_blank.png"
                core.append(plan_row(*sample, "visual_control", arm="control", language=language, condition=condition, image=image))
            for condition in ("real", "blur_s1"):
                image = None if condition == "real" else f"/content/drive/MyDrive/ASRN/controls/corrupt_blur/{sample[0]}_blur_s1.png"
                core.append(plan_row(*sample, "corruption", arm="control", language=language, condition=condition, image=image))
            other = "mass" if sample[3] != "mass" else "nodule"
            supp.append(plan_row(*sample, "term_permutation", arm="supplementary", language=language,
                                 term_condition="permuted", shown_concept=other))
        for contrast in ("FR", "HI"):
            for frame, term in (("EN", "EN"), (contrast, "EN"), ("EN", contrast), (contrast, contrast)):
                supp.append(plan_row(*sample, "term_frame_factorial", arm="supplementary", language=frame, frame=frame,
                                     term=term, answer_format="numeric_10", contrast_language=contrast))
        supp.append(plan_row(*sample, "transliteration", arm="supplementary", language="HI", script_variant="romanized"))
    return pd.DataFrame(core), pd.DataFrame(supp)


@pytest.fixture
def paper_folder(tmp_path) -> Path:
    core, supp = small_plans()
    folder = tmp_path / "paper"
    folder.mkdir()
    core.to_csv(folder / "eval_plan_core.csv", index=False)
    supp.to_csv(folder / "eval_plan_core_supp.csv", index=False)
    return folder


def table_of(folder: Path, arm: str) -> pd.DataFrame:
    return pd.read_parquet(folder / "arms" / arm / "table.parquet")


def test_each_arm_of_the_paper_gets_its_table_and_its_two_files(paper_folder):
    written = plans.write_arms(paper_folder, paper_folder)
    assert list(written) == list(ARMS) == list(plans.EXTRA)
    assert written["main"] == 4 * 3 and written["formats"] == 4 * 3 * 3 and written["negation"] == 4 * 3 * 2
    assert written["factorial"] == 4 * 7  # EN/EN once, then three cells for each of the two languages
    assert written["transliteration"] == 4 * 2 and written["permutation"] == 4 * 3 * 2
    for arm in ARMS:
        experiment = load_experiment(paper_folder / "arms" / arm / "experiment.yaml")
        analysis = load_analysis(paper_folder / "arms" / arm / "analysis.yaml")
        table = table_of(paper_folder, arm)
        controlled = [axis for axis, spec in experiment.axes.items() if spec.type == "controlled"]
        assert set(experiment.axes) <= set(table.columns) and not table.duplicated(["sample", *controlled]).any(), arm
        assert table.groupby("sample")["label"].nunique().max() == 1, arm  # one truth per sample, as Core needs
        assert experiment.roles.sample_id == "sample" and analysis.stats.min_units == 5
        assert len(experiment.models) == 10 and analysis.report.notice.startswith(f"Arm {arm} of the paper.")


def test_arms_share_one_workspace_so_no_question_is_asked_twice(paper_folder):
    plans.write_arms(paper_folder, paper_folder)
    workspaces = {load_experiment(paper_folder / "arms" / arm / "experiment.yaml").workspace for arm in ARMS}
    assert workspaces == {str(paper_folder / "workspace")}


def test_a_negated_question_keeps_the_label_of_the_finding(paper_folder):
    plans.write_arms(paper_folder, paper_folder)
    table = table_of(paper_folder, "negation")
    labels = table.groupby(["sample", "negated"])["label"].first().unstack()
    assert (labels["0"] == labels["1"]).all() and set(table["answer_format"]) == {"numeric_10"}
    negated = table[(table["sample"] == "s3:normal") & (table["negated"] == "1") & (table["language"] == "EN")].iloc[0]
    assert negated["question"] == "Is this chest X-ray abnormal? Answer only 1 for yes or 0 for no."
    assert negated["positive_text"] == "This chest X-ray is abnormal." and negated["label"] == "positive"
    experiment = load_experiment(paper_folder / "arms" / "negation" / "experiment.yaml")
    assert experiment.task.negated == "negated" and experiment.task.answer_sets["numeric_10"].yes == ["1"]


def test_arms_that_compare_with_the_main_question_take_it_for_their_samples(paper_folder):
    plans.write_arms(paper_folder, paper_folder)
    script = table_of(paper_folder, "transliteration")
    assert set(script["language"]) == {"HI"} and set(script["script"]) == {"native", "romanized"}
    asked = table_of(paper_folder, "permutation")
    permuted = asked[asked["term_condition"] == "permuted"].iloc[0]
    assert permuted["shown_concept"] != permuted["concept"] and BANK["concepts"][permuted["shown_concept"]]["EN"] in "".join(
        asked.loc[(asked["sample"] == permuted["sample"]) & (asked["term_condition"] == "permuted"), "question"])
    assert "language" not in table_of(paper_folder, "factorial").columns


def test_a_plan_that_was_not_written_from_this_bank_stops_everything(paper_folder):
    core = pd.read_csv(paper_folder / "eval_plan_core.csv", dtype=str)
    core.loc[0, "prompt"] = "Finding: nodule. Is this finding present?"
    core.to_csv(paper_folder / "eval_plan_core.csv", index=False)
    with pytest.raises(VLMShiftError, match="The plan and the bank disagree on nodule in EN"):
        plans.write_arms(paper_folder, paper_folder)
    assert not (paper_folder / "arms").exists()


def test_missing_plans_say_where_to_find_them(tmp_path):
    with pytest.raises(VLMShiftError, match="eval_plan_core.csv not found. The evaluation plans of the paper are on the drive"):
        plans.write_arms(tmp_path, tmp_path)


def test_without_the_supplementary_plan_only_its_arms_are_missing(paper_folder):
    (paper_folder / "eval_plan_core_supp.csv").unlink()
    assert list(plans.write_arms(paper_folder, paper_folder)) == ["main", "formats", "negation", "controls", "quality"]
    with pytest.raises(VLMShiftError, match="Unknown arm"):
        plans.write_arms(paper_folder, paper_folder, ["dragon"])


def test_image_folders_of_the_study_are_replaced_by_the_local_ones():
    roots = {"/content/images": "D:/all", "/content/images/NIH-ChestXray14": "E:/nih"}
    found = plans.local_images(pd.Series(["/content/images/NIH-ChestXray14/a.png", "/content/images/x/b.png", "/other/c.png"]), roots)
    assert [Path(path).as_posix() for path in found] == ["E:/nih/a.png", "D:/all/x/b.png", "/other/c.png"]


# --- replay of recorded answers ---------------------------------------------------------------------------------

def recorded_answers(plan: pd.DataFrame, model: str, right) -> pd.DataFrame:
    """The prediction file of the study for one model: right where `right(row)` says so, as 1.0 or 0.0."""
    truth = plan["y_true"].astype(int)
    correct = pd.Series([right(row) for row in plan.to_dict("records")])
    said = truth.where(correct, 1 - truth).astype(float)
    return plan.assign(model=model, raw_response=said.map({1.0: "1", 0.0: "0"}), pred=said.astype(str), score_yes=None,
                       score_no=None, valid="1", error=None)


def test_replaying_recorded_answers_gives_the_effects_of_the_paper_design(paper_folder):
    """A model wrong whenever the sentence is in Hindi: a frame effect of -1 for Hindi, none for French,
    and no term effect. No model is called and no image is read."""
    plans.write_arms(paper_folder, paper_folder, ["factorial"])
    supp = pd.read_csv(paper_folder / "eval_plan_core_supp.csv", dtype=str)
    (paper_folder / "preds").mkdir()
    recorded_answers(supp, "llava_med", lambda row: row["frame_language"] != "HI").to_csv(
        paper_folder / "preds" / "pred__llava_med__core_supp.csv", index=False)
    run = replay.replay(paper_folder / "arms" / "factorial", paper_folder / "preds", {})
    results = pd.read_parquet(run / "results.parquet")
    assert set(results["model"]) == {"llava-med"} and len(results) == 4 * 7 and results["status"].eq("valid").all()

    from vlmshift.cli import do_analyze

    analysis = load_analysis(paper_folder / "arms" / "factorial" / "analysis.yaml", str(run))
    analysis.stats.min_units = 2
    report = do_analyze(analysis)
    found = pd.read_parquet(report.parent / "analyses.parquet").set_index(["level_b", "statistic"])["value"]
    hindi, french = "frame_language=HI, term_language=HI", "frame_language=FR, term_language=FR"
    assert found[hindi, "main_frame_language"] == -1 and found[hindi, "main_term_language"] == 0
    assert found[hindi, "interaction"] == 0 and found[french, "main_frame_language"] == 0
    written = (report / "report.md").read_text(encoding="utf-8")
    assert "the images were not read" in pd.read_csv(report / "issues.csv")["message"].str.cat()
    assert "llava-med" in written and "**Synthetic run**" not in written


def test_an_answer_the_study_could_not_read_is_invalid_and_a_failure_is_an_error(paper_folder):
    plans.write_arms(paper_folder, paper_folder, ["main"])
    core = pd.read_csv(paper_folder / "eval_plan_core.csv", dtype=str)
    answers = recorded_answers(core, "chexone", lambda row: True)
    first, second = core.index[(core["arm"] == "main") & (core["experiment"] == "core")][:2]
    answers.loc[first, ["pred", "valid", "raw_response"]] = [None, "0", "The image shows"]
    answers.loc[second, ["pred", "valid", "error"]] = [None, "0", "CUDA out of memory"]
    (paper_folder / "preds").mkdir()
    answers.to_csv(paper_folder / "preds" / "pred__chexone__core.csv", index=False)
    run = replay.replay(paper_folder / "arms" / "main", paper_folder / "preds", {})
    results = pd.read_parquet(run / "results.parquet")
    assert results["status"].value_counts().to_dict() == {"valid": 10, "INVALID": 1, "error": 1}
    assert results.loc[results["status"] == "INVALID", "decision"].isna().all()


def test_replay_without_prediction_files_says_where_they_are(paper_folder):
    plans.write_arms(paper_folder, paper_folder, ["main"])
    (paper_folder / "preds").mkdir()
    with pytest.raises(VLMShiftError, match="No prediction file"):
        replay.replay(paper_folder / "arms" / "main", paper_folder / "preds", {})


# --- audits of the paper ----------------------------------------------------------------------------------------

def test_congruence_reads_the_shift_to_the_language_of_the_country():
    rows = [dict(measure="language_by_country", mode="valid", metric=metric, model="m", split=f"country={country}",
                 level_a="EN", level_b=language, value=value, ci_low=value - 0.1, ci_high=value + 0.1, significant=False,
                 n_units=20, n_samples=20)
            for country, language, value in (("Spain", "ES", 0.05), ("Spain", "FR", -0.2), ("USA", "ES", 0.3), ("China", "ZH", -0.1))
            for metric in ("delta_acc", "chg", "delta_sens")]
    found = audits.congruence(pd.DataFrame(rows))
    assert found[["country", "local_language"]].drop_duplicates().to_dict("records") == [
        {"country": "China", "local_language": "ZH"}, {"country": "Spain", "local_language": "ES"}]
    assert set(found["metric"]) == {"delta_acc", "chg"}  # Spanish in Spain, never French in Spain or Spanish in the USA
    assert found.loc[(found["country"] == "Spain") & (found["metric"] == "delta_acc"), "value"].item() == 0.05
    with pytest.raises(VLMShiftError, match="No measure language_by_country"):
        audits.congruence(pd.DataFrame(rows).assign(measure="language"))


def test_semantic_audit_sees_report_details_in_a_translated_column():
    workbook = pd.DataFrame({
        "dataset": ["PadChest-GR"] * 3 + ["NIH-ChestXray14"],
        "radlex_libelle": ["nodule", "nodule; scoliosis", "nodule", "mass"],
        "text_EN": ["nodule", "nodule", "nodule", "mass"],
        "text_FR": ["nodule du lobe supérieur droit", "nodule de 8 mm", "nodule", "masse"],
    })
    found = audits.semantic_audit(workbook).set_index(["site", "concept"])
    nodule = found.loc[("PadChest-GR", "nodule")]
    assert (nodule["n_rows"], nodule["unique_EN"], nodule["unique_FR"], nodule["ratio_FR_vs_EN"]) == (3, 1, 3, 3)
    assert found.loc[("PadChest-GR", "scoliosis"), "n_rows"] == 1 and found.loc[("NIH-ChestXray14", "mass"), "ratio_FR_vs_EN"] == 1
    assert found.loc[("PadChest-GR", "nodule"), "unique_ZH"] == 0  # a language column the workbook does not have


def test_the_prepared_tables_say_where_each_source_was_acquired(tmp_path):
    from prepare import common

    table = pd.DataFrame([{"image": str(tmp_path / "a.png"), "patient": "PadChest_1", "site": "PadChest", "concept": "nodule", "label": "positive"}])
    written = pd.read_parquet(common.write_source("s", table, tmp_path / "tables"))
    assert set(written["country"]) == {"Spain"} and set(common.COUNTRIES.values()) >= {"USA", "China", "Japan", "Nigeria"}
