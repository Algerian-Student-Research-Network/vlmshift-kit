"""Med package: sources, label rules, concept bank, check, citations, tables of the paper, chaining with Core."""
import hashlib
import inspect
import json
import re
import shutil
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import FIXTURES, ROOT
from vlmshift.cli import do_run
from vlmshift.config.load import load_experiment, read_yaml, write_yaml
from vlmshift.extensions import get_adapter
from vlmshift.inference.base import ModelError

MED = ROOT / "packages" / "med"
sys.path.insert(0, str(MED))

import adapters  # noqa: E402,F401  (registers chexzero and open_clip)
import check  # noqa: E402
import cite  # noqa: E402
import paper  # noqa: E402
from label_rules import nigeria as nigeria_rule  # noqa: E402
from label_rules import nih as nih_rule  # noqa: E402
from label_rules import padchest_gr as padchest_rule  # noqa: E402
from label_rules import ranzcr as ranzcr_rule  # noqa: E402
from label_rules import ricord as ricord_rule  # noqa: E402
from label_rules.template import to_label  # noqa: E402
from prepare import card_problems, common, folder_of, name_of, sources, template  # noqa: E402

BANK = read_yaml(MED / "configs" / "questions.yaml")
LANGUAGES = ["EN", "FR", "ES", "DE", "AR", "HI", "ZH"]


def image(path, size=(8, 8)):
    """A tiny grey image, in the format of its suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 100).save(path)
    return path


def zipped(archive, files: dict):
    """A zip holding files: {name in the zip: path of an image or bytes}."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in files.items():
            handle.writestr(name, content)
    return archive


def png_bytes():
    path = FIXTURES / "toy" / "images" / "img1.png"
    return path.read_bytes()


# --- a fake raw folder for each source, laid out as the real one -----------------------------------------------

def raw_nih(root):
    folder = root / "nih"
    for name in ("00000001_000.png", "00000001_001.png", "00000002_000.png"):
        image(folder / "images_001" / "images" / name)
    pd.DataFrame({
        "Image Index": ["00000001_000.png", "00000001_001.png", "00000002_000.png"],
        "Finding Labels": ["Cardiomegaly|Effusion", "Cardiomegaly", "No Finding"],
        "Patient ID": ["1", "1", "2"],
    }).to_csv(folder / "Data_Entry_2017_v2020.csv", index=False)


def raw_ranzcr(root):
    folder = root / "ranzcr"
    rows = []
    for uid, patient, present in (("1.2.3", "p1", ["ETT - Normal", "NGT - Incompletely Imaged"]), ("1.2.4", "p2", [])):
        image(folder / "train" / f"{uid}.jpg")
        rows.append({"StudyInstanceUID": uid, **{c: int(c in present) for c in ranzcr_rule.COLUMNS}, "PatientID": patient})
    pd.DataFrame(rows).to_csv(folder / "train.csv", index=False)


def raw_padchest_gr(root):
    folder = root / "padchest_gr"
    reports = [
        {"ImageID": "a.png", "findings": [
            {"sentence_en": "Cardiomegaly.", "abnormal": True, "labels": ["cardiomegaly"]},
            {"sentence_en": "No pleural effusion.", "abnormal": False, "labels": ["pleural effusion"]},
        ]},
        {"ImageID": "b.png", "findings": [{"sentence_en": "Normal study.", "abnormal": False, "labels": ["normal"]}]},
    ]
    folder.mkdir(parents=True)
    (folder / "grounded_reports_20240819.json").write_text(json.dumps(reports), encoding="utf-8")
    pd.DataFrame({"ImageID": ["a.png", "b.png"], "PatientID": ["7", "8"], "split": ["train", "test"]}).to_csv(
        folder / "master_table.csv", index=False)
    for name in ("a.png", "b.png"):
        image(folder / "Padchest_GR_files" / name)


def raw_tb_pakistan(root):
    image(root / "tb_pakistan" / "TB Chest X-rays" / "TB.1.jpg")
    image(root / "tb_pakistan" / "Normal Chest X-rays" / "others (1).jpg")


def raw_pneumonia_bangladesh(root):
    top = "Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia"
    zipped(root / "pneumonia_bangladesh" / f"{top}.zip", {
        f"{top}/Training/pneumonia/pneumonia-1.jpg": png_bytes(),
        f"{top}/Testing/normal/normal-1.jpg": png_bytes(),
    })


def raw_nigeria(root):
    zipped(root / "nigeria" / "archive.zip", {
        f"my_dataset/train_folder/{cls}/{cls}-1.png": png_bytes() for cls in ("COVID", "NORMAL", "PNEUMONIA", "TB")
    })


def raw_ptx498(root):
    zipped(root / "ptx498" / "PTX-498-v2-fix.zip", {
        "SiteA/1.1.img.png": png_bytes(), "SiteA/1.2.mask.png": png_bytes(), "SiteB/1.1.img.png": png_bytes(),
    })


def dicom(path, patient, study, sop):
    """A 4 × 4 DICOM image, MONOCHROME1 (white for air): its first pixel is the darkest once shown."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    data = Dataset()
    data.file_meta = meta
    data.PatientID, data.StudyInstanceUID, data.SOPInstanceUID = patient, study, sop
    data.SOPClassUID = meta.MediaStorageSOPClassUID
    data.Rows = data.Columns = 4
    data.SamplesPerPixel, data.PhotometricInterpretation = 1, "MONOCHROME1"
    data.BitsAllocated, data.BitsStored, data.HighBit, data.PixelRepresentation = 16, 12, 11, 0
    data.PixelData = np.arange(16, dtype=np.uint16).tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(path, data, enforce_file_format=True)


def raw_ricord(root):
    folder = root / "ricord"
    export = {
        "labelGroups": [{"labels": [{"id": "L1", "name": "Typical Appearance"}, {"id": "L2", "name": "Negative for Pneumonia"},
                                    {"id": "L3", "name": "Atypical Appearance"}, {"id": "L4", "name": "Mild Opacities"}]}],
        "datasets": [{"annotations": [
            *({"labelId": "L1", "StudyInstanceUID": "s1"} for _ in range(2)), {"labelId": "L3", "StudyInstanceUID": "s1"},
            {"labelId": "L4", "StudyInstanceUID": "s1"},
            *({"labelId": "L2", "StudyInstanceUID": "s2"} for _ in range(3)),
            {"labelId": "L1", "StudyInstanceUID": "s3"}, {"labelId": "L3", "StudyInstanceUID": "s3"},  # a tie
        ]}],
    }
    zipped(folder / "midrc-ricord-1c-da-other-json.zip", {"MIDRC-RICORD-1c.json": json.dumps(export)})
    for patient, study in (("p1", "s1"), ("p2", "s2"), ("p3", "s3")):
        dicom(folder / "MIDRC-RICORD-1C" / patient / study / "1-1.dcm", patient, study, f"{study}.1")


def raw_shenzhen(root):
    image(root / "shenzhen" / "CXR_png" / "CHNCXR_0001_0.png")
    image(root / "shenzhen" / "CXR_png" / "CHNCXR_0327_1.png")
    image(root / "shenzhen" / "Annotations" / "masks" / "CHNCXR_0327_1.png")  # a mask: never read


def raw_montgomery(root):
    image(root / "montgomery" / "CXR_png" / "MCUCXR_0001_0.png")
    image(root / "montgomery" / "CXR_png" / "MCUCXR_0104_1.png")


def raw_jsrt(root):
    for archive, stem in (("Nodule154images", "JPCLN001"), ("NonNodule93images", "JPCNN001")):
        dicom(root / "dicom" / f"{stem}.dcm", "x", "s", stem)
        zipped(root / "jsrt" / f"{archive}.zip", {f"{archive}/{stem}.dcm": (root / "dicom" / f"{stem}.dcm").read_bytes()})


RAW = {name.removeprefix("raw_"): builder for name, builder in globals().items() if name.startswith("raw_")}


@pytest.fixture
def med_data(tmp_path, monkeypatch):
    monkeypatch.setenv("MED_DATA", str(tmp_path))
    return tmp_path


# --- sources --------------------------------------------------------------------------------------------------

def test_the_11_sources_of_the_paper_are_found():
    assert sorted(name_of(source) for source in sources()) == sorted([
        "jsrt", "montgomery", "nigeria", "nih", "padchest_gr", "pneumonia_bangladesh", "ptx498", "ranzcr",
        "ricord", "shenzhen", "tb_pakistan",
    ])
    assert template not in sources()


def test_each_source_has_a_fake_raw_folder_in_these_tests():
    assert sorted(RAW) == sorted(name_of(source) for source in sources())


def test_each_source_writes_the_table_of_the_experiment():
    config = read_yaml(MED / "configs" / "experiment.yaml")
    assert sorted(path.rsplit("/", 1)[1] for path in config["data"]) == sorted(f"{name_of(s)}.parquet" for s in sources())


@pytest.mark.parametrize("source", sources(), ids=name_of)
def test_each_source_prepares_from_its_raw_layout(source, med_data):
    RAW[name_of(source)](med_data)
    table = source.prepare(med_data)
    assert list(table.columns) == common.COLUMNS and len(table) > 0
    assert table["site"].nunique() == 1 and table["patient"].str.startswith(f"{table['site'].iloc[0]}_").all()
    assert set(table["label"]) <= {"positive", "negative", "unknown"}
    assert set(table["concept"]) <= set(BANK["concepts"])
    assert all(Image.open(path).size for path in table["image"])  # every image can be read
    written = pd.read_parquet(common.write_source(name_of(source), table, med_data / "tables"))
    assert sorted(written["language"].unique()) == sorted(LANGUAGES)
    assert set(written["label"]) <= {"positive", "negative"}


def labels_of(table):
    return {(patient.split("_", 1)[1], concept): label for patient, concept, label in
            table[["patient", "concept", "label"]].itertuples(index=False)}


def test_nih_keeps_the_patient_and_reads_no_finding_as_negative(med_data):
    from prepare import nih

    raw_nih(med_data)
    table = nih.prepare(med_data)
    assert table.groupby("image")["patient"].first().tolist() == ["NIH_1", "NIH_1", "NIH_2"]
    one = table[table["image"].str.endswith("00000001_000.png")].set_index("concept")["label"]
    assert one["cardiomegaly"] == one["effusion"] == "positive" and one["nodule"] == "unknown"
    assert table[table["patient"] == "NIH_2"]["label"].eq("negative").all()


def test_nih_stops_when_an_image_is_missing(med_data):
    from prepare import nih

    raw_nih(med_data)
    (med_data / "nih" / "images_001" / "images" / "00000002_000.png").unlink()
    with pytest.raises(common.LayoutError, match="1 images named in the labels are not under"):
        nih.prepare(med_data)


def test_ranzcr_reads_each_device(med_data):
    from prepare import ranzcr

    raw_ranzcr(med_data)
    found = labels_of(ranzcr.prepare(med_data))
    assert found[("p1", "endotracheal_tube")] == found[("p1", "nasogastric_tube")] == "positive"
    assert found[("p1", "central_venous_catheter")] == found[("p2", "endotracheal_tube")] == "negative"


def test_padchest_reads_positive_negative_and_normal_reports(med_data):
    from prepare import padchest_gr

    raw_padchest_gr(med_data)
    found = labels_of(padchest_gr.prepare(med_data))
    assert found[("7", "cardiomegaly")] == "positive" and found[("7", "effusion")] == "negative"
    assert ("7", "scoliosis") not in found  # not stated in an abnormal report: unknown, so absent
    assert found[("8", "scoliosis")] == found[("8", "cardiomegaly")] == "negative"  # a normal study


def test_padchest_says_which_columns_it_found(med_data):
    from prepare import padchest_gr

    raw_padchest_gr(med_data)
    pd.DataFrame({"ImageID": ["a.png"], "StudyID": ["1"]}).to_csv(med_data / "padchest_gr" / "master_table.csv", index=False)
    with pytest.raises(common.LayoutError, match=r"no column \['PatientID'\]. Its columns: \['ImageID', 'StudyID'\]"):
        padchest_gr.prepare(med_data)


def test_sources_without_patient_number_use_one_patient_per_image(med_data):
    from prepare import pneumonia_bangladesh

    raw_pneumonia_bangladesh(med_data)
    table = pneumonia_bangladesh.prepare(med_data)
    assert table["patient"].nunique() == len(table) == 2
    assert dict(zip(table["patient"].str.rsplit("/", n=1).str[1], table["label"])) == {
        "pneumonia-1.jpg": "positive", "normal-1.jpg": "negative"}


def test_an_archive_is_extracted_once(med_data):
    raw_ptx498(med_data)
    archive = med_data / "ptx498" / "PTX-498-v2-fix.zip"
    folder = common.unpacked(archive)
    marker = folder / "kept.txt"
    marker.write_text("x")
    assert common.unpacked(archive) == folder and marker.exists()
    assert not folder.with_name(folder.name + ".partial").exists()


def test_ptx498_keeps_the_images_not_the_masks(med_data):
    from prepare import ptx498

    raw_ptx498(med_data)
    table = ptx498.prepare(med_data)
    assert sorted(table["patient"]) == ["PTX-498_SiteA/1.1.img.png", "PTX-498_SiteB/1.1.img.png"]
    assert table["label"].eq("positive").all()


def test_nlm_sets_read_the_class_at_the_end_of_the_name_and_skip_the_masks(med_data):
    from prepare import montgomery, shenzhen

    raw_shenzhen(med_data)
    raw_montgomery(med_data)
    assert labels_of(shenzhen.prepare(med_data)) == {
        ("CHNCXR_0001_0.png", "tuberculosis"): "negative", ("CHNCXR_0327_1.png", "tuberculosis"): "positive"}
    assert labels_of(montgomery.prepare(med_data))[("MCUCXR_0104_1.png", "tuberculosis")] == "positive"


def test_jsrt_reads_the_dicom_inside_the_zips(med_data):
    from prepare import jsrt

    raw_jsrt(med_data)
    table = jsrt.prepare(med_data)
    assert labels_of(table) == {("JPCLN001", "nodule"): "positive", ("JPCNN001", "nodule"): "negative"}
    assert sorted(path.name for path in (med_data / "jsrt").iterdir()) == ["Nodule154images.zip", "NonNodule93images.zip"]


def test_a_class_folder_missing_is_named(tmp_path):
    image(tmp_path / "TB Chest X-rays" / "a.jpg")
    with pytest.raises(common.LayoutError, match=r"no image in a folder named \['Normal Chest X-rays'\]"):
        common.images_by_class(tmp_path, ["TB Chest X-rays", "Normal Chest X-rays"])


def test_ricord_takes_the_majority_and_converts_the_dicom(med_data):
    from prepare import ricord

    raw_ricord(med_data)
    table = ricord.prepare(med_data)
    assert labels_of(table) == {("p1", "covid19"): "positive", ("p2", "covid19"): "negative"}  # s3 is a tie
    pixels = np.asarray(Image.open(table["image"].iloc[0]))
    assert pixels.shape == (4, 4) and pixels[0, 0] == 255 and pixels[-1, -1] == 0  # MONOCHROME1 turned around


# --- label rules ---------------------------------------------------------------------------------------------

def test_label_rule_never_turns_doubt_into_negative():
    assert [to_label(value) for value in ("1", "0", "-1", "")] == ["positive", "negative", "unknown", "unknown"]


def test_nih_rule_does_not_read_absence_as_negative():
    assert nih_rule.labels("Mass")["mass"] == "positive" and nih_rule.labels("Mass")["hernia"] == "unknown"
    assert set(nih_rule.labels("No Finding").values()) == {"negative"}


def test_nigeria_rule_leaves_the_other_classes_unknown():
    assert nigeria_rule.labels("COVID") == {"covid19": "positive"}
    assert nigeria_rule.labels("NORMAL") == {"covid19": "negative", "pneumonia": "negative", "tuberculosis": "negative"}


def test_ricord_rule_majority_tie_and_invalid():
    assert ricord_rule.labels(["Typical Appearance", "Typical Appearance", "Atypical Appearance"]) == {"covid19": "positive"}
    assert ricord_rule.labels(["Typical Appearance", "Negative for Pneumonia"]) == {"covid19": "unknown"}
    assert ricord_rule.labels(["Invalid Study"]) == {"covid19": "unknown"}
    assert ricord_rule.labels(["Mild Opacities"]) == {"covid19": "unknown"}


def test_padchest_rule_positive_wins_over_negative():
    findings = [{"abnormal": False, "labels": ["Pleural Effusion"]}, {"abnormal": True, "labels": ["pleural effusion"]}]
    assert padchest_rule.labels(findings) == {"effusion": "positive"}


# --- concept bank ---------------------------------------------------------------------------------------------

def test_the_bank_has_35_concepts_in_7_languages():
    assert len(BANK["concepts"]) == 35
    assert list(BANK["templates"]) == LANGUAGES
    for concept, terms in BANK["concepts"].items():
        assert sorted(set(terms) - {"radlex"}) == sorted(LANGUAGES), concept
        assert all(str(terms[language]).strip() for language in LANGUAGES), concept


def test_every_template_holds_the_term_and_asks_for_an_english_answer():
    for language, texts in BANK["templates"].items():
        assert sorted(texts) == ["negative", "positive", "question"], language
        assert all("{term}" in text for text in texts.values()), language
        assert "YES" in texts["question"] and "NO" in texts["question"], language


def test_every_concept_a_label_rule_can_give_is_in_the_bank():
    given = {*nih_rule.FINDINGS.values(), *ranzcr_rule.DEVICES, *padchest_rule.CONCEPTS,
             *nigeria_rule.CONCEPTS.values(), "covid19", "pneumothorax", "pneumonia", "tuberculosis", "nodule"}
    assert given <= set(BANK["concepts"])


def test_radlex_identifiers_are_rids_or_left_empty():
    for concept, terms in BANK["concepts"].items():
        assert terms.get("radlex") is None or re.fullmatch(r"RID\d+( \+ RID\d+)*", terms["radlex"]), concept


def test_languages_only_change_the_words():
    table = pd.DataFrame([{"image": "a.png", "patient": "p", "site": "S", "concept": "effusion", "label": "positive"}])
    rows = common.add_languages(table, BANK)
    assert rows["language"].tolist() == LANGUAGES
    assert rows.iloc[0]["question"] == ("Finding: pleural effusion. Is this finding present on this chest X-ray? "
                                        "Answer only YES or NO.")
    assert rows.drop(columns=["language", "question", "positive_text", "negative_text"]).nunique().max() == 1


# --- sampling -------------------------------------------------------------------------------------------------

def sample_table(ill, healthy, images=1):
    rows = []
    for number in range(ill + healthy):
        for shot in range(images):
            rows.append({"image": f"p{number:02d}_{shot}.png", "patient": f"p{number:02d}", "site": "S",
                         "concept": "c", "label": "positive" if number < ill else "negative"})
    return pd.DataFrame(rows)


def test_sample_takes_half_ill_when_there_are_enough():
    chosen = common.sample_patients(sample_table(ill=10, healthy=10), n=6)
    assert chosen.groupby("label")["patient"].nunique().to_dict() == {"negative": 3, "positive": 3}


def test_sample_completes_with_the_other_group():
    assert common.sample_patients(sample_table(ill=2, healthy=10), n=6)["label"].value_counts().to_dict() == {
        "negative": 4, "positive": 2}
    assert common.sample_patients(sample_table(ill=10, healthy=0), n=6)["label"].eq("positive").sum() == 6


def test_sample_keeps_one_image_per_patient_and_is_repeatable():
    first = common.sample_patients(sample_table(ill=10, healthy=10, images=3), n=6)
    assert first["patient"].is_unique and first["image"].str.endswith("_0.png").all()
    assert first.equals(common.sample_patients(sample_table(ill=10, healthy=10, images=3), n=6))


def test_write_source_drops_unknown_before_sampling(med_data):
    table = pd.DataFrame([
        {"image": str(med_data / "a.png"), "patient": "S_1", "site": "S", "concept": "effusion", "label": "positive"},
        {"image": str(med_data / "a.png"), "patient": "S_1", "site": "S", "concept": "nodule", "label": "unknown"},
    ])
    written = pd.read_parquet(common.write_source("s", table, med_data / "tables"))
    assert set(written["concept"]) == {"effusion"} and len(written) == 7


def test_write_source_refuses_a_concept_outside_the_bank(med_data):
    table = pd.DataFrame([{"image": "a.png", "patient": "S_1", "site": "S", "concept": "dragon", "label": "positive"}])
    with pytest.raises(common.LayoutError, match="concepts missing from configs/questions.yaml: \\['dragon'\\]"):
        common.write_source("s", table, med_data / "tables")


# --- check.py ---------------------------------------------------------------------------------------------------

@pytest.fixture
def only_template(med_data, monkeypatch):
    """check.py on the template alone, with its raw file in place."""
    folder = med_data / "example_source"
    folder.mkdir()
    for number in (1, 2):
        shutil.copy(FIXTURES / "toy" / "images" / f"img{number}.png", folder)
    (folder / "labels.csv").write_bytes(  # bytes: the same file, so the same SHA-256, on every system
        b"path,patient_id,pneumothorax\nimg1.png,7,1\nimg2.png,7,-1\n"
    )
    monkeypatch.setattr(check, "sources", lambda: [template])
    return med_data


CONFIG = str(MED / "configs" / "experiment.yaml")


def test_check_says_where_to_get_a_missing_source(med_data):
    message = next(p for p in check.problems(CONFIG) if "Raw data missing" in p and "tb_pakistan" in p)
    from prepare import tb_pakistan

    assert tb_pakistan.SOURCE["page"] in message and f"put its files in {med_data / 'tb_pakistan'}" in message


def test_check_is_quiet_once_every_source_is_there(med_data):
    pytest.importorskip("pydicom")
    for build in RAW.values():
        build(med_data)
    assert not [p for p in check.problems(CONFIG) if "Raw data" in p]


def test_check_names_the_missing_chexzero_weights(med_data):
    assert any("chexzero: weights missing" in p and "rajpurkarlab/CheXzero" in p for p in check.problems(CONFIG))


def test_check_names_a_library_a_source_needs(med_data, monkeypatch):
    monkeypatch.setattr(check, "sources", lambda: [types.SimpleNamespace(
        __name__="prepare.new", RAW="new/a.csv", LIBRARIES=["no_such_library"], SOURCE=template.SOURCE)])
    assert "new: missing library: no_such_library" in check.problems(CONFIG)


def test_check_sees_another_version_of_a_raw_file(only_template):
    assert check.other_versions() == []  # the file of the card
    labels = only_template / "example_source" / "labels.csv"
    labels.write_bytes(b"path,patient_id,pneumothorax\nimg1.png,7,1\n")
    found = check.other_versions()
    assert len(found) == 1 and check.fingerprint(labels) in found[0]


def test_check_gives_the_fingerprint_of_a_card_without_one(only_template, monkeypatch):
    monkeypatch.setitem(template.SOURCE, "sha256", None)
    labels = only_template / "example_source" / "labels.csv"
    assert check.other_versions() == [f"template: its card has no SHA-256 yet. Write this one in it: {check.fingerprint(labels)}"]


def test_fingerprint_of_a_folder_ignores_where_it_was_extracted(tmp_path):
    for top in ("one", "two/deeper"):
        (tmp_path / top / "TB Chest X-rays").mkdir(parents=True)
        (tmp_path / top / "TB Chest X-rays" / "TB.1.jpg").write_bytes(b"1")
    line = f"TB Chest X-rays/TB.1.jpg {hashlib.sha256(b'1').hexdigest()}"
    expected = hashlib.sha256(line.encode()).hexdigest()  # what the card of tb_pakistan was computed with
    assert check.fingerprint(tmp_path / "one") == check.fingerprint(tmp_path / "two") == expected


# --- cards and citations ----------------------------------------------------------------------------------------

def test_every_source_has_a_complete_card():
    assert [problem for source in [*sources(), template] for problem in card_problems(source)] == []


def test_an_incomplete_card_is_refused():
    def card(**change):
        return types.SimpleNamespace(__name__="prepare.new", SOURCE={**template.SOURCE, **change})

    assert card_problems(card(license="", access="free")) == [
        "new: SOURCE has no license",
        "new: SOURCE access must be one of ['open', 'registration', 'agreement', 'credentialed'], not 'free'",
    ]
    assert card_problems(card(sha256="to fill in")) == [
        "new: SOURCE sha256 is not a SHA-256. check.py gives it once the file is there",
    ]
    assert card_problems(card(licence="CC BY")) == ["new: SOURCE has an unknown field licence"]
    assert card_problems(card(sha256=None, note=None)) == []  # both optional


def test_a_raw_folder_is_its_own_folder():
    from prepare import nih, tb_pakistan

    assert folder_of(tb_pakistan).as_posix() == "tb_pakistan" and folder_of(nih).as_posix() == "nih"


def test_sources_page_and_bibliography_are_up_to_date():
    for path, text in cite.texts().items():
        assert path.read_text(encoding="utf-8") == text, f"{path.name} is out of date: run python cite.py in packages/med"


def test_every_citation_has_its_own_key():
    keys = [line.split("{", 1)[1].rstrip(",") for line in (MED / "references.bib").read_text(encoding="utf-8").splitlines()
            if line.startswith("@")]
    assert len(keys) == len(set(keys)) >= 11


def test_med_data_never_goes_to_git():
    assert "packages/med/data/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# --- adapters ---------------------------------------------------------------------------------------------------

def test_med_adapters_are_registered_as_contrastive():
    assert get_adapter("chexzero").kind == get_adapter("open_clip").kind == "contrastive"


def test_open_clip_cannot_choose_a_version():
    with pytest.raises(ModelError, match="revision: open_clip cannot choose a version"):
        get_adapter("open_clip")("m", "hf-hub:x/y", "abc123", {}).load()


def test_chexzero_reads_a_plain_state_dict(tmp_path):
    """The official weights are a state dict of OpenAI CLIP ViT-B/32: here a random one, same shape."""
    open_clip = pytest.importorskip("open_clip")
    torch = pytest.importorskip("torch")
    from adapters.chexzero import ARCHITECTURE, MEAN

    torch.save(open_clip.create_model(ARCHITECTURE).state_dict(), tmp_path / "weights.pt")
    model = get_adapter("chexzero")("chexzero", str(tmp_path / "weights.pt"), None, {})
    model.load()
    wide = image(tmp_path / "wide.png", size=(64, 32))
    pixels = model.preprocess(Image.open(wide).convert("RGB"))
    assert tuple(pixels.shape) == (3, 224, 224)
    assert pixels[0, 0, 0] < pixels[0, 112, 112]  # padded to a square, not cut: the corner is the fill
    assert abs(model.preprocess.transforms[-1].mean[0] - MEAN) < 1e-9
    scores = model.score([str(wide)], ["effusion"], ["no effusion"])
    assert len(scores) == 1 and len(scores[0]) == 2


# --- tables of the paper ----------------------------------------------------------------------------------------

def paper_lines():
    """2 models × 2 languages. Cell A: both classes. Cell B: positives only. Cell C: m2 has no FR answer."""
    rows = []
    for model in ("m1", "m2"):
        for language in ("EN", "FR"):
            both = [("positive", "yes", "valid"), ("negative", "no", "valid"), ("negative", None, "INVALID"),
                    ("positive", "no", "valid")]
            if model == "m2" and language == "FR":
                both = [(label, "yes", "valid") for label, _, _ in both]
            rows += [dict(model=model, site="A", concept="pneumonia", language=language, label=label,
                          decision=decision, status=status) for label, decision, status in both]
            rows.append(dict(model=model, site="B", concept="pneumothorax", language=language, label="positive",
                             decision="yes", status="valid"))
            if not (model == "m2" and language == "FR"):
                rows.append(dict(model=model, site="C", concept="tuberculosis", language=language, label="negative",
                                 decision="no", status="valid"))
    return pd.DataFrame(rows)


def test_cell_scores_count_invalid_as_wrong():
    scores = paper.cell_scores(paper_lines()).set_index(["model", "site", "concept", "language"])
    cell = scores.loc[("m1", "A", "pneumonia", "EN")]
    assert (cell["n"], cell["acc"], cell["bacc"], cell["invalid_rate"]) == (4, 0.5, 0.5, 0.25)
    assert cell["f1"] == pytest.approx(2 / 3)  # 2 TP / (2 positives + 1 yes)
    assert np.isnan(scores.loc[("m1", "B", "pneumothorax", "EN"), "bacc"])  # no negative


def test_complete_case_rule_and_balanced_core():
    scores = paper.cell_scores(paper_lines())
    assert sorted(paper.complete(scores, "acc")["site"].unique()) == ["A", "B"]  # C lacks m2 in FR
    assert sorted(paper.complete(scores, "bacc")["site"].unique()) == ["A"]  # B has no negative


def test_language_gap_is_zero_for_english_and_paired():
    found = paper.languages_table(paper.cell_scores(paper_lines())).set_index("language")
    assert found.loc["EN", "delta_acc"] == 0 and found.loc["EN", "delta_acc_low"] == 0
    # FR minus EN, over the model × cell pairs of the core: m1 0, 0; m2 0 on B, 0.5 - 0.5 on A.
    assert found.loc["FR", "delta_acc"] == pytest.approx(0)
    assert found.loc["FR", "delta_acc_low"] <= found.loc["FR", "delta_acc"] <= found.loc["FR", "delta_acc_high"]


def test_rankings_compare_the_order_of_the_models():
    scores = pd.DataFrame({"model": ["m1", "m2", "m3"] * 2, "site": "S", "concept": "c",
                           "language": ["EN"] * 3 + ["FR"] * 3, "acc": [0.9, 0.5, 0.1, 0.8, 0.6, 0.2]})
    assert paper.rankings(scores)["tau"].tolist() == [pytest.approx(1.0)]


# --- chaining with Core -----------------------------------------------------------------------------------------

def test_a_prepared_source_runs_through_core_and_gives_the_tables(med_data, tmp_path):
    """A prepared source goes through Core without adaptation (fake models, no GPU), then paper.py reads the run."""
    from prepare import tb_pakistan

    image(med_data / "tb_pakistan" / "TB Chest X-rays" / "img1.jpg")  # named as the fake models expect
    image(med_data / "tb_pakistan" / "Normal Chest X-rays" / "img2.jpg")
    common.write_source("tb_pakistan", tb_pakistan.prepare(med_data), med_data / "tables")
    write_yaml({
        "schema_version": 1,
        "data": str(med_data / "tables" / "tb_pakistan.parquet"),
        "roles": {"unit": "patient", "positive_text": "positive_text", "negative_text": "negative_text"},
        "axes": {"language": {"type": "controlled"}, "site": {"type": "observed"}, "concept": {"type": "observed"}},
        "task": {"type": "yes_no"},
        "models": [{"name": name, "adapter": "fake_generative", "params": {"faults": False}} for name in ("m1", "m2")],
    }, tmp_path / "experiment.yaml")
    run = do_run(load_experiment(tmp_path / "experiment.yaml"))
    assert len(pd.read_parquet(run / "results.parquet")) == 2 * 2 * 7  # 2 images, 2 models, 7 languages
    folder = paper.write_tables(run)
    assert pd.read_csv(folder / "matched_cells.csv").to_dict("records") == [{"site": "Pakistan TB", "concept": "tuberculosis"}]
    assert sorted(pd.read_csv(folder / "languages.csv")["language"]) == sorted(LANGUAGES)


def test_prepared_tables_go_where_the_experiment_reads_them():
    folder = inspect.signature(common.write_source).parameters["folder"].default
    paths = read_yaml(MED / "configs" / "experiment.yaml")["data"]
    assert {(MED / "configs" / path).parent.resolve() for path in paths} == {folder.resolve()}
