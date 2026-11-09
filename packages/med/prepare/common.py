"""Tools shared by the preparation scripts: raw data → VLMShift table."""
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from vlmshift.common import write_parquet
from vlmshift.config.load import read_yaml

HERE = Path(__file__).parents[1]
PATIENTS, SEED = 300, 0  # the evaluation core of the paper: 300 patients per source
IMAGES = (".png", ".jpg", ".jpeg")
COLUMNS = ["image", "patient", "site", "concept", "label"]


class LayoutError(Exception):
    """The raw files are not laid out as the script expects. The message says what was found."""


def data_root() -> Path:
    """Folder of the raw data: MED_DATA variable, otherwise packages/med/data."""
    return Path(os.environ.get("MED_DATA", HERE / "data"))


def unpacked(archive: Path) -> Path:
    """The folder of a zip, extracted once next to it: the user only puts the archive in place.

    The extraction goes to a temporary folder first, so an interrupted one is started again.
    """
    folder = archive.parent / archive.name.removesuffix(".zip")
    if not folder.exists():
        partial = folder.with_name(folder.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(partial)
        partial.rename(folder)
    return folder


def images_by_class(folder: Path, classes) -> pd.DataFrame:
    """The images of a source whose class is the name of their folder, wherever that folder is.

    Columns: image (its path), class (as written in classes) and name (its path in the source, the
    identifier of an image when the source gives no patient).
    """
    wanted = {name.lower(): name for name in classes}
    rows = [{"image": str(path), "class": wanted[path.parent.name.lower()], "name": path.relative_to(folder).as_posix()}
            for path in sorted(folder.rglob("*"))
            if path.suffix.lower() in IMAGES and path.parent.name.lower() in wanted]
    if missing := sorted(set(wanted.values()) - {row["class"] for row in rows}):
        raise LayoutError(f"no image in a folder named {missing} under {folder}")
    return pd.DataFrame(rows)


def dicom_to_png(dicom, path: Path) -> None:
    """A DICOM image (read by pydicom) as an 8-bit PNG, the format every model reads: its window
    (VOI LUT) applied, white for bone, the full range used. Written once: the PNG is kept."""
    try:
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom 2
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    pixels = apply_voi_lut(dicom.pixel_array, dicom).astype(float)
    if dicom.get("PhotometricInterpretation") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    pixels = (pixels - pixels.min()) / max(pixels.max() - pixels.min(), 1e-9) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(pixels).astype(np.uint8)).save(path)


def locate(names, folder: Path) -> list[str]:
    """The path of each image a label file names, wherever it is under folder."""
    paths = {path.name: str(path) for path in folder.rglob("*") if path.suffix.lower() in IMAGES}
    if missing := [name for name in names if name not in paths]:
        raise LayoutError(f"{len(missing)} images named in the labels are not under {folder}, such as {missing[0]}")
    return [paths[name] for name in names]


def require(table: pd.DataFrame, columns, where) -> None:
    """Stop with the columns found when the expected ones are not there: another version of the source."""
    if missing := [column for column in columns if column not in table.columns]:
        raise LayoutError(f"{where} has no column {missing}. Its columns: {list(table.columns)}")


def label_rows(images: pd.DataFrame, site: str, labels) -> pd.DataFrame:
    """One row per image and concept its source states, from the labels of each image: {concept: label}.

    images has the columns image and patient. The patient gets the site as prefix: the same number
    can exist in two sources.
    """
    found = [{"image": image, "patient": f"{site}_{patient}", "site": site, "concept": concept, "label": label}
             for image, patient, stated in zip(images["image"], images["patient"], labels)
             for concept, label in stated.items()]
    return pd.DataFrame(found, columns=COLUMNS)


def add_languages(table: pd.DataFrame, bank: dict) -> pd.DataFrame:
    """One row per language. Each language is a variant of the same image and concept (section 3).

    The wording is the same for every concept: only the term changes, so language is the only
    thing that varies between the rows of one image and one concept.
    """
    rows = []
    for row in table.to_dict("records"):
        terms = bank["concepts"][row["concept"]]
        for language, texts in bank["templates"].items():
            filled = {key: text.format(term=terms[language]) for key, text in texts.items()}
            rows.append({**row, "language": language, "question": filled["question"],
                         "positive_text": filled["positive"], "negative_text": filled["negative"]})
    return pd.DataFrame(rows)


def sample_patients(table: pd.DataFrame, n: int = PATIENTS, seed: int = SEED) -> pd.DataFrame:
    """n patients, half with a positive label when there are enough. Each keeps one image, the
    first by path: the patients stay independent, and no patient weighs more than another."""
    table = table[table["image"] == table.groupby("patient")["image"].transform("min")]
    sick = table.groupby("patient")["label"].agg(lambda labels: labels.eq("positive").any())
    ill, healthy = sick.index[sick], sick.index[~sick]
    take_ill = min(len(ill), max(n // 2, n - len(healthy)))
    rng = np.random.default_rng(seed)
    chosen = [*rng.choice(ill, take_ill, replace=False),
              *rng.choice(healthy, min(len(healthy), n - take_ill), replace=False)]
    return table[table["patient"].isin(chosen)]


def relative(path: str, folder: Path) -> str:
    """Relative path when possible: the dataset and the package can then move together."""
    try:
        return os.path.relpath(path, folder)
    except ValueError:  # another drive, on Windows
        return str(Path(path).resolve())


def write_source(name: str, table: pd.DataFrame, folder: Path = HERE / "data" / "tables") -> Path:
    """Write the table of a source where configs/experiment.yaml reads it, whatever MED_DATA is.

    Only the positives and negatives the source states are kept: unknown is never asked (section 4.2).
    """
    bank = read_yaml(HERE / "configs" / "questions.yaml")
    if unknown := sorted(set(table["concept"]) - set(bank["concepts"])):
        raise LayoutError(f"{name}: concepts missing from configs/questions.yaml: {unknown}")
    output = folder / f"{name}.parquet"
    table = sample_patients(table[table["label"].isin(["positive", "negative"])])
    table = table.assign(image=[relative(path, output.parent) for path in table["image"]])
    write_parquet(add_languages(table, bank), output, name)
    return output
