"""JSRT (Japan): 247 radiographs, 154 with a lung nodule and 93 without.

The free edition is two zips of DICOM images (Nodule154images, NonNodule93images). They are read
inside the zips, and written once as PNG under converted/jsrt. No patient number: each image is
its own patient (a limit, docs/decisions.md).
"""
import io
import zipfile
from pathlib import Path

import pandas as pd

from label_rules.jsrt import labels

from .common import LayoutError, dicom_to_png, label_rows

SITE = "JSRT"
RAW = "jsrt"  # a folder: the two zips, kept as they are
ARCHIVES = ["Nodule154images.zip", "NonNodule93images.zip"]
LIBRARIES = ["pydicom"]  # read by check.py

SOURCE = {
    "name": "JSRT database, DICOM edition (Japanese Society of Radiological Technology)",
    "page": "http://imgcom.jsrt.or.jp/download/",
    "access": "open",
    "steps": "Download Nodule154images.zip and NonNodule93images.zip and put them in this folder as they are.",
    "license": "No licence stated. The page asks to cite the paper below; ask JSRT before sharing the images",
    "sha256": None,
    "note": "The clinical file (CLNDAT) is for registered users only; it is not needed here.",
    "bibtex": """@article{shiraishi2000jsrt,
  author = {Shiraishi, Junji and Katsuragawa, Shigehiko and Ikezoe, Junpei and Matsumoto, Tsuneo and Kobayashi, Takeshi and Komatsu, Ken-ichi and Matsui, Mitate and Fujita, Hiroshi and Kodera, Yoshie and Doi, Kunio},
  title = {Development of a Digital Image Database for Chest Radiographs With and Without a Lung Nodule: Receiver Operating Characteristic Analysis of Radiologists' Detection of Pulmonary Nodules},
  journal = {American Journal of Roentgenology},
  volume = {174},
  number = {1},
  pages = {71--74},
  year = {2000},
  doi = {10.2214/ajr.174.1.1740071}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    import pydicom

    rows = []
    for archive in ARCHIVES:
        if not (root / RAW / archive).exists():
            raise LayoutError(f"missing {root / RAW / archive}")
        with zipfile.ZipFile(root / RAW / archive) as zipped:
            for name in sorted(zipped.namelist()):
                if not name.lower().endswith(".dcm"):
                    continue
                stem = Path(name).stem
                png = root / "converted" / "jsrt" / f"{stem}.png"
                if not png.exists():
                    dicom_to_png(pydicom.dcmread(io.BytesIO(zipped.read(name))), png)
                rows.append({"image": str(png), "patient": stem})
    images = pd.DataFrame(rows, columns=["image", "patient"])
    if set(images["patient"].str[:5]) != {"JPCLN", "JPCNN"}:
        raise LayoutError(f"expected images JPCLN### and JPCNN### in {ARCHIVES}, found {sorted(images['patient'])[:5]}")
    return label_rows(images, SITE, images["patient"].map(labels))
