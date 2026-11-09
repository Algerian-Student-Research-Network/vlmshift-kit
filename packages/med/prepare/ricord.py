"""RSNA-RICORD-1c (international): 1,257 DICOM images of 361 patients with a positive COVID-19 test.

The labels are MD.ai annotations, per study. The DICOM images are written once as PNG under
converted/ricord, the format every model reads; only those with a positive or negative label.
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from label_rules.ricord import labels

from .common import LayoutError, dicom_to_png, label_rows

SITE = "RICORD"
RAW = "ricord/midrc-ricord-1c-da-other-json.zip"
LIBRARIES = ["pydicom"]  # read by check.py

SOURCE = {
    "name": "MIDRC-RICORD-1C, version 1 (The Cancer Imaging Archive, 2021)",
    "page": "https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/",
    "access": "open",
    "steps": "Download the images with the NBIA Data Retriever and the manifest MIDRC-RICORD-1c-manifest-Jan-2021.tcia, "
             "into this folder, then the annotations midrc-ricord-1c-da-other-json.zip, kept as it is.",
    "license": "CC BY-NC 4.0, and the TCIA data usage policy (cite the data and the paper)",
    "sha256": None,
    "bibtex": """@misc{tsai2021ricord1c,
  author = {Tsai, E. and Simpson, S. and Lungren, M. P. and Hershman, M. and Roshkovan, L. and Colak, E. and Erickson, B. J. and Shih, G. and Stein, A. and Kalpathy-Cramer, J. and Shen, J. and Hafez, M. A. F. and John, S. and Rajiah, P. and Pogatchnik, B. P. and Mongan, J. T. and Altinmakas, E. and Ranschaert, E. and Kitamura, F. C. and Topff, L. and Moy, L. and Kanne, J. P. and Wu, C.},
  title = {Data from Medical Imaging Data Resource Center (MIDRC) - RSNA International COVID Radiology Database (RICORD) Release 1c - Chest x-ray, Covid+ (MIDRC-RICORD-1C)},
  publisher = {The Cancer Imaging Archive},
  version = {1},
  year = {2021},
  doi = {10.7937/91ah-v663}
}
@article{tsai2021ricord,
  author = {Tsai, Emily B. and Simpson, Scott and Lungren, Matthew P. and others},
  title = {The RSNA International COVID-19 Open Radiology Database (RICORD)},
  journal = {Radiology},
  volume = {299},
  number = {1},
  pages = {E204--E213},
  year = {2021},
  doi = {10.1148/radiol.2021203957}
}""",
}


def votes(export: dict) -> dict[str, list[str]]:
    """The classes the readers gave to each study, by StudyInstanceUID, from an MD.ai export."""
    names = {label["id"]: label["name"] for group in export.get("labelGroups", []) for label in group.get("labels", [])}
    found = defaultdict(list)
    for dataset in export.get("datasets", []):
        for annotation in dataset.get("annotations", []):
            if annotation.get("labelId") in names and annotation.get("StudyInstanceUID"):
                found[annotation["StudyInstanceUID"]].append(names[annotation["labelId"]])
    return found


def prepare(root: Path) -> pd.DataFrame:
    import pydicom

    with zipfile.ZipFile(root / RAW) as zipped:
        exports = [name for name in zipped.namelist() if name.endswith(".json")]
        if len(exports) != 1:
            raise LayoutError(f"{root / RAW}: expected one JSON file, found {exports}")
        studies = votes(json.loads(zipped.read(exports[0])))
    stated = {study: labels(given) for study, given in studies.items()}
    if not any(label["covid19"] != "unknown" for label in stated.values()):
        names = sorted({name for given in studies.values() for name in given})
        raise LayoutError(f"{root / RAW}: no study with a known class. Classes found: {names}")
    rows = []
    for path in sorted((root / "ricord").rglob("*.dcm")):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        label = stated.get(header.StudyInstanceUID, {"covid19": "unknown"})
        if label["covid19"] == "unknown":
            continue
        png = root / "converted" / "ricord" / f"{header.SOPInstanceUID}.png"
        if not png.exists():
            dicom_to_png(pydicom.dcmread(path), png)
        rows.append({"image": str(png), "patient": header.PatientID, "label": label})
    if not rows:
        raise LayoutError(f"no DICOM image of a labelled study under {root / 'ricord'}")
    images = pd.DataFrame(rows)
    return label_rows(images, SITE, images["label"])
