"""RANZCR CLiP (Australia/New Zealand): 30,083 labelled images, presence and position of 4 devices."""
from pathlib import Path

import pandas as pd

from label_rules.ranzcr import COLUMNS, labels

from .common import label_rows, locate, require

SITE = "RANZCR"
RAW = "ranzcr/train.csv"

SOURCE = {
    "name": "RANZCR CLiP - Catheter and Line Position Challenge (Kaggle, 2021), train set",
    "page": "https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data",
    "access": "registration",
    "steps": "Sign in to Kaggle and accept the rules of the competition, then download train.csv and the train/ folder. "
             "The test set has no labels.",
    "license": "Rules of the competition: non-commercial use only, no redistribution",
    "sha256": None,
    "note": "Its images come from NIH ChestX-ray14, under other names: one image can be in both sources.",
    "bibtex": """@article{tang2021clip,
  author = {Tang, Jennifer S. N. and Seah, Jarrel C. Y. and Zia, Adil and others},
  title = {CLiP, catheter and line position dataset},
  journal = {Scientific Data},
  volume = {8},
  pages = {285},
  year = {2021},
  doi = {10.1038/s41597-021-01066-8}
}
@misc{ranzcr2020clip,
  author = {Seah, Jarrel and Jen and Maggie and Law, Meng and Culliton, Phil and Dowd, Sarah},
  title = {RANZCR CLiP - Catheter and Line Position Challenge},
  year = {2020},
  howpublished = {Kaggle, \\url{https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification}}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / RAW, dtype=str, keep_default_na=False)
    require(raw, ["StudyInstanceUID", "PatientID", *COLUMNS], root / RAW)
    images = pd.DataFrame({"image": locate(raw["StudyInstanceUID"] + ".jpg", root / "ranzcr"),
                           "patient": raw["PatientID"]})
    return label_rows(images, SITE, [labels(row) for row in raw.to_dict("records")])
