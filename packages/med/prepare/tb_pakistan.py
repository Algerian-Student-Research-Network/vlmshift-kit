"""Tuberculosis Chest X-rays (Pakistan): 2,494 tuberculosis and 514 normal images of a local hospital.

No patient number: each image is its own patient (a limit, docs/decisions.md).
"""
from pathlib import Path

import pandas as pd

from label_rules.tb_pakistan import CLASSES, labels

from .common import images_by_class, label_rows

SITE = "Pakistan TB"
RAW = "tb_pakistan"  # a folder: the class of an image is the name of its folder

SOURCE = {
    "name": "Dataset of Tuberculosis Chest X-rays Images, version 3 (Mendeley Data)",
    "page": "https://data.mendeley.com/datasets/8j2g3csprk/3",
    "access": "open",
    "steps": "Download all the files, then put the two folders Normal Chest X-rays and TB Chest X-rays in this folder.",
    "license": "CC BY 4.0",
    # Of the folder (check.py): computed from the SHA-256 that Mendeley publishes for each of the 3,008 files.
    "sha256": "a1a8b8d9ab23fbf607b3063827e8a0ca7e8384eb6a84b5b44aeb1776c697bb84",
    "note": "26 pairs of byte-identical files, each within one class.",
    "bibtex": """@misc{kiran2026tuberculosis,
  author = {Kiran, Saira and Jabeen, Ishrat},
  title = {Dataset of Tuberculosis Chest X-rays Images},
  publisher = {Mendeley Data},
  version = {3},
  year = {2026},
  doi = {10.17632/8j2g3csprk.3}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    images = images_by_class(root / RAW, CLASSES)
    return label_rows(images.assign(patient=images["name"]), SITE, images["class"].map(labels))
