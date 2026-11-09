"""Nigeria Chest X-ray: 2,600 images in four classes of 650 (COVID, NORMAL, PNEUMONIA, TB).

No patient number: each image is its own patient (a limit, docs/decisions.md).
"""
from pathlib import Path

import pandas as pd

from label_rules.nigeria import CLASSES, labels

from .common import images_by_class, label_rows, unpacked

SITE = "Nigeria"
RAW = "nigeria/archive.zip"

SOURCE = {
    "name": "Nigeria Chest X-ray Dataset, version 1 (Kaggle, 2024)",
    "page": "https://www.kaggle.com/datasets/aminumusa/nigeria-chest-x-ray-dataset",
    "access": "registration",
    "steps": "Sign in to Kaggle and download the dataset. Put archive.zip in this folder as it is: it is extracted "
             "by python -m prepare. The kaggle command names it nigeria-chest-x-ray-dataset.zip: rename it.",
    "license": "CC BY-NC-SA 4.0, as declared on Kaggle",
    "sha256": None,
    "note": "Its 2,600 file names and sizes are those of two other Kaggle sets, the COVID-19 Radiography Database "
            "and the Tuberculosis (TB) Chest X-ray Database: the images may not come from Nigeria (docs/decisions.md).",
    "bibtex": """@misc{musa2024nigeria,
  author = {Musa, Aminu and Adamu, Mariya Ibrahim and Kakudi, Habeebah Adamu and Lawal, Yusuf},
  title = {Nigeria Chest X-ray Dataset},
  publisher = {Kaggle},
  year = {2024},
  doi = {10.34740/KAGGLE/DSV/9370352}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    images = images_by_class(unpacked(root / RAW), CLASSES)
    return label_rows(images.assign(patient=images["name"]), SITE, images["class"].map(labels))
