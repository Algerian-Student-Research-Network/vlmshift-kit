"""Normal/Pneumonia of Epic Chittagong (Bangladesh): 2,626 images in the archive of version 3.

No patient number: each image is its own patient (a limit).
"""
from pathlib import Path

import pandas as pd

from label_rules.pneumonia_bangladesh import CLASSES, labels

from .common import images_by_class, label_rows, unpacked

SITE = "Bangladesh"
RAW = "pneumonia_bangladesh/Chest-X-Ray Epic Hospital Chittagong, Bangladesh pneumonia.zip"

SOURCE = {
    "name": "A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh, version 3 (Mendeley Data)",
    "page": "https://data.mendeley.com/datasets/wndbd5r26y/3",
    "access": "open",
    "steps": "Download the archive of version 3 and put it in this folder as it is: it is extracted by python -m prepare.",
    "license": "CC BY 4.0",
    "sha256": "3d07a6f0d3ba37c9a72edba45d05e307ab57ebf2ba2effe2b3373da448dcad0c",  # the one Mendeley publishes
    "note": "The page announces 3,355 images, the archive holds 2,626. Version 6 (2026) removes duplicates and "
            "images whose copies have opposite labels; the paper uses version 3.",
    "bibtex": """@misc{hira2026epic,
  author = {Hira, Md Irfanul Kabir and Bithee, Mst Moriom Akter and Ahmed, Shafee and Akter, Laboni and Anonna, Mst. Jannatul Mawa},
  title = {A Primary Chest X-ray Dataset of Normal and Pneumonia Cases from Epic Chittagong, Bangladesh},
  publisher = {Mendeley Data},
  version = {3},
  year = {2026},
  doi = {10.17632/wndbd5r26y.3}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    images = images_by_class(unpacked(root / RAW), CLASSES)
    return label_rows(images.assign(patient=images["name"]), SITE, images["class"].map(labels))
