"""NIH ChestX-ray14 (USA): 112,120 frontal images of 30,805 patients, 14 findings mined from the reports."""
from pathlib import Path

import pandas as pd

from label_rules.nih import labels

from .common import label_rows, locate, require

SITE = "NIH"
RAW = "nih/Data_Entry_2017_v2020.csv"

SOURCE = {
    "name": "NIH ChestX-ray14, labels of 2020 (Data_Entry_2017_v2020.csv)",
    "page": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
    "access": "open",
    "steps": "Download Data_Entry_2017_v2020.csv and the 12 archives images/images_001.tar.gz to "
             "images_012.tar.gz (batch_download_zips.py does it), then extract the archives in the same folder.",
    "license": "No restriction on use (FAQ_CHESTXRAY.pdf): link the NIH download site, "
               "acknowledge the NIH Clinical Center and cite the paper",
    "sha256": None,
    "note": "LOG_CHESTXRAY.pdf lists pixel-identical images, some under two patient numbers.",
    "bibtex": """@inproceedings{wang2017chestxray,
  author = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  title = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3462--3471},
  year = {2017},
  doi = {10.1109/CVPR.2017.369}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / RAW, dtype=str, keep_default_na=False)
    require(raw, ["Image Index", "Finding Labels", "Patient ID"], root / RAW)
    images = pd.DataFrame({"image": locate(raw["Image Index"], root / "nih"), "patient": raw["Patient ID"]})
    return label_rows(images, SITE, raw["Finding Labels"].map(labels))
