"""PadChest-GR (Spain): 4,555 frontal studies of Hospital Sant Joan d'Alacant, with grounded bilingual reports.

The file layout below follows the code published with the dataset (a master table and a JSON of
reports). The agreement keeps the documentation of the release out of reach without access: if a
name differs, the error lists what the file contains.
"""
import json
from pathlib import Path

import pandas as pd

from label_rules.padchest_gr import labels

from .common import LayoutError, label_rows, locate, require

SITE = "PadChest"
RAW = "padchest_gr/grounded_reports_20240819.json"
TABLE = "padchest_gr/master_table.csv"
IMAGE, PATIENT = "ImageID", "PatientID"

SOURCE = {
    "name": "PadChest-GR, grounded reports of 2024-08-19",
    "page": "https://bimcv.cipf.es/bimcv-projects/padchest-gr/",
    "access": "agreement",
    "steps": "Fill in the request form linked from the page and accept the PADCHEST research use agreement. "
             "Put grounded_reports_20240819.json, master_table.csv and the PNG images in this folder.",
    "license": "PADCHEST research use agreement: research only, no redistribution of any part",
    "sha256": None,
    "bibtex": """@article{coelhodecastro2025padchestgr,
  author = {Coelho de Castro, Daniel and Bustos, Aurelia and Bannur, Shruthi and Hyland, Stephanie and Bouzid, Kenza and Wetscherek, Maria Teodora and S{\\'a}nchez-Valverde, Maria Dolores and Jaques-P{\\'e}rez, Lara and P{\\'e}rez-Rodr{\\'i}guez, Lourdes and Takeda, Kenji and Salinas-Serrano, Jos{\\'e} Mar{\\'i}a and Alvarez-Valle, Javier and Galant-Herrero, Joaqu{\\'i}n and Pertusa, Antonio},
  title = {PadChest-GR: A Bilingual Chest X-ray Dataset for Grounded Radiology Report Generation},
  journal = {NEJM AI},
  volume = {2},
  number = {7},
  year = {2025},
  doi = {10.1056/AIdbp2401120}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    reports = json.loads((root / RAW).read_text(encoding="utf-8"))
    if not isinstance(reports, list) or not all(IMAGE in report and "findings" in report for report in reports):
        raise LayoutError(f"{root / RAW}: expected a list of reports with {IMAGE} and findings")
    table = pd.read_csv(root / TABLE, dtype=str, keep_default_na=False)
    require(table, [IMAGE, PATIENT], root / TABLE)
    patients = table.drop_duplicates(IMAGE).set_index(IMAGE)[PATIENT]
    if missing := [report[IMAGE] for report in reports if report[IMAGE] not in patients.index]:
        raise LayoutError(f"{len(missing)} images of {RAW} are not in {TABLE}, such as {missing[0]}")
    names = [report[IMAGE] for report in reports]
    images = pd.DataFrame({"image": locate(names, root / "padchest_gr"), "patient": patients.loc[names].to_numpy()})
    return label_rows(images, SITE, [labels(report["findings"]) for report in reports])
