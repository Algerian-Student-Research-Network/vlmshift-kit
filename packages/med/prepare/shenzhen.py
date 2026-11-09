"""Shenzhen Hospital CXR Set (China): 336 tuberculosis and 326 normal images of Shenzhen No. 3 People's Hospital.

Only the folder CXR_png is read: the masks of the annotations are PNG files too. No patient
number: each image is its own patient (a limit).
"""
from pathlib import Path

import pandas as pd

from label_rules.shenzhen import labels

from .common import LayoutError, label_rows

SITE = "Shenzhen"
RAW = "shenzhen/CXR_png"  # a folder: the class of an image is the end of its name

SOURCE = {
    "name": "Shenzhen Hospital CXR Set (U.S. National Library of Medicine)",
    "page": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html",
    "access": "open",
    "steps": "Open Shenzhen-Hospital-CXR-Set and download its folder CXR_png (662 PNG) into this folder.",
    "license": "No licence stated. The ReadMe asks not to share the data outside your group, "
               "to credit NLM and Shenzhen No. 3 People's Hospital, and to cite the papers below",
    "sha256": None,
    "bibtex": """@article{jaeger2014twopublic,
  author = {Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\\'a}ng, Yi-Xiang J. and Lu, Pu-Xuan and Thoma, George},
  title = {Two public chest X-ray datasets for computer-aided screening of pulmonary diseases},
  journal = {Quantitative Imaging in Medicine and Surgery},
  volume = {4},
  number = {6},
  pages = {475--477},
  year = {2014},
  doi = {10.3978/j.issn.2223-4292.2014.11.20}
}
@article{jaeger2014automatic,
  author = {Jaeger, Stefan and Karargyris, Alexandros and Candemir, Sema and Folio, Les and Siegelman, Jenifer and Callaghan, Fiona and Xue, Zhiyun and Palaniappan, Kannappan and Singh, Rahul K. and Antani, Sameer and Thoma, George and Wang, Yi-Xiang and Lu, Pu-Xuan and McDonald, Clement J.},
  title = {Automatic Tuberculosis Screening Using Chest Radiographs},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {33},
  number = {2},
  pages = {233--245},
  year = {2014},
  doi = {10.1109/TMI.2013.2284099}
}
@article{candemir2014lung,
  author = {Candemir, Sema and Jaeger, Stefan and Palaniappan, Kannappan and Musco, Jonathan P. and Singh, Rahul K. and Xue, Zhiyun and Karargyris, Alexandros and Antani, Sameer and Thoma, George and McDonald, Clement J.},
  title = {Lung Segmentation in Chest Radiographs Using Anatomical Atlases With Nonrigid Registration},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {33},
  number = {2},
  pages = {577--590},
  year = {2014},
  doi = {10.1109/TMI.2013.2290491}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    paths = sorted((root / RAW).glob("CHNCXR_*.png"))
    if not paths:
        raise LayoutError(f"no image named CHNCXR_<number>_<class>.png in {root / RAW}")
    images = pd.DataFrame({"image": [str(path) for path in paths], "patient": [path.name for path in paths]})
    return label_rows(images, SITE, images["patient"].map(labels))
