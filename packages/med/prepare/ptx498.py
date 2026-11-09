"""PTX-498 (China): 498 images with a pneumothorax, from three hospitals (SiteA 247, SiteB 179, SiteC 72).

Each image comes with its mask: only the files <n>.1.img.png are images. No patient number: each
image is its own patient (a limit, docs/decisions.md).
"""
from pathlib import Path

import pandas as pd

from label_rules.ptx498 import labels

from .common import LayoutError, label_rows, unpacked

SITE = "PTX-498"
RAW = "ptx498/PTX-498-v2-fix.zip"

SOURCE = {
    "name": "PTX-498, version v2-fix (Zenodo)",
    "page": "https://zenodo.org/records/8266529",
    "access": "open",
    "steps": "Download PTX-498-v2-fix.zip (MD5 a1574afa9b1f7f299e9fcf142f916b9e on the page) and put it in this "
             "folder as it is: it is extracted by python -m prepare.",
    "license": "CC BY 4.0",
    "sha256": None,
    "note": "Positive only: every image shows a pneumothorax.",
    "bibtex": """@article{wang2021deepsdm,
  author = {Wang, Yunpeng and Wang, Kang and Peng, Xueqing and Shi, Lili and Sun, Jing and Zheng, Shibao and Shan, Fei and Shi, Weiya and Liu, Lei},
  title = {DeepSDM: Boundary-aware pneumothorax segmentation in chest X-ray images},
  journal = {Neurocomputing},
  volume = {454},
  pages = {201--211},
  year = {2021},
  doi = {10.1016/j.neucom.2021.05.029}
}
@misc{wang2021ptx498,
  author = {Wang, Yunpeng},
  title = {PTX-498: A multi-center pneumothorax segmentation chest X-ray image dataset},
  publisher = {Zenodo},
  version = {v2-fix},
  year = {2021},
  doi = {10.5281/zenodo.8266529}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    folder = unpacked(root / RAW)
    paths = sorted(folder.rglob("*.img.png"))
    if not paths:
        raise LayoutError(f"no image named <n>.1.img.png under {folder}")
    images = pd.DataFrame({"image": [str(path) for path in paths],
                           "patient": [path.relative_to(folder).as_posix() for path in paths]})
    return label_rows(images, SITE, images["image"].map(labels))
