"""Montgomery County CXR Set (USA): 58 tuberculosis and 80 normal posteroanterior images.

Only the folder CXR_png is read: the lung masks are PNG files too, with the same names. No
patient number: each image is its own patient (a limit, docs/decisions.md).
"""
from pathlib import Path

import pandas as pd

from label_rules.montgomery import labels

from .common import LayoutError, label_rows

SITE = "Montgomery"
RAW = "montgomery/CXR_png"  # a folder: the class of an image is the end of its name

SOURCE = {
    "name": "Montgomery County CXR Set (U.S. National Library of Medicine)",
    "page": "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html",
    "access": "open",
    "steps": "Open Montgomery-County-CXR-Set/MontgomerySet and download its folder CXR_png (138 PNG) into this folder.",
    "license": "No licence stated. The ReadMe asks not to share the data outside your group, "
               "to credit NLM and to cite the papers of the Shenzhen set",
    "sha256": "b3680373e7102f5446684720c8af72c6228089a251f5723825d19b9d418bc8a8",  # folder CXR_png, NLM, downloaded 2026-11-09
    "note": "The same three papers as the Shenzhen set: references.bib gives them once.",
    "bibtex": """@misc{montgomery2014set,
  author = {{U.S. National Library of Medicine}},
  title = {Montgomery County CXR Set},
  year = {2014},
  howpublished = {\\url{https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/index.html}},
  note = {Cite with jaeger2014twopublic, jaeger2014automatic and candemir2014lung}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    paths = sorted((root / RAW).glob("MCUCXR_*.png"))
    if not paths:
        raise LayoutError(f"no image named MCUCXR_<number>_<class>.png in {root / RAW}")
    images = pd.DataFrame({"image": [str(path) for path in paths], "patient": [path.name for path in paths]})
    return label_rows(images, SITE, images["patient"].map(labels))
