"""Template of a preparation script. To copy once per source, then adapt. It is never run.

A script provides RAW (the expected raw file, read by check.py), SOURCE (its card) and prepare().
"""
from pathlib import Path

import pandas as pd

from label_rules.template import to_label

SITE = "example_source"
CONCEPT = "pneumothorax"
RAW = f"{SITE}/labels.csv"  # columns: path, patient_id, pneumothorax

# The card of the source. The data is never shared: the card says where to ask for it, which
# version was used and how to cite it. python cite.py turns the cards into SOURCES.md and references.bib.
SOURCE = {
    "name": "Example source (template, not a real source)",  # with its version
    "page": "https://example.org/example-source",  # the official page, where access is asked for
    "access": "open",  # open, registration, agreement or credentialed: see ACCESS in prepare/__init__.py
    "steps": "Download labels.csv and the images it lists.",
    "license": "Copy it from the page of the source",
    "sha256": "d9f6af48853f3135b3201b4650dfcda07481d37ead4f36cea107777011626695",  # of RAW: check.py gives it
    "bibtex": """@misc{example_source,
  title = {Template to copy, not a real source},
  note = {Replace with the BibTeX given on the page of the source}
}""",
}


def prepare(root: Path) -> pd.DataFrame:
    raw = pd.read_csv(root / RAW, dtype=str, keep_default_na=False)
    return pd.DataFrame({
        "image": [str(root / SITE / path) for path in raw["path"]],
        "patient": SITE + "_" + raw["patient_id"],  # the same number can exist in two sources
        "site": SITE,
        "concept": CONCEPT,
        "label": raw[CONCEPT].map(to_label),
    })
