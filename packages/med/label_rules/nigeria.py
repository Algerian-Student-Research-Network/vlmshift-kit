"""Nigeria Chest X-ray: four classes, one folder each, that exclude each other.

The policy of the paper: an image is positive for its class and negative for the two others
among normal, pneumonia and tuberculosis. COVID-19 is not a concept of the paper: an image of
that class states nothing here, and stays unknown for the three.
"""
from .template import one_class

CONCEPTS = {"NORMAL": "normal", "PNEUMONIA": "pneumonia", "TB": "tuberculosis", "COVID": None}
CLASSES = list(CONCEPTS)
ASKED = [concept for concept in CONCEPTS.values() if concept]


def labels(folder: str) -> dict[str, str]:
    return one_class(CONCEPTS[folder], ASKED)
