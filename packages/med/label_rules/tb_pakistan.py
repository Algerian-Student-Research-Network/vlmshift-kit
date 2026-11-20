"""Tuberculosis Chest X-rays (Pakistan): the class is the folder of the image.
The two classes exclude each other: normal or tuberculosis."""
from .template import one_class

CLASSES = {"TB Chest X-rays": "tuberculosis", "Normal Chest X-rays": "normal"}
ASKED = list(CLASSES.values())


def labels(folder: str) -> dict[str, str]:
    return one_class(CLASSES[folder], ASKED)
