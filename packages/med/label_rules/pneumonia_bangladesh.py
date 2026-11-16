"""Normal/Pneumonia, Epic Chittagong (Bangladesh): the class is the folder of the image.
The two classes exclude each other: normal or pneumonia."""
from .template import one_class

CLASSES = {"pneumonia": "pneumonia", "normal": "normal"}
ASKED = list(CLASSES.values())


def labels(folder: str) -> dict[str, str]:
    return one_class(CLASSES[folder], ASKED)
