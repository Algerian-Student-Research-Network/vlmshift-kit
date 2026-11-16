"""Montgomery County CXR Set: the last digit of the file name is the class (NLM ReadMe).
MCUCXR_0001_0.png → normal. MCUCXR_0104_1.png → tuberculosis.
The two classes exclude each other: a normal image is negative for tuberculosis, and a
tuberculosis image is negative for normal.
"""
from .template import one_class

CLASSES = {"0": "normal", "1": "tuberculosis"}
ASKED = list(CLASSES.values())


def labels(name: str) -> dict[str, str]:
    return one_class(CLASSES.get(name.rsplit(".", 1)[0][-1]), ASKED)
