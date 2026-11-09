"""Nigeria Chest X-ray: four classes, one folder each. A class is positive for its own concept.

NORMAL is negative for the three. COVID is not stated negative for pneumonia, nor TB for
COVID-19: an image of one class stays unknown for the concepts of the others (section 4.2).
"""
CONCEPTS = {"COVID": "covid19", "PNEUMONIA": "pneumonia", "TB": "tuberculosis"}
CLASSES = [*CONCEPTS, "NORMAL"]


def labels(folder: str) -> dict[str, str]:
    if folder == "NORMAL":
        return {concept: "negative" for concept in CONCEPTS.values()}
    return {CONCEPTS[folder]: "positive"}
