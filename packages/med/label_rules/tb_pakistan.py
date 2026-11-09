"""Tuberculosis Chest X-rays (Pakistan): the class is the folder of the image."""
CLASSES = {"TB Chest X-rays": "positive", "Normal Chest X-rays": "negative"}


def labels(folder: str) -> dict[str, str]:
    return {"tuberculosis": CLASSES[folder]}
