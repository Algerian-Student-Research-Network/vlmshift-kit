"""Normal/Pneumonia, Epic Chittagong (Bangladesh): the class is the folder of the image."""
CLASSES = {"pneumonia": "positive", "normal": "negative"}


def labels(folder: str) -> dict[str, str]:
    return {"pneumonia": CLASSES[folder]}
