"""JSRT: the file name gives the class. JPCLN### has a lung nodule (154 cases), JPCNN### has none (93)."""
CLASSES = {"JPCLN": "positive", "JPCNN": "negative"}


def labels(name: str) -> dict[str, str]:
    return {"nodule": CLASSES.get(name[:5].upper(), "unknown")}
