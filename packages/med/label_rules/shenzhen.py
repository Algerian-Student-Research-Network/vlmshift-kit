"""Shenzhen Hospital CXR Set: the last digit of the file name is the class (NLM ReadMe).

CHNCXR_0001_0.png → normal, so tuberculosis negative. CHNCXR_0327_1.png → tuberculosis positive.
"""
CLASSES = {"0": "negative", "1": "positive"}


def labels(name: str) -> dict[str, str]:
    return {"tuberculosis": CLASSES.get(name.rsplit(".", 1)[0][-1], "unknown")}
