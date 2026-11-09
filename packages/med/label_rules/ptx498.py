"""PTX-498: every image shows a pneumothorax, contoured by two radiologists. No negative: the
source is positive-only, so accuracy can be computed but not balanced accuracy (section 4.2)."""


def labels(_image: str) -> dict[str, str]:
    return {"pneumothorax": "positive"}
