"""NIH ChestX-ray14: the findings of the "Finding Labels" column, separated by |.

The labels were mined from the reports by NLP. A finding listed is positive. "No Finding" states
that none of the 14 is there: negative for all. A finding not listed next to another one is not
a negative, only not mentioned: unknown (section 4.2).
"""
FINDINGS = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
    "Edema": "edema", "Effusion": "effusion", "Emphysema": "emphysema", "Fibrosis": "fibrosis",
    "Hernia": "hernia", "Infiltration": "infiltration", "Mass": "mass", "Nodule": "nodule",
    "Pleural_Thickening": "pleural_thickening", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax",
}


def labels(finding_labels: str) -> dict[str, str]:
    stated = set(finding_labels.split("|"))
    if stated == {"No Finding"}:
        return {concept: "negative" for concept in FINDINGS.values()}
    return {concept: "positive" if name in stated else "unknown" for name, concept in FINDINGS.items()}
