"""NIH ChestX-ray14: the findings of the "Finding Labels" column, separated by |.

The paper reads these labels as exhaustive over the 14 findings: a finding listed is positive, a
finding not listed is negative, and "No Finding" is the concept normal. The labels were mined from
the reports by NLP, so a negative means "not reported", the known limit of this source.
Pleural_Thickening is asked as pleural_plaque, the RadLex concept the annotated workbook gives it.
"""
FINDINGS = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Consolidation": "consolidation",
    "Edema": "pulmonary_edema", "Effusion": "pleural_effusion", "Emphysema": "emphysema", "Fibrosis": "fibrosis",
    "Hernia": "hiatal_hernia", "Infiltration": "infiltrate", "Mass": "mass", "Nodule": "nodule",
    "Pleural_Thickening": "pleural_plaque", "Pneumonia": "pneumonia", "Pneumothorax": "pneumothorax",
}
CONCEPTS = [*FINDINGS.values(), "normal"]


def labels(finding_labels: str) -> dict[str, str]:
    stated = set(finding_labels.split("|"))
    found = {concept: "positive" if name in stated else "negative" for name, concept in FINDINGS.items()}
    return {**found, "normal": "positive" if stated == {"No Finding"} else "negative"}
