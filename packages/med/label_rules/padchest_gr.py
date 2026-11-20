"""PadChest-GR: each report is a list of sentences, positive (abnormal) or negative, with their labels.

The policy of the paper for this source: a label in a positive sentence is positive. A report
without any positive sentence is a normal study: normal is positive and every other concept is
negative. In an abnormal study, normal is negative and a finding that is not reported stays
unknown: the reports are not assumed to list everything.

The PadChest labels are matched without case, and mapped to the concepts of the paper as the
annotated workbook maps them (alveolar pattern → consolidation, vascular hilar enlargement →
congestion...). Those absent from LABELS are not asked.
"""
LABELS = {
    "atelectasis": "atelectasis", "laminar atelectasis": "atelectasis", "lobar atelectasis": "atelectasis",
    "segmental atelectasis": "atelectasis", "round atelectasis": "atelectasis", "total atelectasis": "atelectasis",
    "alveolar pattern": "consolidation", "consolidation": "consolidation",
    "pleural effusion": "pleural_effusion", "loculated pleural effusion": "pleural_effusion",
    "pleural thickening": "pleural_plaque", "apical pleural thickening": "pleural_plaque",
    "cardiomegaly": "cardiomegaly", "nodule": "nodule", "multiple nodules": "nodule", "hiatal hernia": "hiatal_hernia",
    "endotracheal tube": "endotracheal_tube", "nsg tube": "nasogastric_tube",
    "central venous catheter": "central_venous_catheter",
    "central venous catheter via jugular vein": "central_venous_catheter",
    "central venous catheter via subclavian vein": "central_venous_catheter",
    "scoliosis": "scoliosis", "vertebral degenerative changes": "spondylosis_deformans",
    "aortic atheromatosis": "atherosclerosis", "air trapping": "air_trapping", "hyperinflated lung": "air_trapping",
    "interstitial pattern": "interstitial_lung_disease", "vascular hilar enlargement": "congestion",
    "fracture": "fracture", "callus rib fracture": "fracture", "rib fracture": "fracture",
    "pacemaker": "pacemaker", "dual chamber device": "pacemaker", "single chamber device": "pacemaker",
    "bronchiectasis": "bronchiectasis", "goiter": "goiter", "osteopenia": "osteopenia",
}
CONCEPTS = sorted({*LABELS.values(), "normal"})


def labels(findings: list[dict]) -> dict[str, str]:
    positive = {LABELS[name] for finding in findings if finding.get("abnormal")
                for label in finding.get("labels") or [] if (name := str(label).strip().lower()) in LABELS}
    if not any(finding.get("abnormal") for finding in findings):
        return {concept: "positive" if concept == "normal" else "negative" for concept in CONCEPTS}
    return {**{concept: "positive" for concept in positive}, "normal": "negative"}
