"""PadChest-GR: each report is a list of sentences, positive (abnormal) or negative, with their labels.

A label in a positive sentence is positive. A label in a negative sentence only ("no pleural
effusion") is negative. A report without any positive sentence is a normal study: negative for
every concept PadChest-GR annotates. Anything else is not stated: unknown (section 4.2).
The PadChest labels are matched without case; those absent from LABELS are not asked.
"""
LABELS = {
    "atelectasis": "atelectasis", "laminar atelectasis": "atelectasis", "lobar atelectasis": "atelectasis",
    "segmental atelectasis": "atelectasis", "round atelectasis": "atelectasis", "total atelectasis": "atelectasis",
    "consolidation": "consolidation", "infiltrates": "infiltration", "pneumothorax": "pneumothorax",
    "pulmonary edema": "edema", "emphysema": "emphysema", "pulmonary fibrosis": "fibrosis",
    "pleural effusion": "effusion", "loculated pleural effusion": "effusion", "pneumonia": "pneumonia",
    "pleural thickening": "pleural_thickening", "apical pleural thickening": "pleural_thickening",
    "cardiomegaly": "cardiomegaly", "nodule": "nodule", "multiple nodules": "nodule", "mass": "mass",
    "pulmonary mass": "mass", "hiatal hernia": "hernia", "tuberculosis": "tuberculosis", "covid 19": "covid19",
    "endotracheal tube": "endotracheal_tube", "nsg tube": "nasogastric_tube",
    "central venous catheter": "central_venous_catheter",
    "central venous catheter via jugular vein": "central_venous_catheter",
    "central venous catheter via subclavian vein": "central_venous_catheter",
    "swan-ganz catheter": "swan_ganz_catheter", "swan ganz catheter": "swan_ganz_catheter",
    "aortic elongation": "aortic_elongation", "scoliosis": "scoliosis",
    "vertebral degenerative changes": "vertebral_degenerative_changes",
    "aortic atheromatosis": "aortic_atheromatosis", "air trapping": "air_trapping",
    "interstitial pattern": "interstitial_pattern", "vascular hilar enlargement": "hilar_enlargement",
    "hilar enlargement": "hilar_enlargement", "costophrenic angle blunting": "costophrenic_angle_blunting",
    "alveolar pattern": "alveolar_pattern", "callus rib fracture": "rib_fracture", "rib fracture": "rib_fracture",
    "pacemaker": "pacemaker", "dual chamber device": "pacemaker", "single chamber device": "pacemaker",
    "calcified granuloma": "calcified_granuloma", "diaphragmatic eventration": "diaphragmatic_eventration",
    "volume loss": "volume_loss", "bronchiectasis": "bronchiectasis",
}
CONCEPTS = sorted(set(LABELS.values()))


def labels(findings: list[dict]) -> dict[str, str]:
    stated = {}
    for finding in findings:
        for label in finding.get("labels") or []:
            if concept := LABELS.get(str(label).strip().lower()):
                if finding.get("abnormal"):
                    stated[concept] = "positive"
                else:
                    stated.setdefault(concept, "negative")
    if not any(finding.get("abnormal") for finding in findings):
        return {concept: stated.get(concept, "negative") for concept in CONCEPTS}
    return stated
