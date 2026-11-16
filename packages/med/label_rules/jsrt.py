"""JSRT: the file name gives the class. JPCLN### has a lung nodule (154 cases), JPCNN### has none (93).

The paper asks more than this rule can: from the diagnosis of each nodule in the clinical file of
JSRT, it asks about lung cancer, granuloma and hamartoma. The file name only says whether there is
a nodule, so this rule asks about nodule. The exact questions of the paper on JSRT come from its
evaluation plan (plans.py), not from this rule.
"""
CLASSES = {"JPCLN": "positive", "JPCNN": "negative"}


def labels(name: str) -> dict[str, str]:
    return {"nodule": CLASSES.get(name[:5].upper(), "unknown")}
