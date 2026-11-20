"""RSNA-RICORD-1c: each study was classified by several radiologists (MD.ai annotations).

All its patients had a positive COVID-19 test. The paper asks about lung opacity, the RadLex
concept behind the three appearances: typical, indeterminate and atypical all describe opacities,
so they are positive. RICORD defines "Negative for pneumonia" as no lung opacity: negative.
The class chosen by most readers is kept; a tie between an appearance and its absence is unknown.
Invalid study is dropped: unknown.
"""
from collections import Counter

APPEARANCES = {
    "typical appearance": "positive", "indeterminate appearance": "positive", "atypical appearance": "positive",
    "negative for pneumonia": "negative", "invalid study": "unknown",
}


def labels(votes: list[str]) -> dict[str, str]:
    counts = Counter(APPEARANCES[vote.strip().lower()] for vote in votes if vote.strip().lower() in APPEARANCES)
    ranked = counts.most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return {"opacity": "unknown"}
    return {"opacity": ranked[0][0]}
