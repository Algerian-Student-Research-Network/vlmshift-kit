"""RSNA-RICORD-1c: each study was classified by several radiologists (MD.ai annotations).

All its patients had a positive COVID-19 test: the class says whether the image shows it. The
class chosen by most readers is kept, a tie is unknown. Typical appearance → positive, Negative
for pneumonia → negative. Indeterminate and atypical are not a statement either way, and Invalid
study is dropped: unknown.
"""
from collections import Counter

APPEARANCES = {
    "typical appearance": "positive", "negative for pneumonia": "negative",
    "indeterminate appearance": "unknown", "atypical appearance": "unknown", "invalid study": "unknown",
}


def labels(votes: list[str]) -> dict[str, str]:
    counts = Counter(vote.strip().lower() for vote in votes if vote.strip().lower() in APPEARANCES)
    ranked = counts.most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return {"covid19": "unknown"}
    return {"covid19": APPEARANCES[ranked[0][0]]}
