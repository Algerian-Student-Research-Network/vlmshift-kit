"""The yes/no task type, reading answers from a list (section 3)."""
from __future__ import annotations

import unicodedata

from ..common import nfc
from .base import Task

# Lists used when no reader is declared.
DEFAULT_ANSWERS = {"yes": ["yes"], "no": ["no"]}


# How a column says that a question is negated ("Is there no nodule?"), or that it is not.
NEGATED = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}


def negated_rows(column) -> list[bool | None]:
    """For each value of a negation column: negated, not negated, or None when it says neither."""
    return [NEGATED.get(str(value).strip().lower()) for value in column]


def is_noise(char: str) -> bool:
    """Punctuation, space or control character, in every language. Also ` and ~, which format text."""
    return unicodedata.category(char)[0] in "PZC" or char in "`~"


def clean(text: str) -> str:
    """Lowercase, without spaces or punctuation at the start and end."""
    chars = list(nfc(text).lower())
    while chars and is_noise(chars[0]):
        chars.pop(0)
    while chars and is_noise(chars[-1]):
        chars.pop()
    return "".join(chars)


class YesNoTask(Task):
    name = "yes_no"

    def read(self, response: str, answers: dict) -> str | None:
        text = clean(response)
        for decision in ("yes", "no"):
            if text in {clean(answer) for answer in answers[decision]}:
                return decision
        return None

    def is_correct(self, decision: str, label: str) -> bool | None:
        if label == "unknown":
            return None
        return (decision == "yes") == (label == "positive")
