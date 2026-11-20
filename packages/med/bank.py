"""The concept bank of the paper: every text sent to a model comes from here.

configs/questions.yaml holds the 35 concepts in 7 languages and the wordings; configs/romanized.yaml
holds the Hindi, Chinese and Arabic questions written in Latin letters. This module turns a concept,
a language and a kind of question into the exact sentence of the paper, and guards the rule that
an official run needs validated terms.
"""
from pathlib import Path

from vlmshift.common import VLMShiftError
from vlmshift.config.load import read_yaml

HERE = Path(__file__).parent
LANGUAGES = ("EN", "FR", "ES", "HI", "ZH", "AR", "DE")
VARIANTS = ("canonical", "paraphrase_1", "paraphrase_2", "negated")
FORMATS = ("english_yes_no", "numeric_10", "label_ab", "localized_yes_no")
VALIDATED = "validated"


def load_bank() -> dict:
    return read_yaml(HERE / "configs" / "questions.yaml")


def load_romanized() -> dict:
    return read_yaml(HERE / "configs" / "romanized.yaml")


def question(bank: dict, concept: str, language: str, variant: str = "canonical", answer_format: str = "english_yes_no",
             term_language: str | None = None) -> str:
    """The question about a concept: its wording in `language`, then how to answer.

    term_language: the language of the medical term when it differs from that of the sentence
    (the frame × term experiment). `normal` is a statement about the whole image: it has its own
    wordings and no term, and its paraphrases are its canonical question.
    """
    wordings = bank["templates"][language]
    if concept == "normal":
        text = wordings["normal"]["negated" if variant == "negated" else "canonical"]
    else:
        text = wordings[variant].format(term=bank["concepts"][concept][term_language or language])
    return f"{text} {bank['answers'][language][answer_format]}"


def pair(bank: dict, concept: str, language: str, variant: str = "canonical", term_language: str | None = None) -> tuple[str, str]:
    """The two sentences a contrastive model compares: the finding is there, the finding is not.

    For a negated question the pair is turned around: its first sentence states what a yes to the
    question means, that the finding is absent.
    """
    wordings = bank["templates"][language]
    if concept == "normal":
        present, absent = wordings["normal"]["pair"]
    else:
        key = "canonical_pair" if variant in ("canonical", "negated") else f"{variant}_pair"
        present, absent = (text.format(term=bank["concepts"][concept][term_language or language]) for text in wordings[key])
    return (absent, present) if variant == "negated" else (present, absent)


def romanized(texts: dict, concept: str, language: str) -> tuple[str, str, str]:
    """The question and the pair of a concept in Latin letters: (question, present, absent)."""
    found = texts["concepts"][concept][language]
    return found["question"], found["positive"], found["negative"]


def answer_sets(bank: dict) -> dict:
    """The answers accepted for each format. Localized yes/no accepts the yes and no of every
    language: the format of a row does not say its language, and no word means yes in one and no in another."""
    languages = bank["languages"].values()
    return {
        "english_yes_no": {"yes": ["yes"], "no": ["no"]},
        "numeric_10": {"yes": ["1"], "no": ["0"]},
        "label_ab": {"yes": ["A"], "no": ["B"]},
        "localized_yes_no": {"yes": sorted({word for language in languages for word in language["yes"]}),
                             "no": sorted({word for language in languages for word in language["no"]})},
    }


def unvalidated(bank: dict) -> list[str]:
    """The concepts whose terms no clinician has reviewed yet."""
    return sorted(concept for concept, terms in bank["concepts"].items() if terms.get("status") != VALIDATED)


def draft_notice(bank: dict) -> str | None:
    """What a report must say while some terms are drafts. None once every term is validated."""
    drafts = unvalidated(bank)
    if not drafts:
        return None
    return (f"DRAFT TERMS: {len(drafts)} of {len(bank['concepts'])} concepts have translations not yet reviewed by a clinician "
            "(configs/questions.yaml). This is not an official run: a language effect may come from a wording.")


def gate(allowed: bool) -> str | None:
    """Refuse a run on draft terms unless it was asked for. Return what its report must say."""
    bank = load_bank()
    drafts = unvalidated(bank)
    if drafts and not allowed:
        raise VLMShiftError(
            f"{len(drafts)} of {len(bank['concepts'])} concepts have terms that no clinician has reviewed "
            f"(status: needs_validation in configs/questions.yaml), such as {', '.join(drafts[:3])}. "
            "An official run needs validated terms. To run anyway, add --allow-unvalidated: the report will say so"
        )
    return draft_notice(bank)
