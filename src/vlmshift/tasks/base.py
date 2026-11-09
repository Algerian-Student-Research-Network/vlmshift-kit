"""Task interface: ask the question, read the answer, say whether it is correct (section 7.3)."""
from __future__ import annotations

from string import Formatter


def check_template(template: str) -> None:
    """Each {...} of a template must name a column. Raise ValueError otherwise."""
    try:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
    except ValueError as error:
        raise ValueError(f"template {template!r}: {error}. Write {{{{ or }}}} for a brace in the text") from error
    if not all(field.isidentifier() for field in fields):
        raise ValueError(f"template {template!r}: each {{...}} must contain a column name, such as {{question}}")


def holes(template: str) -> list[str]:
    """The columns a template asks for: "Is there {concept}?" → [concept]. The template was checked when read."""
    return [field for _, field, _, _ in Formatter().parse(template) if field]


class Task:
    name: str

    def build_prompt(self, row: dict, template: str | None) -> str:
        return (template or "{question}").format(**row)

    def build_contrastive(self, row: dict, pos: str, neg: str) -> tuple[str, str]:
        return pos.format(**row), neg.format(**row)

    def read(self, response: str, answers: dict) -> str | None:
        raise NotImplementedError

    def is_correct(self, decision: str, label: str) -> bool | None:
        raise NotImplementedError
