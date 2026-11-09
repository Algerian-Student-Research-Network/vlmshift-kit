"""Fake models for the test dataset (tests/fixtures/toy, section 7.5): their answers follow a fixed rule."""
import math
import re
from pathlib import Path

from ..base import Model, ModelError


def image_number(path: str) -> int:
    """"images/img3.png" → 3. Any other image is a failure."""
    found = re.search(r"img(\d+)$", Path(path).stem)
    if not found or not 1 <= int(found[1]) <= 8:
        raise ModelError(f"unknown image: {path}")
    return int(found[1])


def has_circle(number: int) -> bool:
    return number % 2 == 1  # odd images: a circle


def is_french(text: str) -> bool:
    return "cercle" in text


class FakeGenerative(Model):
    """Correct in English. In French: two reversed answers, one "maybe", one failure."""

    kind = "generative"

    def generate(self, images, prompts, params):
        return [self._answer(image_number(i), p) for i, p in zip(images, prompts)]

    def _answer(self, number: int, prompt: str) -> str:
        if not is_french(prompt):
            return "Yes." if has_circle(number) else "No."
        if self.params.get("faults", True):
            if number == 6:
                return "maybe"
            if number == 8:
                raise ModelError("simulated failure")
        inverted = number in (1, 4)
        return "oui" if has_circle(number) != inverted else "non"


class FakeContrastive(Model):
    """Correct in English. In French: one tie and one error. Its vectors encode the site."""

    kind = "contrastive"
    supports_embed = True

    def score(self, images, positives, negatives):
        return [self._scores(image_number(i), p) for i, p in zip(images, positives)]

    def _scores(self, number: int, positive: str) -> tuple[float, float]:
        if is_french(positive) and number == 2:
            return 0.5, 0.5  # tie → INVALID
        wrong = is_french(positive) and number == 3
        return (1.0, 0.0) if has_circle(number) != wrong else (0.0, 1.0)

    def embed(self, images):
        numbers = [image_number(image) for image in images]
        return [[1.0 if n <= 4 else -1.0, math.sin(n), math.cos(n)] for n in numbers]


def fake_reader(prompt: str) -> str:
    """Fake small LLM: looks only at the first word. It does not know Hindi, on purpose."""
    words = re.findall(r"\w+", prompt.rsplit("Answer:", 1)[1].lower())
    first = words[0] if words else ""
    return {"yes": "yes", "oui": "yes", "no": "no", "non": "no"}.get(first, "unclear")
