"""Create the shared toy dataset (section 7.5): 8 images, 16 rows, a broken table.

Run once: python tests/fixtures/toy/make_toy.py
The images are simple shapes: no license problem.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent

QUESTIONS = {
    "EN": "Is there a circle in this image? Answer yes or no.",
    "FR": "Y a-t-il un cercle dans cette image ? Répondez par oui ou non.",
}
POSITIVE = {"EN": "a picture with a circle", "FR": "une image avec un cercle"}
NEGATIVE = {"EN": "a picture with no circle", "FR": "une image sans cercle"}

# (image number, subject, site, label). Odd images: a circle; even images: a square.
IMAGES = [
    (1, "subj1", "A", "positive"),
    (2, "subj1", "A", "negative"),
    (3, "subj2", "A", "positive"),
    (4, "subj2", "A", "negative"),
    (5, "subj3", "B", "positive"),
    (6, "subj3", "B", "negative"),
    (7, "subj4", "B", "unknown"),
    (8, "subj4", "B", "unknown"),
]


def draw_image(number: int, site: str, path: Path) -> None:
    """Draw a shape. Light or dark background depending on the site; different size per image."""
    background = 220 if site == "A" else 90
    image = Image.new("L", (32, 32), background)
    box = (6, 6, 16 + number, 16 + number)
    draw = ImageDraw.Draw(image)
    (draw.ellipse if number % 2 else draw.rectangle)(box, fill=0)
    image.save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def toy_rows() -> list[dict]:
    """The toy table: one row per image and per language, so 16 rows."""
    return [
        {
            "image": f"images/img{number}.png",
            "question": QUESTIONS[language],
            "positive_text": POSITIVE[language],
            "negative_text": NEGATIVE[language],
            "label": label,
            "language": language,
            "site": site,
            "subject": subject,
        }
        for number, subject, site, label in IMAGES
        for language in QUESTIONS
    ]


def broken_rows() -> list[dict]:
    """The broken table: one example of each edge case of step 1."""

    def row(id, sample, subject, label, language, image="images/img1.png", question="Q?"):
        return {
            "id": id,
            "sample": sample,
            "image": image,
            "question": question,
            "label": label,
            "language": language,
            "device": "X",  # a single level: no comparison possible
            "subject": subject,
        }

    good = row("1", "s1", "subj1", "positive", "EN")
    return [
        good,
        dict(good),  # strictly identical row
        row("2", "s1", "subj1", "positive", "FR"),
        row("", "s2", "subj2", "negative", "EN"),  # empty original identifier
        row("2", "s2", "subj2", "negative", "FR"),  # duplicated original identifier
        row("3", "s3", "subj3", "positive", "EN", image="images/absent.png"),  # image not found
        row("4", "s4", "subj4", "maybe", "EN"),  # invalid label
        row("5", "s5", "subj5", "", "EN"),  # empty label
        row("6", "s6", "subj6", "positive", "EN"),
        row("7", "s6", "subj7", "positive", "FR"),  # two subjects in one sample
        row("8", "s7", "subj8", "positive", "EN"),
        row("9", "s7", "subj8", "negative", "FR"),  # two labels in one sample
        row("10", "s8", "subj9", "positive", "EN"),
        row("11", "s8", "subj9", "positive", "EN", question="Other?"),  # same levels
    ]


def annotated_rows() -> list[dict]:
    """Hand-labeled answers, to test the readers (list and small LLM)."""
    answers = [
        ("Yes.", "EN", "yes"), ("yes", "EN", "yes"), ("  YES!", "EN", "yes"), ("No.", "EN", "no"),
        ("no", "EN", "no"), ("NO", "EN", "no"), ("Yes, there is one.", "EN", "yes"),
        ("No, there is not.", "EN", "no"), ("maybe", "EN", "unclear"), ("I cannot tell", "EN", "unclear"),
        ("Oui.", "FR", "yes"), ("oui", "FR", "yes"), ("Non.", "FR", "no"), ("non", "FR", "no"),
        ("Oui, il y en a un.", "FR", "yes"), ("Non, il n'y en a pas.", "FR", "no"),
        ("peut-être", "FR", "unclear"), ("हाँ", "HI", "yes"), ("हाँ।", "HI", "yes"),
        ("नहीं", "HI", "no"), ("नहीं।", "HI", "no"), ("शायद", "HI", "unclear"),
    ]
    return [{"response": r, "language": language, "expected": e} for r, language, e in answers]


def main() -> None:
    (HERE / "images").mkdir(exist_ok=True)
    for number, _, site, _ in IMAGES:
        draw_image(number, site, HERE / "images" / f"img{number}.png")
    write_csv(HERE / "toy.csv", toy_rows())
    write_csv(HERE / "broken.csv", broken_rows())
    write_csv(HERE / "annotated_answers.csv", annotated_rows())


if __name__ == "__main__":
    main()
