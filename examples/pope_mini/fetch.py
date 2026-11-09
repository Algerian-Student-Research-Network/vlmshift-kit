"""POPE mini: 100 COCO images and their 1,800 POPE questions, fetched from their sources.

POPE (Li et al., EMNLP 2023) asks "Is there a <object> in the image?" about 500 images of
COCO val2014. Each image gets 3 objects that are in it and 3 that are not. The absent objects
are chosen in three modes, with 3,000 questions each:

- random: any object of COCO;
- popular: among the most frequent objects;
- adversarial: among the objects that often appear with those of the image.

The questions about present objects are the same in the three modes. Only the absent object
changes, and the models of the paper get worse from random to adversarial (reference.csv).

The questions are under the MIT licence. The images follow the terms of Flickr, so they are
not in this repository: this script fetches them. Run once, from anywhere:

    python examples/pope_mini/fetch.py

It writes data/pope_mini.csv and data/images/. A file already there is not fetched again.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
# A fixed commit: the files on main changed after the paper (spelling fixes), and a file on disk is never fetched again.
QUESTIONS = "https://raw.githubusercontent.com/RUCAIBox/POPE/109985c74a00a88ba5341f9b0c1d4900f2a1f789/output/coco/coco_pope_{mode}.json"
IMAGES = "http://images.cocodataset.org/val2014/{name}"
MODES = ("random", "popular", "adversarial")
N_IMAGES = 100
ASKED = re.compile(r"Is there (an? (.+)) in the image\?")


def fetch(url: str, target: Path) -> Path:
    """Download once. A partial file is never left behind."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as answer:
            partial.write_bytes(answer.read())
        partial.replace(target)
    return target


def read_questions(text: str) -> list[dict]:
    """One JSON object per line: question_id, image, text, label."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def chosen_images(questions: dict[str, list[dict]], count: int = N_IMAGES) -> list[str]:
    """The first images by name that every mode asks about: the choice never changes."""
    common = set.intersection(*({row["image"] for row in rows} for rows in questions.values()))
    return sorted(common)[:count]


def table_rows(questions: dict[str, list[dict]], images: list[str]) -> list[dict]:
    """One row per question about a chosen image, in the format VLMShift reads."""
    kept, rows = set(images), []
    for mode, lines in questions.items():
        for line in lines:
            if line["image"] not in kept:
                continue
            asked = ASKED.fullmatch(line["text"])
            if not asked:
                raise ValueError(f"Unexpected POPE question: {line['text']!r}")
            rows.append({
                "image": f"images/{line['image']}",
                "image_id": line["image"].rsplit("_", 1)[1].split(".")[0],
                "mode": mode,
                "object": asked.group(2),
                "phrase": asked.group(1),  # with its article: "an apple"
                "question": line["text"],
                "label": {"yes": "positive", "no": "negative"}[line["label"]],
            })
    return rows


def main() -> None:
    questions = {
        mode: read_questions(fetch(QUESTIONS.format(mode=mode), DATA / "questions" / f"coco_pope_{mode}.json").read_text(encoding="utf-8"))
        for mode in MODES
    }
    images = chosen_images(questions)
    for number, name in enumerate(images, 1):
        fetch(IMAGES.format(name=name), DATA / "images" / name)
        print(f"\rImages: {number}/{len(images)}", end="", flush=True)
    print()
    rows = table_rows(questions, images)
    with open(DATA / "pope_mini.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table: {len(rows)} questions about {len(images)} images -> {DATA / 'pope_mini.csv'}")


if __name__ == "__main__":
    sys.exit(main())
