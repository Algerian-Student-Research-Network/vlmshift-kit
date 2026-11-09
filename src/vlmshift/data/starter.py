"""A starting point: a table and two configuration files, made from a folder of images."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..common import VLMShiftError
from .checks import INTERNAL, LABELS, RESERVED

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TABLE, EXPERIMENT, ANALYSIS = "table.csv", "experiment.yaml", "analysis.yaml"


def parse_questions(items: list[str]) -> dict[str, str]:
    """Turn ["EN=Is there...?", "FR=Y a-t-il...?"] into one question per level."""
    questions = {}
    for item in items:
        level, separator, text = (part.strip() for part in item.partition("="))
        if not (separator and level and text):
            raise VLMShiftError(f"--question must look like LEVEL=text, not: {item}")
        if level in questions:
            raise VLMShiftError(f"--question: level given twice: {level}")
        questions[level] = text
    if len(questions) < 2:
        raise VLMShiftError("Give at least two --question, one per level to compare")
    return questions


def find_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise VLMShiftError(f"Folder not found: {folder}")
    images = sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_TYPES)
    if not images:
        raise VLMShiftError(f"No image found in {folder}. Known types: {sorted(IMAGE_TYPES)}")
    return images


def label_of(image: Path, folder: Path) -> str:
    """A first subfolder named positive, negative or unknown, in any case, gives the label of its images."""
    parents = image.relative_to(folder).parts[:-1]
    return parents[0].lower() if parents and parents[0].lower() in LABELS else "unknown"


def check_axis(axis: str) -> None:
    """The axis becomes a column and a YAML key: a simple name, not used by another column."""
    if not axis.isidentifier():  # letters of any alphabet, digits and _, not starting with a digit
        raise VLMShiftError(
            f"--axis must be a simple name, with letters, digits and _, not starting with a digit, such as language: not {axis!r}"
        )
    if axis in RESERVED | INTERNAL | {"subject"}:
        raise VLMShiftError(f"--axis {axis}: this name is already used by the table. Choose another one, such as language")


def experiment_text(axis: str) -> str:
    return f"""# Made by vlmshift init. Read it and adapt it.
schema_version: 1
data: {TABLE}

roles:
  unit: subject                # images of one subject are not independent

axes:
  {axis}: {{type: controlled}}  # the condition you compare

task:
  type: yes_no
  answers:                     # add the words for yes and no in each language
    yes: ["yes"]
    no: ["no"]

models:                        # this model needs an NVIDIA card: see the documentation for others
  - {{name: smolvlm, adapter: vlmevalkit, id: SmolVLM-256M}}
"""


def analysis_text(axis: str, reference: str) -> str:
    return f"""# Made by vlmshift init. Read it and adapt it.
schema_version: 1
run: runs/to-replace           # filled in for you by vlmshift all

axes:
  {axis}: {{reference: {json.dumps(reference, ensure_ascii=False)}}}  # compare everything to this level
"""


def write_starter(images: Path, axis: str, questions: dict[str, str]) -> pd.DataFrame:
    """Write the three files next to the images folder. An existing file is never replaced."""
    check_axis(axis)
    folder = images.resolve()
    target = folder.parent
    taken = [name for name in (TABLE, EXPERIMENT, ANALYSIS) if (target / name).exists()]
    if taken:
        raise VLMShiftError(f"Already in {target}: {taken}. Nothing was changed.")
    table = pd.DataFrame([
        {
            "image": path.relative_to(target).as_posix(),
            "question": question,
            "label": label_of(path, folder),
            axis: level,
            "subject": path.relative_to(folder).as_posix(),  # one subject per image file, to adapt
        }
        for path in find_images(folder)
        for level, question in questions.items()
    ])
    table.to_csv(target / TABLE, index=False, encoding="utf-8")
    (target / EXPERIMENT).write_text(experiment_text(axis), encoding="utf-8")
    (target / ANALYSIS).write_text(analysis_text(axis, next(iter(questions))), encoding="utf-8")
    return table
