"""Checks and edge cases of step 1.

Three severities in practice:
- raise VLMShiftError: immediate stop (the table cannot be read);
- issues.error: everything is checked and reported, then the import stops;
- issues.warn: reported, the import continues.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

from ..common import DEEP_MODES, Issues, VLMShiftError
from .ids import content

ROLES = {
    "image", "question", "label", "sample_id", "unit",
    "original_id", "positive_text", "negative_text",
}
RESERVED = ROLES | {
    "id", "image_hash", "prompt", "positive_prompt", "negative_prompt", "model", "run",
    "response", "decision", "decision_source", "reader_response",
    "score_pos", "score_neg", "margin", "status", "message",
}
LABELS = {"positive", "negative", "unknown"}
INTERNAL = {"_root", "_path"}  # added while reading: the folder of the file, the full path of the image


def require_columns(table: pd.DataFrame, columns, what: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise VLMShiftError(f"Columns not found for {what}: {missing}")


def apply_roles(table: pd.DataFrame, roles) -> pd.DataFrame:
    """Give the role columns their standard name (e.g. outcome → label).

    A column already named after a role, such as sample_id, takes that role. One column can
    serve two roles, such as a case number for sample_id and unit: each role gets a copy.
    """
    declared = {role: column for role, column in roles.model_dump().items() if column}
    for role in sorted(ROLES - set(declared)):
        if role in table.columns and role not in declared.values():
            declared[role] = role
    # An "id" column comes from the dataset: it is its original identifier, not ours.
    if "id" in table.columns and "id" not in declared.values():
        if "original_id" in declared:
            raise VLMShiftError(
                f"The table has an id column and an original identifier ({declared['original_id']}). "
                "VLMShift writes its own id: rename or remove the id column"
            )
        declared["original_id"] = "id"
    needed = [column for role, column in declared.items() if role != "question"]
    require_columns(table, needed, "the roles")  # the question is checked in step 3
    clash = (set(table.columns) - set(declared.values())) & RESERVED
    if clash:
        raise VLMShiftError(
            f"Reserved names used by free columns: {sorted(clash)}. Rename them, or declare them in roles if that is what they are"
        )
    first: dict[str, str] = {}
    for role, column in declared.items():
        first.setdefault(column, role)
    renamed = table.rename(columns=first)
    for role, column in declared.items():
        if role not in renamed.columns:  # the second role of a column
            renamed[role] = renamed[first[column]]
    return renamed


def add_image_hash(table: pd.DataFrame, issues: Issues) -> pd.DataFrame:
    """Add the full path (_path) and the fingerprint of each image."""
    from PIL import Image

    def is_deep(data: bytes) -> bool:
        """More than 8 bits per pixel. Only the header is read."""
        try:
            return Image.open(BytesIO(data)).mode in DEEP_MODES
        except Exception:  # not an image Pillow knows: the model will say so
            return False

    paths = [
        str(Path(root) / ("" if pd.isna(image) else image))
        for root, image in zip(table["_root"], table["image"])
    ]
    hashes, deep = {}, 0
    for path in set(paths):  # each file is read only once
        file = Path(path)
        data = file.read_bytes() if file.is_file() else None
        hashes[path] = hashlib.sha256(data).hexdigest() if data is not None else None
        deep += data is not None and is_deep(data)
    if deep:
        issues.warn(
            "image_16bit",
            f"{deep} images have more than 8 bits per pixel. Contrastive models scale them; a generative model "
            "may read them almost white. Convert them to 8 bits to be sure",
        )
    return table.drop(columns="_root").assign(_path=paths, image_hash=[hashes[path] for path in paths])


def drop_duplicates(table: pd.DataFrame, issues: Issues) -> pd.DataFrame:
    """Identical rows: only one is kept. They are compared as the id sees them: the same row
    read from two data folders, or written in two Unicode forms, is still the same row."""
    repeated = pd.Series(content(table, {"id", "_path"}), index=table.index).duplicated()
    for id in table.loc[repeated, "id"]:
        issues.warn("duplicate_line", "Row identical to another one: removed", id)
    return table[~repeated]


def check_collisions(table: pd.DataFrame, issues: Issues) -> None:
    """After removing duplicates, two rows with the same id have different contents."""
    for id in table.loc[table["id"].duplicated(), "id"]:
        issues.error("id_collision", "Two different contents have the same identifier", id)


def drop_missing_images(table: pd.DataFrame, issues: Issues) -> pd.DataFrame:
    missing = table["image_hash"].isna()
    for id, path in zip(table.loc[missing, "id"], table.loc[missing, "_path"]):
        issues.warn("image_missing", f"Image not found, row excluded: {path}", id)
    return table[~missing]


def check_labels(table: pd.DataFrame, issues: Issues) -> None:
    """An empty or unknown label is an error: never turned into negative."""
    wrong = ~table["label"].isin(LABELS)
    for id, label, path in zip(table.loc[wrong, "id"], table.loc[wrong, "label"], table.loc[wrong, "_path"]):
        shown = "empty" if pd.isna(label) else repr(label)
        issues.error("label_invalid", f"Invalid label ({shown}) for {path}: write positive, negative or unknown", id)


def check_original_id(table: pd.DataFrame, issues: Issues) -> None:
    if "original_id" not in table.columns:
        return
    wrong = table["original_id"].isna() | table["original_id"].duplicated(keep=False)
    for id in table.loc[wrong, "id"]:
        issues.warn("original_id_invalid", "Original identifier empty or duplicated", id)


def check_samples(table: pd.DataFrame, controlled: list[str], issues: Issues) -> None:
    samples = table.groupby("sample_id")
    for column, code in (("unit", "sample_many_units"), ("label", "sample_many_labels")):
        counts = samples[column].nunique()
        for sample in counts[counts > 1].index:
            issues.error(code, f"Sample {sample}: several values of {column}")
    if not controlled:  # no pairs to make
        return
    # Several questions about one sample at the same levels: which one pairs with which is unknown.
    same_levels = table.duplicated(["sample_id", *controlled], keep=False)
    for id in table.loc[same_levels, "id"]:
        issues.error(
            "same_levels",
            "Two rows of one sample at the same levels of the controlled axes: the pairs would be chosen at random. "
            "Add a column that tells these questions apart and stays the same at every level (concept_id, for example)",
            id,
        )


def check_axes(table: pd.DataFrame, axes: list[str], issues: Issues) -> None:
    for axis in axes:
        if table[axis].nunique() < 2:
            issues.warn("axis_single_level", f"Axis {axis}: only one level, no comparison")
