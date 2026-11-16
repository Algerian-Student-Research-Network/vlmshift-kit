"""Reading the input files (CSV, Parquet or JSONL)."""
from pathlib import Path

import pandas as pd

from ..common import VLMShiftError, read_csv_text, read_jsonl
from .checks import INTERNAL


def read_file(path: Path) -> pd.DataFrame:
    """Read a file. Everything becomes text; an empty cell becomes null."""
    if not path.exists():
        raise VLMShiftError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):  # one row per line, as {"image": ..., "label": ...}
        rows = read_jsonl(path)
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise VLMShiftError(f"{path} must hold one row per line, written as keys and values")
        table = pd.DataFrame(rows)
    else:
        table = pd.read_parquet(path) if suffix == ".parquet" else read_csv_text(path)
    if internal := sorted(INTERNAL & set(table.columns)):
        raise VLMShiftError(f"Reserved names used by free columns in {path.name}: {internal}. Rename them")
    return table.astype("string").replace("", pd.NA).reset_index(drop=True)


def read_table(paths: list[Path]) -> pd.DataFrame:
    """Join the data files end to end (same columns expected).

    Each row keeps the folder of its file (_root) to find its images.
    """
    frames = [read_file(path).assign(_root=str(path.parent)) for path in paths]
    for path, frame in zip(paths[1:], frames[1:]):
        if differ := sorted(set(frame.columns) ^ set(frames[0].columns)):
            raise VLMShiftError(f"{path.name} and {paths[0].name} must have the same columns. Different: {differ}")
    return pd.concat(frames, ignore_index=True)
