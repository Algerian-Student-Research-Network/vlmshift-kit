"""Reading the input files (CSV or Parquet)."""
from pathlib import Path

import pandas as pd

from ..common import VLMShiftError, read_csv_text
from .checks import INTERNAL


def read_file(path: Path) -> pd.DataFrame:
    """Read a file. Everything becomes text; an empty cell becomes null."""
    if not path.exists():
        raise VLMShiftError(f"File not found: {path}")
    table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else read_csv_text(path)
    if internal := sorted(INTERNAL & set(table.columns)):
        raise VLMShiftError(f"Reserved names used by free columns in {path.name}: {internal}. Rename them")
    return table.astype("string").replace("", pd.NA)


def read_table(paths: list[Path]) -> pd.DataFrame:
    """Join the data files end to end (same columns expected).

    Each row keeps the folder of its file (_root) to find its images.
    """
    frames = [read_file(path).assign(_root=str(path.parent)) for path in paths]
    for path, frame in zip(paths[1:], frames[1:]):
        if differ := sorted(set(frame.columns) ^ set(frames[0].columns)):
            raise VLMShiftError(f"{path.name} and {paths[0].name} must have the same columns. Different: {differ}")
    return pd.concat(frames, ignore_index=True)
