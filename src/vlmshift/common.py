"""Small tools shared by all steps."""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from io import StringIO
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1"
MIN_UNITS = 30  # below this number of independent units, a number is fragile
DEEP_MODES = ("I;16", "I;16B", "I;16L", "I", "F")  # images with more than 8 bits per pixel


def to_levels(column: pd.Series, observed: bool = True, basis: pd.Series | None = None) -> pd.Series:
    """Levels of an axis, the same in the audit, the coverage and the measures.

    An observed numeric axis with more than 4 values, such as an age, is cut into quartiles,
    named by rank then interval: `Q1 (4.999, 16.25]`. Sorted as text, as everywhere, they keep
    the order of the numbers. The quartiles are those of `basis`, the whole table: rows left out
    at step 3 do not move them. A controlled axis keeps the levels chosen by the experimenter.
    Empty stays (empty).
    """
    basis = column if basis is None else basis
    present = basis.dropna()
    numbers = pd.to_numeric(present, errors="coerce")
    levels = column.astype(object)
    if observed and numbers.notna().all() and present.nunique() > 4:
        edges = pd.qcut(numbers, 4, duplicates="drop", retbins=True)[1]
        values = pd.to_numeric(column, errors="coerce")
        quartiles = pd.cut(values, edges, include_lowest=True, duplicates="drop")
        names = [f"Q{rank} {interval}" for rank, interval in enumerate(quartiles.cat.categories, 1)]
        levels = quartiles.cat.rename_categories(names).astype(str)
    return levels.where(column.notna(), "(empty)")


class VLMShiftError(Exception):
    """Configuration or data error: the tool stops with a clear message."""


class Step(NamedTuple):
    """What a step hands to the next one: its table, its folder, its fingerprint."""

    df: pd.DataFrame
    dir: Path
    hash: str


# --- Fingerprints (section 7.4) ---


def sha16(text: str) -> str:
    """SHA-256 cut to 16 characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def canonical(value) -> str:
    """Sorted JSON without spaces: same content → same text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def nfc(text: str) -> str:
    """One Unicode form: two spellings of the same character become identical."""
    return unicodedata.normalize("NFC", text)


# --- Files ---


def write_parquet(df: pd.DataFrame, path: Path, fingerprint: str = "") -> None:
    """Write a table with its schema version and fingerprint in the metadata.

    We write a temporary file first: an interruption never leaves
    a half-written file.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {
        **(table.schema.metadata or {}),
        b"vlmshift.schema": SCHEMA_VERSION.encode(),
        b"vlmshift.fingerprint": fingerprint.encode(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), temporary)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file. Skip a last line cut short by an interruption.

    Lines end at a newline only: an answer may hold other line separators, such as U+2028.
    Each line is decoded alone, so a line cut inside a character loses that line, not the file.
    """
    if not path.exists():
        return []
    rows = []
    for line in path.read_bytes().split(b"\n"):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return rows


def read_csv_text(path: Path) -> pd.DataFrame:
    """A CSV as a spreadsheet may save it: UTF-8, or Windows-1252 from Excel; commas, or
    semicolons when the spreadsheet is set to French. The first line tells the separator."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as error:
            raise VLMShiftError(f"{path} is neither UTF-8 nor Windows-1252: save it as CSV UTF-8") from error
    header = text.split("\n", 1)[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    try:
        return pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, sep=separator)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise VLMShiftError(f"{path} cannot be read as a table (separator {separator!r}): {error}") from error


def write_csv(df: pd.DataFrame, path: Path, issues: Issues | None = None) -> None:
    """A copy to read in a spreadsheet. If that file is open, it is not updated, and the step goes on."""
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except PermissionError:
        message = f"{path.name} is open in another program: it was not updated. Close it and run the command again"
        if issues is None:
            print(f"  ! {message}")
        else:
            issues.warn("csv_locked", message)


def open_jsonl(path: Path):
    """Open a JSONL file to add lines. A last line cut by a stop is closed first:
    the next line must not be glued to it, or it would be lost too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cut = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as end:
            end.seek(-1, os.SEEK_END)
            cut = end.read(1) != b"\n"
    file = open(path, "a", encoding="utf-8")
    if cut:
        file.write("\n")
    return file


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Add lines to a JSONL file. Used as a cache: nothing is lost if everything stops."""
    with open_jsonl(path) as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


class JsonlWriter:
    """A JSONL file kept open while a model runs. Each batch is handed to the system at once.

    Opening and closing the file at each batch is slow on Windows: the antivirus reads
    the whole file again at each closing, so a long run slows down as its cache grows.
    """

    def __init__(self, path: Path):
        self.path, self.file = path, None

    def write(self, rows: list[dict]) -> None:
        if self.file is None:  # opened at the first write: nothing to write, no file
            self.file = open_jsonl(self.path)
        self.file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        self.file.flush()  # if the run is stopped, this batch is already in the file

    def close(self) -> None:
        if self.file:
            self.file.close()


# --- Issues ---


class Issues:
    """The issues of a step, meant for issues.parquet."""

    COLUMNS = ["step", "severity", "code", "id", "message"]

    def __init__(self, step: int):
        self.step = step
        self.rows: list[list] = []

    def warn(self, code: str, message: str, id: str | None = None) -> None:
        self.rows.append([self.step, "warning", code, id, message])

    def error(self, code: str, message: str, id: str | None = None) -> None:
        self.rows.append([self.step, "error", code, id, message])

    def errors(self) -> list[str]:
        return [row[4] for row in self.rows if row[1] == "error"]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)


def join_issues(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Join the issues of several steps."""
    filled = [frame for frame in frames if len(frame)]
    return pd.concat(filled, ignore_index=True) if filled else pd.DataFrame(columns=Issues.COLUMNS)
