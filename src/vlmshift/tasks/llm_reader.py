"""Reading answers with a small local LLM, with its safeguards (section 3)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ..common import VLMShiftError, append_jsonl, canonical, read_csv_text, read_jsonl, sha16
from ..config.schema import ReaderLLM
from ..inference.adapters.fake import fake_reader
from .yes_no import clean

PROMPT = (
    "Classify this answer to a yes/no question. "
    "Reply with exactly one word: yes, no or unclear.\nAnswer: {response}\nClassification:"
)


def load_llm(config: ReaderLLM) -> Callable[[str], str]:
    """Load the small LLM. Return a text → text function (temperature 0)."""
    if config.model == "fake_reader":
        return fake_reader
    try:
        from transformers import pipeline

        generator = pipeline("text-generation", model=config.model, revision=config.revision)
    except Exception as error:
        raise VLMShiftError(f"Cannot load the small LLM: {config.model} ({error})") from error

    def ask(prompt: str) -> str:
        output = generator(prompt, max_new_tokens=4, do_sample=False, return_full_text=False)
        return output[0]["generated_text"]

    return ask


class LLMReader:
    """Sort a raw answer into yes, no or unclear. Each answer is asked only once."""

    def __init__(self, config: ReaderLLM, workspace: Path):
        self.ask = load_llm(config)
        self.cache_path = workspace / "reader_cache" / f"{sha16(canonical(config.model_dump()))}.jsonl"
        self.cache = {row["response"]: row["output"] for row in read_jsonl(self.cache_path)}

    def read(self, response: str) -> tuple[str | None, str]:
        """Return (decision or None, raw answer of the LLM)."""
        if response not in self.cache:
            self.cache[response] = self.ask(PROMPT.format(response=response))
            append_jsonl(self.cache_path, [{"response": response, "output": self.cache[response]}])
        output = self.cache[response]
        words = output.split()
        first = clean(words[0]) if words else ""
        return (first if first in ("yes", "no") else None), output


def validate_reader(reader: LLMReader, path: Path) -> pd.DataFrame:
    """Measure the LLM's accuracy, language by language, on a hand-labeled set."""
    if not path.exists():
        raise VLMShiftError(f"reader_llm.validation file not found: {path}")
    table = read_csv_text(path)
    missing = {"response", "language", "expected"} - set(table.columns)
    if missing:
        raise VLMShiftError(f"Columns missing from {path}: {sorted(missing)}")
    expected = [clean(value) for value in table["expected"]]
    if wrong := sorted(set(expected) - {"yes", "no", "unclear"}):
        raise VLMShiftError(f"{path}: the expected column holds yes, no or unclear, not {wrong}")
    found = [reader.read(response)[0] or "unclear" for response in table["response"]]
    table["correct"] = [a == b for a, b in zip(found, expected)]
    grouped = table.groupby("language")["correct"]
    return grouped.agg(n_responses="size", accuracy="mean").reset_index()
