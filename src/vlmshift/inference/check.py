"""Before a long run: ask each model a few real questions and show what comes back.

In a few minutes you see what would otherwise appear hours later: no graphics card, a
model that does not load, answers the reader does not understand. The answers are not
kept: the run asks these questions again, so the check changes no result.
"""
from __future__ import annotations

from time import perf_counter

import pandas as pd

from ..common import Step, VLMShiftError
from ..tasks import get_task
from ..tasks.llm_reader import LLMReader
from .base import ModelError
from .run_log import library_versions
from .runner import answer_lists, attempt, call_model, chosen_models, duration, make_model, read_output

QUESTIONS = 4  # enough to see the levels of the axes, without waiting
SHOWN_ANSWER = 200  # characters: a long answer already says what it needs to


def graphics_cards() -> list[str]:
    """The NVIDIA cards that torch can use."""
    try:
        import torch
    except (ImportError, OSError):  # OSError: a torch whose DLLs do not load, common on Windows
        return []
    cards = [torch.cuda.get_device_properties(number) for number in range(torch.cuda.device_count())]
    return [f"{card.name} ({card.total_memory / 1e9:.0f} GB)" for card in cards]


def show_machine() -> None:
    """The first thing to look at when a model does not load."""
    versions = library_versions()
    names = ("python", "torch", "transformers", "vlmeval")
    print("Graphics cards:", ", ".join(graphics_cards()) or "none seen by torch")
    print("Libraries:", ", ".join(f"{name} {versions[name]}" for name in names if name in versions))


def sample_rows(variants: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """A few rows that differ on the axes, so that each level is seen as far as possible."""
    return variants.drop_duplicates(axes).head(QUESTIONS)


def ask(model, row: pd.DataFrame, params: dict, task, answers, reader) -> tuple[bool, float]:
    """Send one row and print what happened. Return (answer understood, seconds)."""
    started = perf_counter()
    output = call_model(model, row, params)[0]
    seconds = perf_counter() - started
    decision = read_output(output, task, answers, reader)["decision"]
    line = row.iloc[0]
    if model.kind == "generative":
        print(f"  Question: {line['prompt']}")
        print(f"  Answer:   {output['response'][:SHOWN_ANSWER]!r}")
    else:
        print(f"  Phrases:  {line['positive_prompt']!r} {output['score_pos']:.3f}, {line['negative_prompt']!r} {output['score_neg']:.3f}")
    print(f"  Read as:  {decision or 'NOT UNDERSTOOD'} ({seconds:.1f} s)")
    return decision is not None, seconds


def check_model(model, rows: pd.DataFrame, total: int, params: dict, task, answers, reader) -> bool:
    """Return True if the model answered. Each question is tried as often as in the run."""
    print(f"{model.name} ({model.id})")
    started, results = perf_counter(), []
    try:
        attempt(model.load, params["retries"])
        print(f"  Loaded in {duration(perf_counter() - started)}")
        for note in model.notes:
            print(f"  ! model_note: {note}")
        for position in range(len(rows)):
            try:
                results.append(attempt(lambda: ask(model, rows.iloc[[position]], params, task, answers, reader), params["retries"]))
            except ModelError as error:
                print(f"  FAILED: {error}")
    except ModelError as error:
        print(f"  FAILED: {error}")
        return False
    finally:
        model.unload()
    if len(results) < len(rows):
        print("  ! Some questions failed: in the run, these rows would get the error status.")
    if not results:
        return False
    if not all(understood for understood, _ in results):
        print("  ! Rows like these would be INVALID: complete task.answers, or raise max_new_tokens if the answer is cut.")
    seconds = results[-1][1]  # the first call is often slower
    print(f"  All {total} variants: about {duration(seconds * total / model.batch_size)}")
    return True


def check_models(config, variants: Step, only: list[str] | None = None) -> None:
    """Try each model in turn, even after a failure: one look shows every problem."""
    if variants.df.empty:
        raise VLMShiftError("No variant to ask: read the messages above")
    show_machine()
    llm = config.task.reader_llm
    reader = LLMReader(llm, config.path(config.workspace)) if llm else None
    task, answers = get_task(config.task.type), answer_lists(config, reader)
    rows = sample_rows(variants.df, list(config.axes))
    failed = []
    for entry in chosen_models(config, only):
        model = make_model(config, entry)
        if not check_model(model, rows, len(variants.df), model.params, task, answers, reader):
            failed.append(entry.name)
    if failed:
        raise VLMShiftError(f"No answer from {failed}: read the messages above before a long run")
