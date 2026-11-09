"""Step 4: send each variant to each model and write the results table.

The raw outputs go into the model's cache, batch after batch. The results table
is then rebuilt from this cache: resuming and changing the reader are
therefore free, and an interrupted run still gives a partial table.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd

from ..common import (
    Issues, JsonlWriter, Step, VLMShiftError, append_jsonl, canonical, join_issues, read_jsonl, sha16,
    write_csv, write_parquet,
)
from ..config.load import config_hash, write_yaml
from ..extensions import get_adapter
from ..tasks import get_task
from ..tasks.llm_reader import LLMReader, validate_reader
from ..tasks.yes_no import DEFAULT_ANSWERS
from .base import Model, ModelError
from .run_log import library_versions, now, write_run_log

RESULT_COLUMNS = [
    "id", "model", "run", "response", "decision", "decision_source", "reader_response",
    "score_pos", "score_neg", "margin", "status", "message",
]
KIND_TEXTS = {"generative": ["prompt"], "contrastive": ["positive_prompt", "negative_prompt"]}
PROGRESS_SECONDS = 60  # a run can last hours: often enough to follow it, not enough to fill a log


# --- Following a long run ---


def duration(seconds: float) -> str:
    """A rough duration, easy to read."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


class Progress:
    """Print where a model stands. The errors are shown too: a broken model is seen early."""

    def __init__(self, name: str, total: int, errors: dict):
        self.name, self.total, self.errors, self.done = name, total, errors, 0
        self.known = len(errors)  # the errors of other tables share this dict
        self.started = self.shown = perf_counter()

    def add(self, count: int) -> None:
        self.done += count
        clock = perf_counter()
        if clock - self.shown >= PROGRESS_SECONDS:
            self.shown = clock
            left = (clock - self.started) / self.done * (self.total - self.done)
            print(f"  {self.name}: {self.done} of {self.total}, {len(self.errors) - self.known} errors, about {duration(left)} left", flush=True)


# --- Calling a model ---


def attempt(action, retries: int):
    """Run action. On a ModelError, try again up to `retries` times."""
    for remaining in range(retries, -1, -1):
        try:
            return action()
        except ModelError:
            if remaining == 0:
                raise


def batches(table: pd.DataFrame, size: int):
    for start in range(0, len(table), size):
        yield table.iloc[start:start + size]


def call_model(model: Model, batch: pd.DataFrame, params: dict) -> list[dict]:
    """A batch of variants → one raw output per variant."""
    images = batch["image"].tolist()
    if model.kind == "generative":
        answers = model.generate(images, batch["prompt"].tolist(), params)
        if not all(isinstance(answer, str) for answer in answers):
            raise ModelError("answer that is not text")
        outputs = [{"response": answer} for answer in answers]
    else:
        scores = model.score(images, batch["positive_prompt"].tolist(), batch["negative_prompt"].tolist())
        outputs = [{"score_pos": positive, "score_neg": negative} for positive, negative in scores]
    if len(outputs) != len(batch):
        raise ModelError(f"{len(batch)} variants sent, {len(outputs)} outputs received")
    return outputs


def question_ids(variants: pd.DataFrame, kind: str) -> list[str]:
    """What the model receives: the image and its texts. The same question is asked only once,
    whatever the table, the labels or the axes around it."""
    columns = ["image_hash", *KIND_TEXTS[kind]]
    return [sha16(canonical(list(row))) for row in variants[columns].itertuples(index=False)]


def run_batch(model, batch, params, retries, cache: JsonlWriter, errors: dict) -> None:
    """Process a batch. If it fails, redo it row by row to isolate the faulty row."""
    try:
        outputs = attempt(lambda: call_model(model, batch, params), retries)
    except ModelError as error:
        if len(batch) == 1:
            errors[batch["q"].iloc[0]] = str(error)
        else:
            for position in range(len(batch)):
                run_batch(model, batch.iloc[[position]], params, retries, cache, errors)
        return
    cache.write([{"q": q, **output} for q, output in zip(batch["q"], outputs)])


def call_embed(model: Model, batch: pd.DataFrame) -> list:
    vectors = model.embed(batch["image"].tolist())
    if len(vectors) != len(batch):
        raise ModelError(f"{len(batch)} images sent, {len(vectors)} vectors received")
    return vectors


def run_embed(model, batch, retries, cache: JsonlWriter, issues: Issues) -> None:
    try:
        vectors = attempt(lambda: call_embed(model, batch), retries)
    except ModelError as error:
        issues.warn("embed_error", f"{model.name}: {error}")
        return
    rows = zip(batch["image_hash"], vectors)
    cache.write([{"image_hash": image, "vector": vector} for image, vector in rows])


def run_model(model, wants_vectors, variants, cache_dir, params, retries, errors, issues) -> None:
    """Run a model on everything that is not yet in its cache."""
    outputs_file, vectors_file = cache_dir / "outputs.jsonl", cache_dir / "vectors.jsonl"
    questions = variants.drop_duplicates("q")
    todo = questions[~questions["q"].isin({row.get("q") for row in read_jsonl(outputs_file)})]
    images = variants.iloc[0:0]
    if wants_vectors:  # an image used by several rows is computed only once
        known = {row["image_hash"] for row in read_jsonl(vectors_file)}
        images = variants.drop_duplicates("image_hash")
        images = images[~images["image_hash"].isin(known)]
    if todo.empty and images.empty:
        return

    try:
        attempt(model.load, retries)
    except ModelError as error:
        model.unload()  # a load stopped half-way may already hold weights on the card
        errors.update(dict.fromkeys(todo["q"], f"Cannot load: {error}"))
        if not images.empty:
            issues.warn("embed_error", f"{model.name}: no vector, cannot load: {error}")
        return
    # Models can run in different environments: each one keeps the trace of its own.
    session = {"started_at": now(), "libraries": library_versions(), "notes": model.notes}
    append_jsonl(cache_dir / "sessions.jsonl", [session])
    progress = Progress(model.name, len(todo) + len(images), errors)
    outputs, vectors = JsonlWriter(outputs_file), JsonlWriter(vectors_file)
    try:
        for batch in batches(todo, model.batch_size):
            run_batch(model, batch, params, retries, outputs, errors)
            progress.add(len(batch))
        for batch in batches(images, model.batch_size):
            run_embed(model, batch, retries, vectors, issues)
            progress.add(len(batch))
    finally:
        outputs.close()
        vectors.close()
        model.unload()


# --- Reading the raw outputs ---


def read_response(response: str, task, answers, reader) -> dict:
    """Generative model: the list first, then the small LLM if the list finds nothing."""
    decision = task.read(response, answers) if answers else None
    if decision:
        return {"decision": decision, "decision_source": "list"}
    if not reader:
        return {"decision": None}
    decision, raw = reader.read(response)
    return {"decision": decision, "decision_source": "llm" if decision else None, "reader_response": raw}


def read_scores(positive: float, negative: float) -> dict:
    """Contrastive model: the best-scored phrase wins. Tie → no decision."""
    margin = positive - negative
    return {"decision": "yes" if margin > 0 else "no" if margin < 0 else None, "margin": margin}


def read_output(output: dict, task, answers, reader) -> dict:
    """A raw output → its decision, whatever the kind of model."""
    if "response" in output:
        return read_response(output["response"], task, answers, reader)
    return read_scores(output["score_pos"], output["score_neg"])


def result_rows(variants, questions, name, run_id, outputs, errors, task, answers, reader) -> list[dict]:
    """One row per variant. `questions` gives the question of each variant, the key of the cache."""
    rows = []
    for id, q in zip(variants["id"], questions):
        row = {"id": id, "model": name, "run": run_id}
        if q in outputs:
            output = {key: value for key, value in outputs[q].items() if key != "q"}
            read = read_output(output, task, answers, reader)
            row |= {**output, **read, "status": "valid" if read["decision"] else "INVALID"}
        elif q in errors:
            row |= {"status": "error", "message": errors[q]}
        else:
            continue  # interrupted run: this row has not been processed yet
        rows.append(row)
    return rows


# --- The complete run ---


def make_params(config, entry) -> dict:
    """The settings of models.params replace those of inference."""
    return {**config.inference.model_dump(), **entry.params}


def make_model(config, entry) -> Model:
    return get_adapter(entry.adapter)(entry.name, entry.id or entry.name, entry.revision, make_params(config, entry))


def chosen_models(config, only: list[str] | None) -> list:
    """The models of the experiment, or those of --only. A misspelled name stops everything."""
    names = [entry.name for entry in config.models]
    if unknown := sorted(set(only or []) - set(names)):
        raise VLMShiftError(f"--only: unknown model {unknown}. Models of the experiment: {names}")
    return [entry for entry in config.models if not only or entry.name in only]


def cache_key(config, entry) -> str:
    """Fingerprint of the model: the weights it loads and the settings it uses (section 7.4).

    The questions are keys inside the cache, so the table, the labels and the axes do not
    count: correcting a label or adding a model asks nothing again. The name counts when
    `id` is left out, since the name is then what is loaded. The number of retries is left
    out: it cannot change an answer.
    """
    params = {key: value for key, value in make_params(config, entry).items() if key != "retries"}
    return sha16(canonical([entry.adapter, entry.id or entry.name, entry.revision, params]))


def saved_errors(cache: Path) -> dict:
    """The questions that failed at their last try. With --only, the other models keep theirs."""
    return {row["q"]: row["message"] for row in read_jsonl(cache / "errors.jsonl") if "q" in row}


def save_errors(cache: Path, errors: dict) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"q": q, "message": message}, ensure_ascii=False) + "\n" for q, message in errors.items()]
    (cache / "errors.jsonl").write_text("".join(rows), encoding="utf-8")


def answer_lists(config, reader) -> dict | None:
    """List alone, small LLM alone, both, or default list (section 3)."""
    if config.task.answers:
        return config.task.answers.model_dump()
    return None if reader else DEFAULT_ANSWERS


def write_results(config, variants, run_id, folder, caches, errors, questions, reader, issues) -> None:
    """Rebuild results.parquet and embeddings.parquet from the caches."""
    task, answers = get_task(config.task.type), answer_lists(config, reader)
    images = set(variants.df["image_hash"])
    rows, vectors = [], []
    for entry in config.models:
        name, cache = entry.name, caches[entry.name]
        outputs = {row["q"]: row for row in read_jsonl(cache / "outputs.jsonl") if "q" in row}
        model_rows = result_rows(variants.df, questions[name], name, run_id, outputs, errors[cache], task, answers, reader)
        rows += model_rows
        for row in model_rows:
            if row["status"] == "error":
                issues.warn("model_error", f"{name}: {row['message']}", row["id"])
        for note in dict.fromkeys(note for run in read_jsonl(cache / "sessions.jsonl") for note in run["notes"]):
            issues.warn("model_note", f"{name}: {note}")
        if len(model_rows) < len(variants.df):
            issues.warn("model_pending", f"{name}: {len(variants.df) - len(model_rows)} variants not yet processed")
        if entry.embeddings:  # the cache may also hold the vectors of other tables
            found = {row["image_hash"]: row for row in read_jsonl(cache / "vectors.jsonl") if row["image_hash"] in images}
            vectors += [{"model": name, **row} for row in found.values()]
            if missing := len(images) - len(found):
                issues.warn("vectors_pending", f"{name}: {missing} of {len(images)} images have no vector yet: the probe leaves them out")
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    write_parquet(results, folder / "results.parquet", run_id)
    write_csv(results, folder / "results.csv", issues)  # to read the raw answers in a spreadsheet
    if vectors:
        write_parquet(pd.DataFrame(vectors), folder / "embeddings.parquet", run_id)
    earlier = pd.read_parquet(variants.dir / "issues.parquet")  # the issues of step 3
    write_parquet(join_issues([earlier, issues.frame()]), folder / "issues.parquet", run_id)


def run_inference(config, variants: Step, only: list[str] | None = None) -> Path:
    """Run the models, or only those in `only`. Return the run folder.

    The table also includes the answers already cached for the other models. Running each
    model in its own environment with `only` therefore gives a single run, complete at the end.
    """
    started_at = now()
    chosen = chosen_models(config, only)
    workspace = config.path(config.workspace)
    run_id = config_hash(config, variants.hash)
    folder = workspace / "runs" / run_id
    write_yaml(config.model_dump(), folder / "experiment.yaml")

    # Everything that can fail without a model is checked before the first inference.
    models = {entry.name: make_model(config, entry) for entry in config.models}
    for entry in config.models:
        if entry.embeddings and not models[entry.name].supports_embed:
            raise VLMShiftError(f"{entry.name}: embeddings: true, but no embed method")
    llm = config.task.reader_llm
    reader = LLMReader(llm, workspace) if llm else None
    if llm and llm.validation:
        validation = validate_reader(reader, config.path(llm.validation))
        write_parquet(validation, folder / "reader_validation.parquet", run_id)

    # The audit of this experiment goes with the run: another experiment on the same table
    # writes its own audit there, with other axes.
    audit = variants.dir.parents[1] / "audit"
    (folder / "audit").mkdir(exist_ok=True)
    for name in ("counts.parquet", "samples.parquet", "issues.parquet"):
        shutil.copyfile(audit / name, folder / "audit" / name)

    issues = Issues(step=4)
    # One cache per model in the workspace, shared by every table: a question already asked
    # is never asked again. Two entries that load the same model share their cache.
    caches = {e.name: workspace / "cache" / cache_key(config, e) for e in config.models}
    errors = {cache: saved_errors(cache) for cache in set(caches.values())}
    questions = {name: question_ids(variants.df, model.kind) for name, model in models.items()}
    started = []
    try:
        for entry in chosen:
            model, cache = models[entry.name], caches[entry.name]
            for q in questions[entry.name]:
                errors[cache].pop(q, None)  # its failed questions are asked again
            started.append(entry.name)
            run_model(model, entry.embeddings, variants.df.assign(q=questions[entry.name]), cache, model.params,
                      model.params["retries"], errors[cache], issues)
    finally:  # even if interrupted, we write what was obtained
        for cache in {caches[name] for name in started}:
            save_errors(cache, errors[cache])
        write_results(config, variants, run_id, folder, caches, errors, questions, reader, issues)
        write_run_log(folder / "run_log.json", {
            "run": run_id,
            "table": variants.dir.parents[1].name,
            "variants": variants.hash,
            "models": {
                entry.name: {
                    **models[entry.name].info(),
                    "adapter": entry.adapter,
                    "params": models[entry.name].params,
                    "cache": caches[entry.name].name,
                    "embeddings": entry.embeddings,
                    "sessions": read_jsonl(caches[entry.name] / "sessions.jsonl"),
                }
                for entry in config.models
            },
            "reader_llm": llm.model_dump() if llm else None,
            "inference": config.inference.model_dump(),
            "started_at": started_at,
        })
    if silent := [name for name in started if questions[name] and errors[caches[name]].keys() >= set(questions[name])]:
        first = errors[caches[silent[0]]][questions[silent[0]][0]]
        raise VLMShiftError(
            f"No answer from {silent}: every variant failed. First error: {first}\n"
            f"The run folder is written ({folder}). Fix the cause, then run the same command again."
        )
    return folder
