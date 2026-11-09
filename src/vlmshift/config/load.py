"""Read the YAML files, check them, compute their fingerprint."""
from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ..common import VLMShiftError, canonical, sha16
from .schema import AnalysisConfig, Block, ExperimentConfig, Root


def _yaml() -> YAML:
    # Read as YAML 1.2: yes and no stay text, without quotes.
    return YAML(typ="safe", pure=True)


def read_yaml(path: Path) -> dict:
    """Read a YAML file of keys and values. Any problem stops with a clear message."""
    try:
        with open(path, encoding="utf-8") as file:
            content = _yaml().load(file) or {}
    except OSError as error:
        raise VLMShiftError(f"Cannot read {path}: {error}") from error
    except YAMLError as error:  # a missing space or a bad indent, for example
        raise VLMShiftError(f"{path} is not valid YAML. Check the indentation near this place:\n{error}") from error
    if not isinstance(content, dict):
        raise VLMShiftError(f"{path} must contain keys and values, such as schema_version: 1")
    return content


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        _yaml().dump(data, file)


def problems(error: ValidationError) -> str:
    """One line per problem: the key, what is wrong, the value written. Where a key accepts two
    forms, such as answer lists or a file name, the form you did not use is left out."""
    found = error.errors(include_url=False)

    def other_form(problem) -> bool:
        return problem["loc"][-1:] in (("str",), ("list[str]",))

    def keys(cls: type) -> set[str]:  # every key of the files, to suggest the right one after a typo
        return set(cls.model_fields) | {key for sub in cls.__subclasses__() for key in keys(sub)}

    def explain(problem: dict) -> str:
        if problem["type"] == "extra_forbidden":
            close = get_close_matches(str(problem["loc"][-1]), sorted(keys(Block)), n=1)
            return "unknown key" + (f": did you mean {close[0]}?" if close else "")
        if problem["type"] in ("model_type", "dict_type"):  # a word where a block of keys was expected
            return "should be a block of keys and values, written on the following lines"
        return problem["msg"].removeprefix("Value error, ")

    lines = []
    for problem in found:
        where = problem["loc"]
        if other_form(problem) and any(
            other["loc"][:len(where) - 1] == where[:-1] and not other_form(other) for other in found
        ):
            continue
        place = ".".join(str(part) for part in where if "[" not in str(part) and part != "str") or "the file"
        value = problem.get("input")
        wrote = f" (you wrote: {value!r})" if value is None or isinstance(value, (str, int, float)) else ""
        lines.append(f"- {place}: {explain(problem)}{wrote}")
    return "\n".join(lines)


def _build(cls: type[Root], raw: dict, path: Path) -> Root:
    try:
        config = cls.model_validate(raw)
    except ValidationError as error:
        raise VLMShiftError(f"{path} is not valid:\n{problems(error)}") from error
    config._dir = path.resolve().parent
    return config


def load_experiment(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = read_yaml(path)
    # Answer lists in a separate file: we copy them here. This way, editing
    # that file changes the fingerprint, and the frozen copy of the run depends on nothing else.
    task = raw.get("task")
    answers = task.get("answers") if isinstance(task, dict) else None  # a wrong task is reported below
    if isinstance(answers, str):
        task["answers"] = read_yaml(path.parent / answers)
    return _build(ExperimentConfig, raw, path)


def load_analysis(path: str | Path, run: str | None = None) -> AnalysisConfig:
    """`run` replaces the key of the same name (used by --run and by `vlmshift all`)."""
    raw = read_yaml(Path(path))
    if run:  # typed in the terminal: relative to the current folder, if it exists there
        raw["run"] = str(Path(run).resolve()) if Path(run).is_dir() else run
    return _build(AnalysisConfig, raw, Path(path))


def run_path(config: AnalysisConfig) -> Path:
    """Run folder: relative to analysis.yaml, or to its workspace/ folder."""
    for folder in (config.path(config.run), config.path("workspace") / config.run):
        if folder.is_dir():
            return folder
    raise VLMShiftError(f"Run not found: {config.run}")


def config_hash(config: Root, *extra: str) -> str:
    """Fingerprint of a configuration, paths excluded (section 7.4)."""
    content = config.model_dump(exclude={"data", "workspace", "run"})
    return sha16(canonical([content, *extra]))
