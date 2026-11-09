"""Check that everything is ready: data available, environments installed.

Usage, from packages/med: python check.py configs/experiment.yaml
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

from prepare import ACCESS, card_problems, folder_of, name_of, sources
from prepare.common import data_root

import adapters  # noqa: F401  (registers the Med adapters)
from vlmshift.config.load import load_experiment

# Libraries needed by each adapter.
LIBRARIES = {
    "vlmevalkit": ["vlmeval"],
    "contrastive": ["torch", "transformers"],
    "huggingface": ["torch", "transformers"],
    "open_clip": ["torch", "open_clip"],
    "chexzero": ["torch", "open_clip"],
}
CHEXZERO = ("Download it from the Google Drive folder linked from github.com/rajpurkarlab/CheXzero: "
            "https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno")


def digest(path: Path) -> str:
    """SHA-256 of the bytes of a file, read by blocks."""
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def fingerprint(path: Path) -> str:
    """SHA-256 of a raw file. For a folder, of the line "<folder>/<name> <SHA-256>" of each file, sorted:
    the name of its folder is the class, and the place where the archive was extracted does not matter."""
    if path.is_file():
        return digest(path)
    lines = sorted(f"{file.parent.name}/{file.name} {digest(file)}" for file in path.rglob("*")
                   if file.is_file() and not file.name.startswith("."))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def where_to_get(source, root: Path) -> str:
    """Everything needed to get a missing source, from its card."""
    card = source.SOURCE
    return (
        f"Raw data missing: {root / source.RAW}\n"
        f"  {card['name']}. Access: {ACCESS[card['access']]}. {card['steps']}\n"
        f"  Ask for it at {card['page']}, then put its files in {root / folder_of(source)}"
    )


def problems(experiment_path: str) -> list[str]:
    """The list of what is missing. Empty if everything is ready."""
    config = load_experiment(experiment_path)
    root = data_root()
    found = []
    for source in sources():
        found += card_problems(source) or ([] if (root / source.RAW).exists() else [where_to_get(source, root)])
        found += [f"{name_of(source)}: missing library: {library}" for library in getattr(source, "LIBRARIES", [])
                  if importlib.util.find_spec(library) is None]
    for model in config.models:
        for library in LIBRARIES.get(model.adapter, []):
            if importlib.util.find_spec(library) is None:
                found.append(f"{model.name}: missing library: {library}")
        if model.adapter == "chexzero" and not Path(model.id).exists():
            found.append(f"{model.name}: weights missing: {Path(model.id).resolve()}. {CHEXZERO}")
    return found


def other_versions() -> list[str]:
    """Raw files that are not the version of their card. The run can go on, but its numbers may differ.

    The message gives the SHA-256 of the file: for a new source, it is the one to write in its card.
    """
    root = data_root()
    found = []
    for source in sources():
        path = root / source.RAW
        if path.exists() and (found_digest := fingerprint(path)) != source.SOURCE.get("sha256"):
            found.append(
                f"{name_of(source)}: {path} is not the version of the card, "
                f"so the numbers may differ from those of the benchmark. Its SHA-256: {found_digest}"
                if source.SOURCE.get("sha256") else
                f"{name_of(source)}: its card has no SHA-256 yet. Write this one in it: {found_digest}"
            )
    return found


if __name__ == "__main__":
    missing = problems(sys.argv[1])
    for line in [*missing, *other_versions()]:
        print(line)
    if not missing:
        print("Everything is ready.")
    sys.exit(1 if missing else 0)
