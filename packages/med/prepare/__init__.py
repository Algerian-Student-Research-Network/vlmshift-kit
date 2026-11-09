"""Data preparation: one module per source, next to common.py."""
import importlib
import pkgutil
import re
from pathlib import Path

# How a source is obtained, from the simplest to the strictest.
ACCESS = {
    "open": "free download",
    "registration": "an account on the site of the source",
    "agreement": "a data use agreement to sign",
    "credentialed": "an identity check and a training before access, as on PhysioNet",
}
# The card of a source, SOURCE in its script: what anyone needs to find it, check its version and cite it.
CARD = ("name", "page", "access", "steps", "license", "bibtex")
# sha256: of RAW, None until someone has the data (check.py prints it). note: what a user must know.
OPTIONAL = ("sha256", "note")


def sources() -> list:
    """All the source scripts of this folder, but the template. Adding a file is enough to add a source."""
    names = [info.name for info in pkgutil.iter_modules(__path__) if info.name not in ("common", "__main__", "template")]
    return [importlib.import_module(f"{__name__}.{name}") for name in names]


def name_of(source) -> str:
    """The name of a source: that of its script."""
    return source.__name__.rsplit(".", 1)[-1]


def folder_of(source) -> Path:
    """Where the files of a source go, under the data root. RAW is a file, or a folder when it has no suffix."""
    raw = Path(source.RAW)
    return raw.parent if raw.suffix else raw


def card_problems(source) -> list[str]:
    """What is missing from the card of a source. Empty if it is complete."""
    card = getattr(source, "SOURCE", {})
    found = [f"{name_of(source)}: SOURCE has no {field}" for field in CARD if not card.get(field)]
    found += [f"{name_of(source)}: SOURCE has an unknown field {field}" for field in card if field not in CARD + OPTIONAL]
    if card.get("access") and card["access"] not in ACCESS:
        found.append(f"{name_of(source)}: SOURCE access must be one of {list(ACCESS)}, not {card['access']!r}")
    if card.get("sha256") and not re.fullmatch(r"[0-9a-f]{64}", card["sha256"]):
        found.append(f"{name_of(source)}: SOURCE sha256 is not a SHA-256. check.py gives it once the file is there")
    return found
