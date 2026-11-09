"""Run log: versions, dates, parameters, fingerprints."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

LIBRARIES = (
    "vlmshift", "numpy", "pandas", "pyarrow", "scikit-learn", "torch", "transformers", "accelerate", "bitsandbytes", "vlmeval",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass  # library not installed: nothing to note
    return versions


def write_run_log(path: Path, content: dict) -> None:
    log = {**content, "libraries": library_versions(), "finished_at": now()}
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
