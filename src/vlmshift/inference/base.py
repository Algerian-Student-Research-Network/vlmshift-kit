"""Model interface: what a model must provide (section 7.3)."""
from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from typing import Literal

from ..common import VLMShiftError


class ModelError(Exception):
    """Technical failure of a model. The engine tries again, then sets the status error."""


@contextmanager
def guard():
    """Turn any failure of an external library into a ModelError."""
    try:
        yield
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(f"{type(error).__name__}: {error}") from error


class Model:
    """Base class. A generative model writes generate, a contrastive one writes score.

    Images are passed by their path. Lists have the same length, results come in the same order.
    """

    kind: Literal["generative", "contrastive"]
    supports_embed: bool = False
    batch_size: int = 8

    def __init__(self, name: str, id: str, revision: str | None, params: dict):
        self.name, self.id, self.revision, self.params = name, id, revision, params
        self.batch_size = params.get("batch_size", self.batch_size)
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise VLMShiftError(f"{name}: params.batch_size must be a whole number of at least 1, not {self.batch_size}")
        self.model = None
        self.notes: list[str] = []  # what the run log and the report must say about this model

    def info(self) -> dict:
        """Written to run_log.json."""
        return {"name": self.name, "id": self.id, "revision": self.revision, "kind": self.kind}

    def load(self) -> None:
        pass

    def unload(self) -> None:
        """Free the memory before moving on to the next model."""
        self.model = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, images: list[str], prompts: list[str], params: dict) -> list[str]:
        raise NotImplementedError

    def score(
        self, images: list[str], positives: list[str], negatives: list[str]
    ) -> list[tuple[float, float]]:
        raise NotImplementedError

    def embed(self, images: list[str]) -> list[list[float]]:
        raise NotImplementedError
