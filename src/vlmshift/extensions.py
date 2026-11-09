"""Adapter registry: a name (written in models.adapter) → a Model class."""
from __future__ import annotations

from .common import VLMShiftError
from .inference.adapters.contrastive import ContrastiveModel
from .inference.adapters.fake import FakeContrastive, FakeGenerative
from .inference.adapters.huggingface import HuggingFaceModel
from .inference.adapters.vlmevalkit import VLMEvalKitModel

_ADAPTERS = {
    "vlmevalkit": VLMEvalKitModel,
    "huggingface": HuggingFaceModel,
    "contrastive": ContrastiveModel,
    "fake_generative": FakeGenerative,
    "fake_contrastive": FakeContrastive,
}


def register_adapter(name: str, cls: type) -> None:
    """Make a model usable in experiment.yaml (e.g. called by the Med package)."""
    _ADAPTERS[name] = cls


def get_adapter(name: str) -> type:
    if name not in _ADAPTERS:
        raise VLMShiftError(f"Unknown adapter: {name}. Known adapters: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]


def adapter_kind(name: str) -> str | None:
    """Kind of an adapter, or None if it is not registered (e.g. re-reading a run to analyze it)."""
    return _ADAPTERS[name].kind if name in _ADAPTERS else None
