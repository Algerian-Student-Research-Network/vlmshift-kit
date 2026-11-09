"""Schemas of experiment.yaml and analysis.yaml (section 7.1).

Each block is a class; each key is a typed attribute with its default value.
An unknown key causes an error: a typo does not go unnoticed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ..extensions import adapter_kind
from ..tasks.base import check_template, holes
from ..tasks.yes_no import clean


class Block(BaseModel):
    # validate_default: a default value and the same value written by hand
    # must give exactly the same fingerprint.
    model_config = ConfigDict(extra="forbid", validate_default=True)


class Root(Block):
    """A configuration file. Its paths are relative to its folder."""

    _dir: Path = PrivateAttr(default=Path("."))

    def path(self, value: str) -> Path:
        return (self._dir / value).resolve()


# --- experiment.yaml ---


class Roles(Block):
    image: str = "image"
    question: str = "question"
    label: str = "label"
    sample_id: str | None = None
    unit: str | None = None
    original_id: str | None = None
    positive_text: str | None = None
    negative_text: str | None = None


class Axis(Block):
    type: Literal["controlled", "observed"]


class Answers(Block):
    yes: list[str] = Field(min_length=1)
    no: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def no_overlap(self):
        # Compared as the reader compares them: "Yes." and "yes" are the same answer.
        both = {clean(a) for a in self.yes} & {clean(a) for a in self.no}
        if both:
            raise ValueError(f"answers in both yes and no: {sorted(both)}")
        return self


class ReaderLLM(Block):
    model: str
    revision: str | None = None
    validation: str | None = None


class Contrastive(Block):
    positive_template: str
    negative_template: str

    @model_validator(mode="after")
    def valid_templates(self):
        check_template(self.positive_template)
        check_template(self.negative_template)
        return self


class TaskConfig(Block):
    type: Literal["yes_no"]
    template: str | None = None
    answers: Answers | str | None = None  # a path is replaced by its content when read
    reader_llm: ReaderLLM | None = None
    contrastive: Contrastive | None = None

    @model_validator(mode="after")
    def valid_template(self):
        if self.template:
            check_template(self.template)
        return self

    def columns(self) -> set[str]:
        """The columns the templates ask for, whatever the models. Like the question, they are
        text of the question: a word translated in each language changes between the variants."""
        texts = [self.template]
        if self.contrastive:
            texts += [self.contrastive.positive_template, self.contrastive.negative_template]
        return {hole for text in texts if text for hole in holes(text)}


class ModelConfig(Block):
    name: str
    adapter: str
    id: str | None = None
    revision: str | None = None
    params: dict = {}
    embeddings: bool = False


class Inference(Block):
    temperature: float = Field(0, ge=0)
    max_new_tokens: int = Field(16, gt=0)
    retries: int = Field(2, ge=0)


class ExperimentConfig(Root):
    schema_version: Literal[1]
    data: str | list[str]
    workspace: str = "./workspace"
    roles: Roles = Field(default_factory=Roles)
    axes: dict[str, Axis] = Field(min_length=1)
    task: TaskConfig
    models: list[ModelConfig] = Field(min_length=1)
    inference: Inference = Field(default_factory=Inference)

    @model_validator(mode="after")
    def check_rules(self):
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("two models have the same name")
        has_columns = self.roles.positive_text and self.roles.negative_text
        for model in self.models:
            contrastive = adapter_kind(model.adapter) == "contrastive"
            if contrastive and not (self.task.contrastive or has_columns):
                raise ValueError(
                    f"contrastive model {model.name}: task.contrastive is required, "
                    "or the roles positive_text and negative_text"
                )
        return self


# --- analysis.yaml ---


class AxisSettings(Block):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # reference: 1

    reference: str | None = None
    levels: list[str] | None = None

    @model_validator(mode="after")
    def reference_in_levels(self):
        if self.levels and self.reference and self.reference not in self.levels:
            raise ValueError(f"the reference {self.reference} is not in levels")
        return self


class Measure(Block):
    name: str
    compare: str
    split_by: list[str] = []

    @model_validator(mode="after")
    def no_common_axis(self):
        if self.compare in self.split_by:
            raise ValueError(f"measure {self.name}: compare and split_by share an axis")
        return self


class Stats(Block):
    bootstrap: int = Field(2000, ge=100)
    alpha: float = Field(0.05, gt=0, lt=1)
    seed: int = 0
    min_units: int = Field(30, ge=1)


class Latent(Block):
    targets: list[str] = Field(min_length=1)
    folds: int = Field(5, ge=2)


class Report(Block):
    notice: str | None = None


class AnalysisConfig(Root):
    schema_version: Literal[1]
    run: str
    axes: dict[str, AxisSettings] = {}
    measures: list[Measure] | None = None
    stats: Stats = Field(default_factory=Stats)
    latent: Latent | None = None
    report: Report = Field(default_factory=Report)

    @model_validator(mode="after")
    def unique_measures(self):
        names = [measure.name for measure in self.measures or []]
        if len(names) != len(set(names)):
            raise ValueError("two measures have the same name")
        return self


def export_json_schemas(folder: Path) -> None:
    """Export the schemas as JSON Schema (documentation, editor autocompletion)."""
    for name, cls in (("experiment", ExperimentConfig), ("analysis", AnalysisConfig)):
        text = json.dumps(cls.model_json_schema(), indent=2, ensure_ascii=False)
        (folder / f"{name}.schema.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    export_json_schemas(Path(sys.argv[1]))
