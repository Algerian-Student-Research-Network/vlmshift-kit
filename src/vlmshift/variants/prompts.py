"""Write the exact text sent to the models: prompt, positive_prompt, negative_prompt."""
from __future__ import annotations

import pandas as pd

from ..common import Issues, VLMShiftError
from ..data.checks import require_columns
from ..extensions import get_adapter
from ..tasks.base import Task, holes


def templates_for(config) -> dict[str, str]:
    """The templates that are needed: only those of the model kinds present."""
    kinds = {get_adapter(model.adapter).kind for model in config.models}
    templates = {}
    if "generative" in kinds:
        templates["prompt"] = config.task.template or "{question}"
    if "contrastive" in kinds:
        # Columns come before templates: this is how the language is varied.
        columns = config.roles.positive_text and config.roles.negative_text
        phrases = config.task.contrastive
        if not (columns or phrases):
            raise VLMShiftError("Contrastive model without positive and negative phrases")
        templates["positive_prompt"] = "{positive_text}" if columns else phrases.positive_template
        templates["negative_prompt"] = "{negative_text}" if columns else phrases.negative_template
    return templates


def add_prompts(table: pd.DataFrame, config, templates: dict[str, str], task: Task, issues: Issues) -> pd.DataFrame:
    columns = sorted({hole for template in templates.values() for hole in holes(template)})
    roles = {column: role for role, column in config.roles.model_dump().items() if column and column != role}
    if renamed := [f"{{{roles[hole]}}} instead of {{{hole}}}" for hole in columns if hole in roles and hole not in table]:
        raise VLMShiftError(f"In the templates, a role column keeps its standard name: write {', '.join(renamed)}")
    require_columns(table, columns, "the templates")

    # An empty value prevents filling the template: the row is excluded and reported.
    empty = table[columns].isna().any(axis=1)
    for id in table.loc[empty, "id"]:
        issues.warn("empty_template_value", "Empty value to fill the template: row excluded", id)

    variants = table[~empty].reset_index(drop=True)
    if variants.empty:
        unfilled = [column for column in columns if table[column].isna().any()]
        raise VLMShiftError(
            "No variant left: every row lacks a value to fill the templates. Check the columns they ask for "
            f"({', '.join(unfilled or columns)})"
        )
    rows = variants.to_dict("records")
    variants["prompt"] = variants["positive_prompt"] = variants["negative_prompt"] = None
    if "prompt" in templates:
        variants["prompt"] = [task.build_prompt(row, templates["prompt"]) for row in rows]
    if "positive_prompt" in templates:
        pairs = [
            task.build_contrastive(row, templates["positive_prompt"], templates["negative_prompt"])
            for row in rows
        ]
        variants["positive_prompt"] = [positive for positive, _ in pairs]
        variants["negative_prompt"] = [negative for _, negative in pairs]
    return variants
