"""Step 3: prepare exactly what will be sent to the models."""
from ..common import Issues, Step, VLMShiftError, canonical, sha16, write_csv, write_parquet
from ..data.checks import require_columns
from ..tasks import get_task
from .prompts import add_prompts, templates_for


def check_answer_sets(config, table) -> None:
    """Each row names the answer list that reads it. A row without one could only be INVALID."""
    column = config.task.answers_by
    if not column:
        return
    require_columns(table, [column], "task.answers_by")
    known = set(config.task.answer_sets)
    if unknown := sorted(set(table[column].fillna("(empty)")) - known):
        raise VLMShiftError(
            f"task.answer_sets has no list for these values of the column {column}: {unknown}. Lists declared: {sorted(known)}"
        )


def prepare_variants(config, table: Step) -> Step:
    require_columns(table.df, config.axes, "the axes")
    check_answer_sets(config, table.df)
    issues = Issues(step=3)
    templates = templates_for(config)
    variants = add_prompts(table.df, config, templates, get_task(config.task.type), issues)

    # Only the templates used count. The answer lists do not change the texts sent, but a
    # model of another kind receives other texts: two experiments never share a folder by mistake.
    fingerprint = sha16(canonical([table.hash, templates]))
    folder = table.dir / "variants" / fingerprint
    write_parquet(variants, folder / "variants.parquet", fingerprint)
    write_csv(variants, folder / "variants.csv", issues)  # to read the exact texts in a spreadsheet
    write_parquet(issues.frame(), folder / "issues.parquet", fingerprint)
    return Step(variants, folder, fingerprint)
