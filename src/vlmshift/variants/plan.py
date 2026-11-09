"""Step 3: prepare exactly what will be sent to the models."""
from ..common import Issues, Step, canonical, sha16, write_csv, write_parquet
from ..data.checks import require_columns
from ..tasks import get_task
from .prompts import add_prompts, templates_for


def prepare_variants(config, table: Step) -> Step:
    require_columns(table.df, config.axes, "the axes")
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
