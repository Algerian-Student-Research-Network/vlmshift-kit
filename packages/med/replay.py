"""Replay the recorded answers of the paper through the analysis, without calling any model.

Usage, from packages/med, after plans.py:
    python replay.py main                 one arm, from the prediction files in data/paper/preds/
    python replay.py factorial --allow-unvalidated

The study kept every answer of every model (pred__<model>__core.csv and ..._core_supp.csv, in
preds/by_model/ on its drive). This script puts them where a run of Core would have written
them, then runs the analysis and the report of Core on them. It needs neither the images nor a
GPU: it is how the numbers of the paper are computed again, and how the analysis is checked
against them.

The images are not read, so the table is written without the image checks of step 1, and each
image is identified by its path. The report says that it is a replay.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from bank import gate
from plans import as_table

import adapters  # noqa: F401  (registers the Med models)
from vlmshift.audit.counts import run_audit
from vlmshift.cli import do_analyze
from vlmshift.common import Issues, Step, VLMShiftError, canonical, sha16, write_parquet
from vlmshift.config.load import config_hash, load_analysis, load_experiment, write_yaml
from vlmshift.data import checks
from vlmshift.data.ids import row_ids
from vlmshift.data.io import read_table
from vlmshift.extensions import get_adapter
from vlmshift.inference.run_log import write_run_log
from vlmshift.inference.runner import RESULT_COLUMNS
from vlmshift.variants.plan import prepare_variants

HERE = Path(__file__).parent
# The name of each model in the files of the study → its name in configs/experiment.yaml.
MODELS = {
    "biomedclip": "biomedclip", "medsiglip": "medsiglip", "chexzero": "chexzero", "clip_vitl14": "clip-vit-l14",
    "siglip_so400m": "siglip-so400m", "medgemma_4b": "medgemma-4b", "lingshu_7b": "lingshu-7b", "chexone": "chexone",
    "llava_med": "llava-med", "qwen25vl_7b": "qwen2.5-vl-7b",
}
MATCH = ["sample_id", "image", "question", "positive_text"]  # what tells one question of an arm from another


def table_without_images(config) -> Step:
    """Step 1 of Core, without opening any image: each one is identified by its path."""
    issues = Issues(step=1)
    table = checks.apply_roles(read_table([config.path(config.data)]), config.roles).drop(columns="_root")
    table["image_hash"] = [hashlib.sha256(str(path).encode("utf-8")).hexdigest() for path in table["image"]]
    table["id"] = row_ids(table)
    checks.check_labels(table.assign(_path=table["image"]), issues)
    if errors := issues.errors():
        raise VLMShiftError("Table rejected:\n- " + "\n- ".join(errors))
    issues.warn("replay", "Replay of recorded answers: the images were not read, and are identified by their path")
    table = table.sort_values("id").reset_index(drop=True)
    fingerprint = sha16(canonical(table["id"].tolist()))
    folder = config.path(config.workspace) / "tables" / fingerprint
    write_parquet(issues.frame(), folder / "issues.parquet", fingerprint)
    write_parquet(table, folder / "table.parquet", fingerprint)
    return Step(table, folder, fingerprint)


def recorded(folder: Path, roots: dict) -> pd.DataFrame:
    """Every recorded answer found in a folder: the columns of the arm tables, then what the model gave."""
    frames = []
    for path in sorted(folder.glob("pred__*.csv")):
        answers = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", pd.NA)
        for column, default in (("script_variant", "native"), ("term_condition", "true")):
            answers[column] = answers[column].fillna(default) if column in answers else default
        answers["shown_concept"] = answers["shown_concept"].fillna(answers["pathology"]) if "shown_concept" in answers else answers["pathology"]
        given = answers[["model", "raw_response", "pred", "score_yes", "score_no", "valid", "error"]].reset_index(drop=True)
        asked = as_table(answers, roots).rename(columns={"sample": "sample_id"})[MATCH].reset_index(drop=True)
        frames.append(pd.concat([asked, given], axis=1))
    if not frames:
        raise VLMShiftError(f"No prediction file (pred__<model>__core.csv) in {folder}: copy them from preds/by_model/ on the drive of the study")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["model", *MATCH])


def results_of(variants: pd.DataFrame, answers: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """The recorded answers as the results table of Core. An answer the study could not read is INVALID."""
    found = variants[["id", *MATCH]].merge(answers, on=MATCH)
    said = pd.to_numeric(found["pred"], errors="coerce")  # written 1 or 1.0 depending on the file
    valid = pd.to_numeric(found["valid"], errors="coerce").eq(1) & said.isin([0, 1])
    failed = found["error"].notna()
    yes, no = pd.to_numeric(found["score_yes"], errors="coerce"), pd.to_numeric(found["score_no"], errors="coerce")
    results = pd.DataFrame({
        "id": found["id"], "model": found["model"].map(MODELS), "run": run_id, "response": found["raw_response"],
        "decision": said.map({1: "yes", 0: "no"}).where(valid), "decision_source": pd.Series("recorded", index=found.index).where(valid),
        "reader_response": None, "score_pos": yes, "score_neg": no, "margin": yes - no,
        "status": valid.map({True: "valid", False: "INVALID"}).where(~failed, "error"), "message": found["error"],
    })
    return results[results["model"].notna()][RESULT_COLUMNS]


def replay(arm: Path, predictions: Path, roots: dict) -> Path:
    """Write the run of an arm from recorded answers. Return the run folder."""
    config = load_experiment(arm / "experiment.yaml")
    table = table_without_images(config)
    run_audit(config, table)
    variants = prepare_variants(config, table)
    run_id = config_hash(config, variants.hash, "replay")
    results = results_of(variants.df, recorded(predictions, roots), run_id)
    if results.empty:
        raise VLMShiftError(f"No recorded answer matches the questions of {arm.name}: are these the prediction files of this arm?")
    folder = config.path(config.workspace) / "runs" / run_id
    present = set(results["model"])
    config.models = [model for model in config.models if model.name in present]  # only the models that were recorded
    write_yaml(config.model_dump(), folder / "experiment.yaml")
    write_parquet(results, folder / "results.parquet", run_id)
    (folder / "audit").mkdir(exist_ok=True)
    for name in ("counts.parquet", "samples.parquet", "columns.parquet", "issues.parquet"):
        shutil.copyfile(table.dir / "audit" / name, folder / "audit" / name)
    issues = Issues(step=4)
    for model, part in results.groupby("model"):
        if missing := len(variants.df) - len(part):
            issues.warn("model_pending", f"{model}: {missing} variants have no recorded answer")
    write_parquet(pd.concat([pd.read_parquet(variants.dir / "issues.parquet"), issues.frame()], ignore_index=True),
                  folder / "issues.parquet", run_id)
    write_run_log(folder / "run_log.json", {
        "run": run_id, "table": table.hash, "variants": variants.hash, "replay": str(predictions),
        "models": {model.name: {"name": model.name, "id": model.id, "revision": model.revision, "adapter": model.adapter,
                                "kind": get_adapter(model.adapter).kind, "params": model.params, "sessions": []}
                   for model in config.models},
    })
    return folder


def main(argv: list[str]) -> None:
    allowed = "--allow-unvalidated" in argv
    arms = [word for word in argv if not word.startswith("--")]
    data = HERE / "data" / "paper"
    try:
        if not arms:
            raise VLMShiftError("Name the arm to replay, such as: python replay.py main")
        notice = gate(allowed)
        roots = json.loads((data / "image_roots.json").read_text(encoding="utf-8")) if (data / "image_roots.json").exists() else {}
        for arm in arms:
            run = replay(data / "arms" / arm, data / "preds", roots)
            analysis = load_analysis(data / "arms" / arm / "analysis.yaml", str(run))
            said = ["REPLAY: recorded answers of the study, no model was called.", notice, analysis.report.notice]
            analysis.report.notice = " ".join(filter(None, said))
            print(f"{arm}: report -> {do_analyze(analysis)}")
    except VLMShiftError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
