"""The tables of the paper, from a finished run.

Usage, from packages/med:
    python paper.py configs/experiment.yaml configs/analysis.yaml   run, VLMShift report, then the tables
    python paper.py --run <run folder>                               the tables again, without any model

Rules of the paper (sections 3 and 4.2):
- a cell is a (source, concept) pair. A cell is kept only if every model has an answer there in
  every language: the complete-case rule. Balanced accuracy has its own core, the cells where it
  is defined everywhere, so with positives and negatives;
- an unreadable answer (INVALID) counts as wrong, never as no. A failed call (error) is not an answer;
- the gap with English is averaged over model × cell pairs. Its 95% interval comes from 10,000
  paired draws of these pairs: the seven languages of a pair are drawn together.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "EN"
DRAWS, SEED = 10_000, 0
CELL = ["site", "concept"]


def load_lines(run: Path) -> pd.DataFrame:
    """One row per answer, with its model, source, concept, language and label."""
    log = json.loads((run / "run_log.json").read_text(encoding="utf-8"))
    tables = run.parents[1] / "tables" / log["table"]
    variants = pd.read_parquet(tables / "variants" / log["variants"] / "variants.parquet")
    results = pd.read_parquet(run / "results.parquet")
    lines = results[results["status"] != "error"].merge(variants[["id", *CELL, "language", "label"]], on="id")
    return lines[lines["label"].isin(["positive", "negative"])]


def cell_scores(lines: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, F1, balanced accuracy and INVALID rate of each model × source × concept × language."""
    positive = lines["label"] == "positive"
    counts = pd.DataFrame({
        "pos": positive, "neg": ~positive, "yes": lines["decision"] == "yes",
        "tp": positive & (lines["decision"] == "yes"), "tn": ~positive & (lines["decision"] == "no"),
        "invalid": lines["status"] == "INVALID",
    })
    sums = counts.groupby([lines["model"], lines["site"], lines["concept"], lines["language"]]).sum()
    n = sums["pos"] + sums["neg"]
    return pd.DataFrame({
        "n": n,
        "acc": (sums["tp"] + sums["tn"]) / n,
        "f1": 2 * sums["tp"] / (sums["pos"] + sums["yes"]),  # 2 TP / (2 TP + FP + FN)
        "bacc": (sums["tp"] / sums["pos"] + sums["tn"] / sums["neg"]) / 2,  # empty without both classes
        "invalid_rate": sums["invalid"] / n,
    }).reset_index()


def complete(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The cells where every model has the metric in every language: the complete-case rule."""
    defined = scores.dropna(subset=[metric])
    size = scores["model"].nunique() * scores["language"].nunique()
    full = defined.groupby(CELL)["model"].transform("size") == size
    return defined[full]


def models_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Table 3: one row per model. Range: best minus worst language, in accuracy."""
    core, binary = complete(scores, "acc"), complete(scores, "bacc")
    by_language = core.groupby(["model", "language"])["acc"].mean().unstack()
    return pd.DataFrame({
        "acc": core.groupby("model")["acc"].mean(),
        "f1": core.groupby("model")["f1"].mean(),
        "bacc": binary.groupby("model")["bacc"].mean(),
        "language_range": by_language.max(axis=1) - by_language.min(axis=1),
    }).sort_values("bacc", ascending=False).reset_index()


def gaps(core: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of a metric by language, its gap with English and the 95% interval of that gap."""
    delta = f"delta_{metric}"
    columns = [metric, delta, f"{delta}_low", f"{delta}_high"]
    wide = core.pivot_table(index=["model", *CELL], columns="language", values=metric)
    if wide.empty or REFERENCE not in wide:
        return pd.DataFrame(columns=columns)
    differences = wide.sub(wide[REFERENCE], axis=0)
    draws = np.random.default_rng(SEED).integers(0, len(wide), size=(DRAWS, len(wide)))
    rows = {}
    for language in wide.columns:
        values = differences[language].to_numpy()
        low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
        rows[language] = [wide[language].mean(), values.mean(), low, high]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def languages_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Table 4: one row per language, accuracy on its core and balanced accuracy on its own."""
    accuracy = gaps(complete(scores, "acc"), "acc")
    balanced = gaps(complete(scores, "bacc"), "bacc")
    return accuracy.join(balanced, how="outer").rename_axis("language").reset_index()


def rankings(core: pd.DataFrame) -> pd.DataFrame:
    """Kendall τ between the model rankings of two languages, in each source (accuracy core)."""
    from scipy.stats import kendalltau

    rows = []
    for site, part in core.groupby("site"):
        accuracy = part.groupby(["language", "model"])["acc"].mean().unstack()
        for a, b in combinations(accuracy.index, 2):
            rows.append({"site": site, "language_a": a, "language_b": b,
                         "tau": kendalltau(accuracy.loc[a], accuracy.loc[b])[0]})
    return pd.DataFrame(rows, columns=["site", "language_a", "language_b", "tau"])


def tables(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core = complete(scores, "acc")
    return {
        "cells": scores,
        "matched_cells": core[CELL].drop_duplicates().reset_index(drop=True),
        "models": models_table(scores),
        "languages": languages_table(scores),
        "domain_language": core.groupby(["site", "language"])["acc"].mean().unstack().reset_index().rename_axis(columns=None),
        "rankings": rankings(core),
    }


def write_tables(run: Path) -> Path:
    """Write each table in CSV, and paper.md that shows them together."""
    folder = run / "paper"
    folder.mkdir(exist_ok=True)
    found = tables(cell_scores(load_lines(run)))
    text = ["# Tables of the paper", "", f"Run `{run.name}`. Every number per cell is in `cells.csv`.", ""]
    for name, table in found.items():
        table.to_csv(folder / f"{name}.csv", index=False)
        if name != "cells":
            text += [f"## {name}", "", "```text", table.to_string(index=False, float_format="{:.3f}".format), "```", ""]
    (folder / "paper.md").write_text("\n".join(text), encoding="utf-8")
    return folder


def main(argv: list[str]) -> None:
    from vlmshift.common import VLMShiftError

    try:
        if argv[0] == "--run":
            run = Path(argv[1])
        else:
            import adapters  # noqa: F401  (registers the Med models)
            from vlmshift.cli import do_analyze, do_run
            from vlmshift.config.load import load_analysis, load_experiment

            run = do_run(load_experiment(argv[0]))
            do_analyze(load_analysis(argv[1], str(run)))
        print(f"Paper tables -> {write_tables(run)}")
    except VLMShiftError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
