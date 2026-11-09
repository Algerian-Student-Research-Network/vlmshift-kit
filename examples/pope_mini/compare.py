"""Put the numbers measured on POPE mini next to those published in the POPE paper.

    python examples/pope_mini/compare.py <path of report.md>

Give the path that `vlmshift all` prints on its last line (the analysis folder works too).
Two tables are printed and written next to report.md:

- pope_comparison.csv: for each model of the paper that was run, accuracy, recall, specificity and
  yes ratio in each mode, measured (with its 95% interval) and published;
- pope_drops.csv: how many points of accuracy each model loses from random to popular and to
  adversarial, measured for every model that was run, and published for the five of the paper.

Read them with three facts in mind. POPE mini has 100 of the 500 images of the paper: the
interval says how far the measure can move from one set of images to another. It is the plain 95%
interval, a little narrower than in the report, which corrects for the number of comparisons. Each mode has as
many present objects as absent ones, so balanced accuracy is the accuracy of the paper. And the
paper reads the answers in its own way: papers that run the same model often differ by a few points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference.csv"
METRICS = {"ba": "accuracy", "sens": "recall", "spec": "specificity", "yes_rate": "yes_ratio"}
MODES = ["random", "popular", "adversarial"]


def published(reference: pd.DataFrame) -> pd.DataFrame:
    """The paper gives accuracy and recall; with as many positives as negatives,
    specificity follows: accuracy is the mean of recall and specificity."""
    return reference.assign(specificity=(2 * reference["accuracy"] - reference["recall"]).round(2))


ADAPTERS = ("huggingface", "vlmevalkit")  # the columns of reference.csv that name a model for VLMShift


def paper_names(experiment: dict, reference: pd.DataFrame) -> dict[str, str]:
    """Run model name → name in the paper, for the models of the paper, by their id for the adapter."""
    known = {
        (adapter, row[adapter]): row["model"]
        for adapter in ADAPTERS for row in reference.to_dict("records") if isinstance(row[adapter], str)
    }
    # Without id, the name is what the adapter loads, as in the runner.
    asked = {model["name"]: (model["adapter"], model.get("id") or model["name"]) for model in experiment["models"]}
    return {name: known[key] for name, key in asked.items() if key in known}


def whole(metrics: pd.DataFrame) -> pd.DataFrame:
    """The rows of the measure mode, on all the data, from the valid answers."""
    return metrics[metrics["measure"].eq("mode") & metrics["split"].isna() & metrics["mode"].eq("valid")]


def measured(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per model, mode and metric, in points (0 to 100)."""
    rows = whole(metrics)
    rows = rows[rows["level_b"].isna() & rows["metric"].isin(METRICS)]
    return pd.DataFrame({
        "model": rows["model"],
        "mode": rows["level_a"],
        "metric": rows["metric"].map(METRICS),
        "measured": (100 * rows["value"]).round(1),
        "low": (100 * rows["ci95_low"]).round(1),
        "high": (100 * rows["ci95_high"]).round(1),
    })


def comparison(found: pd.DataFrame, paper: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """The measures of the models of the paper, next to the published value."""
    published_long = paper.melt(id_vars=["model", "mode"], value_vars=list(METRICS.values()), var_name="metric", value_name="published")
    mine = found[found["model"].isin(names)].assign(paper_model=lambda table: table["model"].map(names))
    table = mine.merge(published_long.rename(columns={"model": "paper_model"}), on=["paper_model", "mode", "metric"])
    table["inside"] = table["published"].between(table["low"], table["high"])
    return order(table)[["model", "paper_model", "mode", "metric", "measured", "low", "high", "published", "inside"]]


def drops(metrics: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    """Points of accuracy lost from random to each other mode: measured, then published."""
    rows = whole(metrics)
    rows = rows[rows["metric"].eq("delta_ba") & rows["level_a"].eq("random")]
    mine = pd.DataFrame({
        "model": rows["model"],
        "source": "measured",
        "mode": rows["level_b"],
        "drop": (-100 * rows["value"]).round(1),
        "low": (-100 * rows["ci95_high"]).round(1),
        "high": (-100 * rows["ci95_low"]).round(1),
    })
    accuracy = paper.pivot(index="model", columns="mode", values="accuracy")
    theirs = pd.DataFrame([
        {"model": model, "source": "paper", "mode": mode, "drop": round(accuracy.loc[model, "random"] - accuracy.loc[model, mode], 1)}
        for model in paper["model"].unique() for mode in MODES[1:]
    ])
    return order(pd.concat([mine, theirs], ignore_index=True), first="source")


def order(table: pd.DataFrame, first: str = "model") -> pd.DataFrame:
    rank = table["mode"].map(MODES.index)
    return table.assign(rank=rank).sort_values([first, "model", "rank"], kind="stable").drop(columns="rank").reset_index(drop=True)


def analysis_folder(path: Path) -> Path | None:
    """From report.md, its report/ folder or the analysis folder, find the analysis folder:
    runs/<run>/analyses/<analysis>/. report/ holds a copy of metrics.parquet too, so the run's
    experiment.yaml, two levels up, is what tells them apart."""
    for folder in (path, *path.parents[:2]):
        if (folder / "metrics.parquet").is_file() and (folder.parent.parent / "experiment.yaml").is_file():
            return folder
    return None


def main(argv: list[str]) -> int:
    folder = analysis_folder(Path(argv[0]).resolve()) if len(argv) == 1 else None
    if folder is None:
        print(__doc__.split("\n\n")[1].strip(), file=sys.stderr)
        print("No metrics.parquet there: give the report.md printed by `vlmshift all`.", file=sys.stderr)
        return 2
    # A column with no estimable number at all is read back as text: make it numbers again.
    metrics = pd.read_parquet(folder / "metrics.parquet").astype({"value": float, "ci95_low": float, "ci95_high": float})
    experiment = YAML(typ="safe").load((folder.parent.parent / "experiment.yaml").read_text(encoding="utf-8"))
    paper = published(pd.read_csv(REFERENCE))
    names = paper_names(experiment, paper)
    report = folder / "report"
    report.mkdir(exist_ok=True)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        if names:
            table = comparison(measured(metrics), paper, names)
            table.to_csv(report / "pope_comparison.csv", index=False)
            print(f"Measured and published, in points ({table['inside'].sum()} of {len(table)} published values inside the interval):\n")
            print(table.to_string(index=False), "\n")
        else:
            print("No model of the paper in this run: only the drops can be compared.\n")
        lost = drops(metrics, paper)
        lost.to_csv(report / "pope_drops.csv", index=False)
        print("Accuracy lost from random, in points:\n")
        print(lost.to_string(index=False, na_rep=""))
    print(f"\nTables -> {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
