"""Step 6: the report. Tables (CSV and Parquet) and a readable summary (report.md).

Everything is read from the files of the previous steps: no number is recomputed
in another way, so they stay identical to those of the metrics table.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ..common import MIN_UNITS, join_issues, to_levels, write_csv, write_parquet
from ..config.load import load_experiment, read_yaml

NOT_ESTIMABLE = "not estimable"
SHOWN_ISSUES = ["model_pending", "vectors_pending", "model_note", "probe_skipped", "axis_empty"]  # they change how the numbers are read
LEFT_OUT = ["image_missing", "duplicate_line", "empty_template_value"]  # rows that never reached the models
HEADLINES = ["delta_ba", "delta_yes_rate", "chg", "delta_inv_rate"]  # the summary speaks about these
MOST_FINDINGS = 30  # per model: the others stay in metrics.csv
FEW_VALID = 0.5  # below this share of valid answers, a level is named in the summary


# --- The tables ---


def show(row: dict) -> str:
    """One number: value, corrected interval, exploratory interval, counts.

    The * sign marks a significant shift: the meaning never depends on a color.
    """
    if pd.isna(row["value"]):
        return f"{NOT_ESTIMABLE} (n={row['n_units']})"
    star = "*" if row["significant"] is True else ""
    return (
        f"{row['value']:.3f}{star} [{row['ci_low']:.3f}; {row['ci_high']:.3f}]"
        f" (exploratory 95% [{row['ci95_low']:.3f}; {row['ci95_high']:.3f}])"
        f" n={row['n_units']} units"
    )


def matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """One measure as a cross table: one row per cell, one column per level or pair."""
    table = metrics.assign(
        split=metrics["split"].fillna("(all)"),
        column=[a if pd.isna(b) else f"{a} -> {b}" for a, b in zip(metrics["level_a"], metrics["level_b"])],
    )
    pivot = table.pivot(index=["model", "metric", "mode", "split"], columns="column", values="display")
    return pivot.reset_index().rename_axis(columns=None)


def coverage(
    variants: pd.DataFrame, results: pd.DataFrame, models: list[str], axes: list[str], controlled=(), table=None,
) -> pd.DataFrame:
    """For each model, in total then by level of each axis: rows expected, obtained, and missing.

    A model can fail on one level only, such as a language it does not speak: the total would hide it.
    Quartiles are those of the table, as in the audit and the measures.
    """
    groups = [("(all)", variants)]
    for axis in axes:
        levels = to_levels(variants[axis], axis not in controlled, None if table is None else table[axis])
        groups += [(f"{axis} = {level}", part) for level, part in variants.groupby(levels)]
    rows = []
    for model in models:
        status = results.loc[results["model"] == model].set_index("id")["status"]
        for level, part in groups:
            counts = status.reindex(part["id"]).value_counts()
            valid, invalid, error = (int(counts.get(name, 0)) for name in ("valid", "INVALID", "error"))
            rows.append({
                "model": model, "level": level, "expected": len(part), "valid": valid, "invalid": invalid,
                "error": error, "missing": len(part) - valid - invalid - error, "coverage": valid / max(len(part), 1),
            })
    return pd.DataFrame(rows)


def reader_share(lines: pd.DataFrame, axes: list[str], controlled, table: pd.DataFrame) -> pd.DataFrame:
    """Share of decisions made by the small LLM, by model and by level (safeguard)."""
    decided = lines[lines["decision_source"].notna()]
    rows = []
    for axis in axes:
        levels = to_levels(decided[axis], axis not in controlled, table[axis])
        for (model, level), group in decided.groupby(["model", levels]):
            share = (group["decision_source"] == "llm").mean()
            rows.append({"model": model, "axis": axis, "level": level, "n_decisions": len(group), "llm_share": share})
    return pd.DataFrame(rows, columns=["model", "axis", "level", "n_decisions", "llm_share"])


def load_inputs(analysis_dir: Path) -> SimpleNamespace:
    """Read the outputs of steps 1 to 5."""
    run_dir = analysis_dir.parents[1]
    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    table_dir = run_dir.parents[1] / "tables" / log["table"]
    audit_dir = run_dir / "audit"  # the audit of this experiment, kept with the run
    validation = run_dir / "reader_validation.parquet"
    return SimpleNamespace(
        log=log,
        analysis=analysis_dir.name,
        experiment=load_experiment(run_dir / "experiment.yaml"),
        notice=read_yaml(analysis_dir / "analysis.yaml")["report"]["notice"],
        table=pd.read_parquet(table_dir / "table.parquet"),
        variants=pd.read_parquet(table_dir / "variants" / log["variants"] / "variants.parquet"),
        results=pd.read_parquet(run_dir / "results.parquet"),
        metrics=pd.read_parquet(analysis_dir / "metrics.parquet"),
        probes=pd.read_parquet(analysis_dir / "probes.parquet"),
        counts=pd.read_parquet(audit_dir / "counts.parquet"),
        samples=pd.read_parquet(audit_dir / "samples.parquet"),
        validation=pd.read_parquet(validation) if validation.exists() else pd.DataFrame(),
        issues=join_issues([
            pd.read_parquet(path)
            for path in (table_dir / "issues.parquet", audit_dir / "issues.parquet",
                         run_dir / "issues.parquet", analysis_dir / "issues.parquet")
        ]),
    )


def build_tables(data: SimpleNamespace) -> dict[str, pd.DataFrame]:
    metrics = data.metrics.assign(display=[show(row) for row in data.metrics.to_dict("records")])
    models = [model.name for model in data.experiment.models]
    controlled = [axis for axis, spec in data.experiment.axes.items() if spec.type == "controlled"]
    lines = data.results.merge(data.variants, on="id")
    axes = list(data.experiment.axes)
    return {
        "metrics": metrics,
        "probes": data.probes,
        "audit_counts": data.counts,
        "coverage": coverage(data.variants, data.results, models, axes, controlled, data.table),
        "reader_share": reader_share(lines, axes, controlled, data.table),
        "reader_validation": data.validation,
        "issues": data.issues,
        **{f"matrix_{name}": matrix(part) for name, part in metrics.groupby("measure")},
    }


# --- The readable summary ---


def block(table: pd.DataFrame) -> str:
    """A table that reads well in report.md."""
    return "```text\n" + table.to_string(index=False) + "\n```" if len(table) else "Nothing to report."


def finding(row: dict) -> str:
    """One significant shift, in plain words."""
    place = f"{row['measure']}, {row['level_a']} to {row['level_b']}"
    if pd.notna(row["split"]):
        place += f", {row['split']}"
    value, up = row["value"], row["value"] > 0
    number = "{:.0%}" if row["metric"] == "chg" else "{:.3f}"
    size = number.format(abs(value))
    words = {
        "delta_ba": f"balanced accuracy {'rises' if up else 'falls'} by {size}",
        "delta_yes_rate": f"the model says yes {'more' if up else 'less'} often, by {size}",
        "delta_inv_rate": f"unreadable answers {'rise' if up else 'fall'} by {size}",
        "chg": f"{size} of the answers changed",
    }[row["metric"]]
    likely = f"likely between {number.format(row['ci_low'])} and {number.format(row['ci_high'])}"
    return f"- {place}: {words} ({likely}, {row['n_units']} units)."


def weak_levels(covered: pd.DataFrame) -> list[str]:
    """The levels where most answers cannot be used: their numbers rest on few rows, or on none."""
    weak = covered[covered["level"].ne("(all)") & (covered["coverage"] < FEW_VALID)]
    return [
        f"- {row.model}, {row.level}: {row.valid} valid answer(s) out of {row.expected} "
        f"({row.invalid} INVALID, {row.error} failed, {row.missing} not asked yet)."
        for row in weak.itertuples()
    ]


def unmeasured(metrics: pd.DataFrame) -> list[str]:
    """The measures with shifts that could not be estimated: the reader must know what is missing."""
    shifts = metrics[metrics["level_b"].notna() & metrics["mode"].eq("valid")]
    lines = []
    for (model, measure), part in shifts.groupby(["model", "measure"]):
        if missing := int(part["value"].isna().sum()):
            lines.append(f"- {model}, {measure}: {missing} of {len(part)} shift numbers.")
    return lines


def summary_section(data, metrics: pd.DataFrame, covered: pd.DataFrame) -> list[str]:
    """The main significant shifts, in plain sentences. Every number stays in metrics.csv."""
    significant = metrics[metrics["significant"].eq(True)]
    usual = significant["mode"].eq("valid") | significant["metric"].eq("delta_inv_rate")  # it only exists in strict mode
    main = significant[significant["metric"].isin(HEADLINES) & usual]
    tested = int(metrics["significant"].notna().sum())
    lines = [
        f"Run `{data.log['run']}`, analysis `{data.analysis}`, {len(data.experiment.models)} model(s), "
        f"{len(data.variants)} variants.",
        f"{len(significant)} significant shift(s) out of {tested} tested.",
        "",
    ]
    if weak := weak_levels(covered):
        lines += ["**Few usable answers** at these levels. Read some raw answers in `results.csv`, in the run folder, before the numbers:", "", *weak, ""]
    if missing := unmeasured(metrics):
        lines += [
            "**Not estimable**, for lack of units, of complete samples or of valid answers "
            "(the threshold is `stats.min_units`):", "", *missing, "",
        ]
    if not tested:  # "no significant shift" would suggest that something was measured
        return [*lines, "No shift could be tested: every number is not estimable."]
    if main.empty:
        return [*lines, "No significant shift among the main measures."]
    for model, part in main.groupby("model"):
        ordered = part.sort_values("value", key=abs, ascending=False)
        lines += [f"**{model}**", "", *[finding(row) for row in ordered.head(MOST_FINDINGS).to_dict("records")], ""]
        if len(ordered) > MOST_FINDINGS:
            lines += [f"And {len(ordered) - MOST_FINDINGS} more in `metrics.csv`.", ""]
    if (main["n_units"] < MIN_UNITS).any():
        lines += [f"**Few units**: some of these numbers rest on fewer than {MIN_UNITS} units. Read them with care.", ""]
    lines.append("All the details are in `metrics.csv`.")
    return lines


def audit_section(data) -> list[str]:
    samples = data.samples
    return [
        f"{len(samples)} samples, from {samples['n_variants'].min()} to {samples['n_variants'].max()} "
        f"variants per sample, {int(samples['missing'].notna().sum())} incomplete.",
        "",
        block(data.counts),
    ]


def probe_section(probes: pd.DataFrame) -> list[str]:
    if probes.empty:
        return ["No probe computed."]
    shown = probes.astype(object).where(probes.notna(), NOT_ESTIMABLE)
    return [block(shown), "A probe that succeeds shows that the information is present, not that the model uses it."]


def reading_section(llm, shares: pd.DataFrame, validation: pd.DataFrame) -> list[str]:
    if not llm:
        return ["No small LLM declared."]
    checked = block(validation) if len(validation) else "**Reader not validated**: no labeled set provided."
    return [
        f"Small LLM: {llm.model}.", "", "Share of decisions made by the LLM:", block(shares), "",
        "LLM accuracy by language:", checked,
    ]


def coverage_section(covered: pd.DataFrame, issues: pd.DataFrame) -> list[str]:
    incomplete = (covered["missing"] > 0) | (covered["error"] > 0)  # any level will do: the total is in it too
    warning = ["**Partial coverage**: incomplete run or failures.", ""] if incomplete.any() else []
    shown = issues.loc[issues["code"].isin(SHOWN_ISSUES), "message"]
    details = [*[f"- {message}" for message in shown], ""] if len(shown) else []
    left = issues["code"].value_counts().reindex(LEFT_OUT).dropna().astype(int)
    before = [
        f"**Left out before the models**: {left.sum()} row(s) "
        f"({', '.join(f'{code}: {count}' for code, count in left.items())}). The table above does not count them.",
        "",
    ] if len(left) else []
    return [block(covered), "", *warning, *before, *details, f"{len(issues)} issue(s) reported: see `issues.csv`."]


def render(data, tables: dict) -> str:
    measures = sorted(data.metrics["measure"].unique())
    sections = {
        "Summary": summary_section(data, tables["metrics"], tables["coverage"]),
        "Audit": audit_section(data),
        "Results by measure": [
            *[f"- `matrix_{name}.csv`: measure {name}" for name in measures],
            "- `metrics.csv`: all the numbers, with intervals and counts",
        ],
        "Latent space probe": probe_section(data.probes),
        "Reading the answers": reading_section(
            data.experiment.task.reader_llm, tables["reader_share"], data.validation
        ),
        "Coverage limits": coverage_section(tables["coverage"], data.issues),
    }
    lines = ["# VLMShift report", "", *([f"> {data.notice}", ""] if data.notice else [])]
    if data.issues["code"].isin(["model_pending", "vectors_pending"]).any():
        lines += ["> **Partial run**: some models did not finish, see \"Coverage limits\".", ""]
    for title, content in sections.items():
        lines += [f"## {title}", "", *content, ""]
    return "\n".join(lines)


def write_report(analysis_dir: Path) -> Path:
    """Write all the report tables and report.md. Return the report folder."""
    data = load_inputs(analysis_dir)
    tables = build_tables(data)
    folder = analysis_dir / "report"
    folder.mkdir(exist_ok=True)
    for name, table in tables.items():
        write_csv(table, folder / f"{name}.csv")
        write_parquet(table, folder / f"{name}.parquet", analysis_dir.name)
    (folder / "report.md").write_text(render(data, tables), encoding="utf-8")
    return folder
