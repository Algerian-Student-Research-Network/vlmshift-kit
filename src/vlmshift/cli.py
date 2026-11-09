"""Command line: init, import, audit, prepare, check, run, analyze, all.

Each command redoes the previous ones if needed. They are safe to run again: same
content, same fingerprint, same folder. No hidden state to keep between two commands.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .audit.counts import run_audit
from .common import Step, VLMShiftError
from .config.load import load_analysis, load_experiment
from .data.starter import parse_questions, write_starter
from .data.table import build_table
from .inference.check import check_models
from .inference.runner import run_inference
from .report.export import write_report
from .stats.measures import run_analysis
from .variants.plan import prepare_variants


def show_issues(path: Path, step: int | None = None) -> None:
    """Print the issues of a step: one line per kind, so that no problem stays hidden in a file."""
    issues = pd.read_parquet(path)
    issues = issues if step is None else issues[issues["step"] == step]
    for code, group in issues.groupby("code", sort=False):
        print(f"  ! {code} ({len(group)}): {group['message'].iloc[0]}")


def do_init(images: str, axis: str, questions: list[str]) -> None:
    def shown(path: Path) -> str:  # as short as possible from the current folder
        try:
            return os.path.relpath(path)
        except ValueError:  # another drive, on Windows
            return str(path)

    table = write_starter(Path(images), axis, parse_questions(questions))
    target = Path(images).resolve().parent  # the files are written next to the images folder
    labels = table.drop_duplicates("image")["label"].value_counts()
    details = ", ".join(f"{labels[name]} {name}" for name in ("positive", "negative", "unknown") if name in labels)
    print(f"Table: {len(table)} rows, {labels.sum()} images ({details}) -> {shown(target / 'table.csv')}")
    experiment = shown(target / "experiment.yaml")
    print(f"Next: read {experiment}, then run: vlmshift prepare {experiment}")


def do_import(config) -> Step:
    table = build_table(config)
    print(f"Table: {len(table.df)} rows, {table.df['sample_id'].nunique()} samples -> {table.dir}")
    show_issues(table.dir / "issues.parquet")
    return table


def do_audit(config, show: bool = False) -> Step:
    table = do_import(config)
    counts = run_audit(config, table)
    print(f"Audit -> {table.dir / 'audit'}")
    show_issues(table.dir / "audit" / "issues.parquet")
    if show:
        print(counts[~counts["axis"].str.contains(" x ")].to_string(index=False))
    return table


def do_prepare(config) -> Step:
    variants = prepare_variants(config, do_audit(config))
    print(f"Variants: {len(variants.df)} rows -> {variants.dir}")
    show_issues(variants.dir / "issues.parquet")
    return variants


def do_check(config, only: list[str] | None = None) -> None:
    check_models(config, do_prepare(config), only)
    print("Next: read the answers above. If they look right, start the run.")


def do_run(config, only: list[str] | None = None) -> Path:
    folder = run_inference(config, do_prepare(config), only)
    print(f"Run -> {folder}")
    show_issues(folder / "issues.parquet", step=4)  # those of step 3 were printed above
    return folder


def do_analyze(config) -> Path:
    analysis = run_analysis(config)
    show_issues(analysis / "issues.parquet")
    report = write_report(analysis)
    print(f"Report -> {report / 'report.md'}")
    return report


COMMANDS = {
    "import": "check the table and the images",
    "audit": "count the rows of each level, before any model",
    "prepare": "write the exact texts sent to the models",
    "check": "ask each model a few real questions, before a long run",
    "run": "ask every model about every variant",
    "analyze": "compute the shifts and write the report, without calling any model",
    "all": "do every step, from the table to the report",
}


def parser() -> argparse.ArgumentParser:
    main_parser = argparse.ArgumentParser(prog="vlmshift", description=__doc__.splitlines()[0])
    commands = main_parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("init", help="make a table and two YAML files from a folder of images")
    start.add_argument("images", help="folder of images; subfolders positive/ and negative/ give the labels")
    start.add_argument("--question", action="append", required=True, metavar="LEVEL=TEXT", help="the question at one level; give one per level")
    start.add_argument("--axis", default="language", help="name of the compared condition (default: language)")
    for name, words in COMMANDS.items():
        command = commands.add_parser(name, help=words, description=words[0].upper() + words[1:] + ".")
        command.add_argument("analysis" if name == "analyze" else "experiment")
    for name in ("check", "run"):
        commands.choices[name].add_argument(
            "--only", nargs="+", metavar="MODEL", help="only these models (useful if they have different environments)"
        )
    commands.choices["analyze"].add_argument("--run", help="run folder: replaces the run key of analysis.yaml")
    commands.choices["all"].add_argument("analysis")
    return main_parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # not in a notebook
        sys.stdout.reconfigure(errors="replace")  # some Windows consoles refuse Unicode
    args = parser().parse_args(argv)
    steps = {"import": do_import, "prepare": do_prepare}
    try:
        if args.command == "init":
            do_init(args.images, args.axis, args.question)
        elif args.command in steps:
            steps[args.command](load_experiment(args.experiment))
        elif args.command == "audit":
            do_audit(load_experiment(args.experiment), show=True)
        elif args.command == "check":
            do_check(load_experiment(args.experiment), args.only)
        elif args.command == "run":
            do_run(load_experiment(args.experiment), args.only)
        elif args.command == "analyze":
            do_analyze(load_analysis(args.analysis, args.run))
        else:
            run = do_run(load_experiment(args.experiment))
            do_analyze(load_analysis(args.analysis, str(run)))
    except VLMShiftError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
