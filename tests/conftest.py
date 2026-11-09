"""The shared toy dataset, ready to use for all the tests."""
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from vlmshift.cli import do_analyze, do_run
from vlmshift.config.load import load_analysis, load_experiment, read_yaml, write_yaml

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
os.environ["HF_HUB_OFFLINE"] = "1"  # the tests download nothing


def copy_toy(target: Path) -> Path:
    shutil.copytree(FIXTURES / "toy", target, ignore=shutil.ignore_patterns("workspace"))
    return target


@pytest.fixture
def toy(tmp_path) -> Path:
    """A copy of the toy folder: each test has its own workspace."""
    return copy_toy(tmp_path / "toy")


@pytest.fixture
def edit(toy):
    """Edit experiment.yaml (or analysis.yaml) of the toy folder, then reload it."""

    def apply(change, name="experiment.yaml"):
        raw = read_yaml(toy / name)
        change(raw)
        write_yaml(raw, toy / name)
        load = load_analysis if name == "analysis.yaml" else load_experiment
        return load(toy / name)

    return apply


@pytest.fixture(scope="session")
def full_run(tmp_path_factory) -> SimpleNamespace:
    """The full pipeline, run only once for all the tests that read its outputs."""
    folder = copy_toy(tmp_path_factory.mktemp("full") / "toy")
    run = do_run(load_experiment(folder / "experiment.yaml"))
    report = do_analyze(load_analysis(folder / "analysis.yaml", str(run)))
    return SimpleNamespace(
        folder=folder,
        run=run,
        analysis=report.parent,
        report=report,
        metrics=pd.read_parquet(report.parent / "metrics.parquet"),
    )
