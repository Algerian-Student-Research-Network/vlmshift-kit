#!/usr/bin/env bash
# Run the whole Med benchmark in one command: check, preparation, inference, report, tables of the paper.
# Raw data: MED_DATA variable, otherwise packages/med/data.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH=.

python check.py configs/experiment.yaml
python -m prepare
# paper.py registers the Med models, runs vlmshift all, then writes the tables of the paper.
python paper.py configs/experiment.yaml configs/analysis.yaml
