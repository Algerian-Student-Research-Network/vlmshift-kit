"""Puts the report of a real demo run on the site, so a reader sees what the tool gives before installing it.

    python examples/pope_mini/fetch.py
    vlmshift all examples/pope_mini/experiment.yaml examples/pope_mini/analysis.yaml
    python tools/site/example_report.py workspace/runs/<run>/analyses/<analysis>/report

report.html holds its style and its figures, so it is copied as it is. The copy is refused if the
page names a folder of this machine, or if it was made by the fake models of the tests.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "docs" / "report-example.html"
LOCAL = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\|(?<![A-Za-z])[A-Za-z]:/(?!/)|/home/|/Users/|/content/")


def publish(report_folder: Path) -> Path:
    text = (report_folder / "report.html").read_text(encoding="utf-8")
    if found := LOCAL.search(text):
        raise SystemExit(f"report.html names a local folder ({text[found.start():found.start() + 40]!r}): not published")
    if "SYNTHETIC" in text.upper():
        raise SystemExit("report.html comes from fake models: publish the report of a real run")
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    return TARGET


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"-> {publish(Path(sys.argv[1]))}")
