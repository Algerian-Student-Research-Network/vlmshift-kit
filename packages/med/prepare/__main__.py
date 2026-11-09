"""Prepare all the sources: python -m prepare (from packages/med)."""
import sys

from . import name_of, sources
from .common import LayoutError, data_root, write_source

failed = []
for source in sources():
    try:
        print(f"{name_of(source)} -> {write_source(name_of(source), source.prepare(data_root()))}")
    except LayoutError as error:  # another version of the source: say what was found, go on with the others
        failed.append(f"{name_of(source)}: {error}")
for line in failed:
    print(line, file=sys.stderr)
sys.exit(1 if failed else 0)
