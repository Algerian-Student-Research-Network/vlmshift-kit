"""tools/site/: the project site must match the code and hold together."""
import re
import subprocess
import sys

import pytest

from conftest import ROOT

DOCS = ROOT / "docs"


def test_site_is_up_to_date():
    """Also fails if a file is not placed, or if an explanation no longer matches the code."""
    pytest.importorskip("markdown_it")
    done = subprocess.run(
        [sys.executable, "tools/site/build.py", "--check"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_images_do_not_depend_on_the_page():
    """A plain image has no page theme around it: its colours must be fixed."""
    assert not [path.name for path in (DOCS / "assets").glob("*.svg") if "var(" in path.read_text(encoding="utf-8")]


def test_home_offers_the_three_choices():
    home = (DOCS / "index.html").read_text(encoding="utf-8")
    for page in ("tutorial.html", "files.html", "documentation.html"):
        assert f'class="choice" href="{page}"' in home


def test_each_step_leads_to_its_files():
    guide = (DOCS / "files.html").read_text(encoding="utf-8")
    assert guide.index('class="pipeline"') < guide.index('id="map"')  # the steps open the file guide
    for number in range(1, 7):
        assert f'href="#step-{number}"' in guide
        assert f'id="step-{number}"' in guide


def test_links_between_pages_all_work():
    pages = {path.name: path.read_text(encoding="utf-8") for path in DOCS.glob("*.html")}
    dead = []
    for name, text in pages.items():
        for target in re.findall(r'href="([^"]+)"', text):
            if target.startswith(("http", "data:", "assets/")):
                continue
            page, _, anchor = target.partition("#")
            page = page or name
            if page not in pages or (anchor and f'id="{anchor}"' not in pages[page]):
                dead.append(f"{name} -> {target}")
    assert not dead
