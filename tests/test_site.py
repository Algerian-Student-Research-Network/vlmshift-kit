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


def test_home_says_where_to_run_it():
    """The site runs nothing: the home page says how to run the tool on a computer, in Colab, or in a container."""
    home = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'id="get-started"' in home
    assert "colab.research.google.com/github/" in home and "examples/pope_mini/colab.ipynb" in home
    assert "pip install" in home and "docker run" in home
    assert "<!--" not in home.split("<body>")[1].replace("<!--SCENE-->", "")  # every placeholder filled


def test_med_page_comes_from_its_readme():
    """The Med guide is written once, in packages/med/README.md; on the site its relative links still lead somewhere."""
    med = (DOCS / "med.html").read_text(encoding="utf-8")
    assert 'href="med.html"' in (DOCS / "index.html").read_text(encoding="utf-8")
    assert "DRAFT TERMS" in med and "replay.py" in med
    links = re.findall(r'<main.*?</main>', med, re.S)[0]
    assert not re.findall(r'href="(?!https?://|#|[\w-]+\.html)[^"]*"', links)
    assert "/blob/main/packages/med/SOURCES.md" in links


def test_reference_lists_every_key_the_code_accepts():
    """The reference page is written from the schemas: each key of the code has a row."""
    from vlmshift.config.schema import AnalysisConfig, ExperimentConfig, ModelConfig, TaskConfig
    reference = (DOCS / "reference.html").read_text(encoding="utf-8")
    for key in ExperimentConfig.model_fields:
        assert f"<code>{key}" in reference, key
    for key in AnalysisConfig.model_fields:
        assert f"<code>{key}" in reference, key
    for key in ModelConfig.model_fields:
        assert f"<code>models[].{key}</code>" in reference, key
    for key in TaskConfig.model_fields:
        assert f"<code>task.{key}" in reference, key
