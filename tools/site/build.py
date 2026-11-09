"""Builds the project site and its images in docs/.

    python tools/site/build.py            writes everything
    python tools/site/build.py --check    fails if something is out of date
    python tools/site/build.py --home-fragment page.html    also writes the home page without its <html> shell

Tutorial, server guide, Colab guide, documentation, design and decisions are written in Markdown, so they also read
well on GitHub. Here they become pages of the site. The file guide comes from file_map.py.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

from markdown_it import MarkdownIt

import art
import file_map

HERE = Path(__file__).parent
DOCS = file_map.ROOT / "docs"
FONTS = ("family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400"
         "&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap")
MENU = [("tutorial.html", "Tutorial"), ("files.html", "File guide"), ("documentation.html", "Documentation")]
MARKDOWN_PAGES = {
    "tutorial": "Tutorial", "server": "NVIDIA server", "colab": "Colab and Kaggle", "documentation": "Documentation",
    "design": "Design", "decisions": "Decisions",
}


def head(title: str) -> str:
    icon = "data:image/svg+xml," + quote(art.fixed(art.logo()))
    return (
        f"<title>{title}</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        f'<meta name="description" content="VLMShift. {art.TAGLINE}">\n'
        f'<link rel="icon" href="{icon}">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{FONTS}">\n'
        '<link rel="stylesheet" href="assets/style.css">\n'
    )


def top(current: str) -> str:
    here = ' aria-current="page"'
    links = "".join(f'<a href="{href}"{here if href == current else ""}>{name}</a>' for href, name in MENU)
    mark = art.logo('aria-hidden="true"')
    brand = f'<a class="brand" href="index.html">{mark}<span>VLMShift</span></a>'
    return f'<header class="top">{brand}<nav aria-label="Site">{links}</nav></header>\n'


def page(title: str, current: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"{head(title)}</head>\n<body>\n{top(current)}{body}\n</body>\n</html>\n"
    )


def slug(text: str) -> str:
    """The same anchors as GitHub, so that links work on both."""
    return re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-")


def markdown_page(name: str) -> str:
    """A Markdown file as a page: anchors on titles, wide tables that scroll, links between pages."""
    parser = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = parser.parse((DOCS / f"{name}.md").read_text(encoding="utf-8"))
    for token, following in zip(tokens, tokens[1:]):
        if token.type == "heading_open":
            token.attrSet("id", slug(following.content))
    content = parser.renderer.render(tokens, parser.options, {})
    content = content.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    return re.sub(r'href="([\w-]+)\.md(#[^"]*)?"', lambda found: f'href="{found[1]}.html{found[2] or ""}"', content)


def home() -> str:
    body = (HERE / "home.html").read_text(encoding="utf-8")
    return (
        body.replace("<!--TAGLINE-->", art.TAGLINE)
        .replace("<!--SCENE-->", art.scene('class="scene wide"') + art.scene('class="scene narrow"', compact=True))
    )


def file_guide() -> str:
    data = json.dumps(file_map.items, ensure_ascii=False).replace("</", "<\\/")
    body = (HERE / "files.html").read_text(encoding="utf-8")
    actual = f"<style>figure.actual #map {{ width: {file_map.W}px; max-width: none; }}</style>\n"
    body = body.replace("<!--PIPELINE-->", art.pipeline(clickable=True))
    return actual + body.replace("<!--SVG-->", file_map.svg).replace("/*DATA*/null", data)


def site() -> dict[Path, str]:
    """Every generated file and its content."""
    pages = {
        "index": page("VLMShift", "index.html", home()),
        "files": page("File guide · VLMShift", "files.html", file_guide()),
    }
    for name, title in MARKDOWN_PAGES.items():
        pages[name] = page(f"{title} · VLMShift", f"{name}.html", f'<main class="reading">\n{markdown_page(name)}</main>')
    outputs = {DOCS / f"{name}.html": content for name, content in pages.items()}
    outputs[art.ASSETS / "style.css"] = (HERE / "style.css").read_text(encoding="utf-8")
    outputs[DOCS / ".nojekyll"] = ""  # GitHub Pages serves these pages as they are
    return {**outputs, **art.asset_files()}


def main() -> None:
    arguments = argparse.ArgumentParser(description="Build the VLMShift site.")
    arguments.add_argument("--check", action="store_true", help="fail if something in docs/ is out of date")
    arguments.add_argument("--home-fragment", help="also write the home page without its <html> shell")
    options = arguments.parse_args()

    outputs = site()
    if options.check:
        stale = [path.name for path, content in outputs.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            sys.exit(f"Out of date: {stale}. Run: python tools/site/build.py")
        return
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    if options.home_fragment:
        Path(options.home_fragment).write_text(head("VLMShift") + top("index.html") + home(), encoding="utf-8", newline="\n")
    print(f"{len(outputs)} files written, {len(file_map.items)} boxes in the file guide")


if __name__ == "__main__":
    main()
