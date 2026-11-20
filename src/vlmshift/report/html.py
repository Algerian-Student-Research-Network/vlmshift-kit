"""report.html: the same report as report.md, in one file that opens in any browser.

The page holds everything: its style, and the figures drawn inline. Nothing is fetched from the
network, so it can be read offline and sent as it is. The text is the one of report.md, turned
into HTML here: titles, paragraphs, lists, notes, tables kept as aligned text.
"""
from __future__ import annotations

import html
import re

STYLE = """
:root { color-scheme: light dark; --ink: #1a1a1a; --muted: #5f6368; --line: #d5d8dc; --back: #ffffff; --soft: #f4f5f7; }
@media (prefers-color-scheme: dark) { :root { --ink: #e8eaed; --muted: #9aa0a6; --line: #3c4043; --back: #17181a; --soft: #222427; } }
body { margin: 0 auto; padding: 24px 16px 64px; max-width: 1080px; background: var(--back); color: var(--ink);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 1.7rem; margin: 0 0 12px; } h2 { font-size: 1.25rem; margin: 36px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); }
nav { margin: 0 0 8px; font-size: 0.92rem; } nav a { margin-right: 14px; white-space: nowrap; }
a { color: inherit; } a:focus-visible { outline: 2px solid currentColor; outline-offset: 2px; }
blockquote { margin: 12px 0; padding: 10px 14px; border-left: 4px solid var(--muted); background: var(--soft); }
blockquote.warning { border-left-color: #b00020; }
pre { overflow-x: auto; padding: 12px; background: var(--soft); border: 1px solid var(--line); border-radius: 6px;
      font: 12.5px/1.45 ui-monospace, Consolas, monospace; font-variant-numeric: tabular-nums; }
code { font: 0.9em ui-monospace, Consolas, monospace; background: var(--soft); padding: 1px 4px; border-radius: 3px; }
figure { margin: 18px 0; overflow-x: auto; } figure svg { max-width: none; background: #ffffff; border: 1px solid var(--line); border-radius: 6px; }
figcaption { font-size: 0.88rem; color: var(--muted); margin-top: 4px; }
ul { padding-left: 22px; } li { margin: 3px 0; }
"""


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def inline(text: str) -> str:
    """Bold and code inside a line. Everything else is shown as written."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def body_of(markdown: str) -> tuple[str, list[str]]:
    """The report as HTML, and its section titles. Only what report.md uses is understood."""
    out, titles, paragraph, items = [], [], [], []
    lines = iter(markdown.split("\n"))

    def close():
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>")
            items.clear()

    for line in lines:
        if line.startswith("```"):  # a table, kept as aligned text
            close()
            block = []
            for inner in lines:
                if inner.startswith("```"):
                    break
                block.append(inner)
            out.append(f'<pre tabindex="0">{html.escape(chr(10).join(block), quote=False)}</pre>')
        elif line.startswith("# "):
            close()
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            close()
            titles.append(line[3:])
            out.append(f'<h2 id="{slug(line[3:])}">{inline(line[3:])}</h2>')
        elif line.startswith("> "):
            close()
            kind = ' class="warning"' if "**" in line else ""
            out.append(f"<blockquote{kind}>{inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            if paragraph:
                close()
            items.append(line[2:])
        elif not line.strip():
            close()
        else:
            if items:
                close()
            paragraph.append(line)
    close()
    return "\n".join(out), titles


def render_html(markdown: str, figures: dict[str, str]) -> str:
    """One page: the report, a table of contents, and the figures with their file names."""
    body, titles = body_of(markdown)
    shown = "".join(
        f'<figure>{svg}<figcaption><code>{name}.svg</code>, <code>{name}.pdf</code></figcaption></figure>'
        for name, svg in figures.items()
    )
    if shown:
        titles.append("Figures")
        body += f'\n<h2 id="figures">Figures</h2>\n<p>Each cell prints its value and its number of units: the color never carries the meaning alone.</p>{shown}'
    links = "".join(f'<a href="#{slug(title)}">{html.escape(title)}</a>' for title in titles)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>VLMShift report</title>\n'
        f"<style>{STYLE}</style>\n</head>\n<body>\n<nav aria-label=\"Sections\">{links}</nav>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )
