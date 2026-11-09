"""Write SOURCES.md, to find each source, and references.bib, to cite it, from the cards in prepare/.

Usage, from packages/med: python cite.py. With --check, compare without writing.
"""
import sys
from pathlib import Path

from prepare import ACCESS, card_problems, folder_of, name_of, sources

HERE = Path(__file__).parent


def texts() -> dict[Path, str]:
    """The content of SOURCES.md and references.bib, from the card of each source."""
    if found := [problem for source in sources() for problem in card_problems(source)]:
        raise SystemExit("\n".join(found))
    cards = [(name_of(source), source, source.SOURCE) for source in sources()]
    lines = [
        "# Med sources",
        "",
        "Written by `cite.py` from the card of each source, `SOURCE` in `prepare/<source>.py`: "
        "edit the cards, then run `python cite.py`.",
        "",
        "The data is not in this repository, and must never be: each user asks for it at its source "
        "and accepts its licence. Put each source in its folder under the data root (`MED_DATA`, "
        "otherwise `packages/med/data`), then run `python check.py configs/experiment.yaml`: "
        "it says what is missing, and whether a file is not the version used here.",
        "",
        "| Source | Access | Licence | Folder |",
        "|---|---|---|---|",
        *[f"| [{card['name']}]({card['page']}) | {card['access']} | {card['license']} | `{folder_of(source).as_posix()}/` |"
          for _, source, card in cards],
        "",
        "Access: " + "; ".join(f"**{level}**, {meaning}" for level, meaning in ACCESS.items()) + ".",
    ]
    for name, source, card in cards:
        version = (f"SHA-256 `{card['sha256']}`" if card.get("sha256") else
                   "SHA-256 not recorded yet: `python check.py` prints it once the data is there")
        lines += [
            "", f"## {card['name']}", "",
            f"- Script: `prepare/{name}.py`",
            f"- How to get it: {card['steps']} Page: {card['page']}",
            f"- {'Folder' if folder_of(source) == Path(source.RAW) else 'File'} checked: "
            f"`{Path(source.RAW).as_posix()}`, {version}",
            *([f"- Note: {card['note']}"] if card.get("note") else []),
            "", "To cite, also in `references.bib`:", "", "```bibtex", card["bibtex"].strip(), "```",
        ]
    bib = ["% Written by cite.py from the cards in prepare/. Cite every source you use.", ""]
    bib += [entry for _, _, card in cards for entry in (card["bibtex"].strip(), "")]
    return {HERE / "SOURCES.md": "\n".join(lines) + "\n", HERE / "references.bib": "\n".join(bib)}


if __name__ == "__main__":
    stale = [path.name for path, text in texts().items()
             if not path.exists() or path.read_text(encoding="utf-8") != text]
    if "--check" in sys.argv:
        print(f"Out of date: {', '.join(stale)}. Run python cite.py" if stale else "Up to date.")
        sys.exit(1 if stale else 0)
    for path, text in texts().items():
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Written: {', '.join(stale)}" if stale else "Already up to date.")
