"""The VLMShift look, in one place: one sage green and its transparent tints, on a white page.

The logo is a circle cut in two: one half has moved, and faded. That gap is what the project
measures. Titles are set in Newsreader, the text in Inter.

Colours and fonts are CSS variables with a fallback. Inside the site they follow the
page theme. Written as image files, for GitHub for example, they use the fallback.
Tints are the one colour with an opacity, so they work on both themes.
"""
import re
from pathlib import Path

TAGLINE = "A simple kit for measuring shift in vision-language models."

PAPER, CARD, INK, ACCENT = "#ffffff", "#ffffff", "#1b2420", "#3f7a5c"
SERIF = "Newsreader, Georgia, 'Times New Roman', serif"
SANS = "Inter, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace"

ASSETS = Path(__file__).parents[2] / "docs" / "assets"


def color(name: str, value: str) -> str:
    return f"var(--{name}, {value})"


def font(name: str, value: str) -> str:
    return color(name, value).replace("'", "")


def fixed(drawing: str) -> str:
    """The drawing with its fallback colours and fonts: for an image file, outside the page."""
    return re.sub(r"var\(--[\w-]+, ([^()]+)\)", r"\1", drawing)


def logo(attributes: str = "") -> str:
    """A circle cut in two: the right half has moved down, and faded."""
    ink = color("accent", ACCENT)
    return (
        f'<svg viewBox="0 0 32 32" role="img" aria-label="VLMShift logo" xmlns="http://www.w3.org/2000/svg" {attributes}>'
        f'<path d="M15,4 A10,10 0 0 0 15,24 Z" fill="{ink}"/>'
        f'<path d="M17,8 A10,10 0 0 1 17,28 Z" fill="{ink}" fill-opacity=".38"/></svg>'
    )


def scene(attributes: str = "", compact: bool = False) -> str:
    """The idea as one line: the answer holds in English, then moves when the question is in French.

    The compact drawing is narrower with larger words, for a phone.
    """
    ink, accent, sans, serif = color("ink", INK), color("accent", ACCENT), font("sans", SANS), font("serif", SERIF)
    if compact:
        width, height, high, low, turn, size, shift_x = 640, 300, 110, 210, 250, 32, 470
    else:
        width, height, high, low, turn, size, shift_x = 1000, 240, 96, 176, 470, 27, 790  # shown at most 680 px wide
    curve = f"C{turn + 30},{high} {turn + 30},{low} {turn + 60},{low}"
    parts = [
        # Where the answer would be with no shift
        f'<path class="ideal" d="M{turn},{high} H{width}" fill="none" stroke="{accent}" stroke-opacity=".45" stroke-width="2.5" '
        'stroke-linecap="round" stroke-dasharray="1 8"/>',
        # The gap between the two: the shift
        f'<path class="gap" d="M{turn},{high} {curve} H{width} V{high} Z" fill="{accent}" fill-opacity=".12"/>',
        # The answer. pathLength="1" lets the page draw it from start to end.
        f'<path class="answer" pathLength="1" d="M4,{high} H{turn} {curve} H{width - 4}" fill="none" stroke="{accent}" '
        'stroke-width="3.5" stroke-linecap="round"/>',
        f'<line class="divider" x1="{turn + 30}" y1="20" x2="{turn + 30}" y2="{height - 8}" stroke="{accent}" '
        'stroke-opacity=".4" stroke-dasharray="3 5"/>',
        f'<text class="ask-en" x="4" y="{high - size}" font-family="{sans}" font-size="{size}" fill="{ink}">Is there a circle? '
        f'<tspan fill="{accent}" font-weight="600">Yes</tspan></text>',
        f'<text class="ask-fr" x="{width - 4}" y="{low + size * 1.8:.0f}" text-anchor="end" font-family="{sans}" font-size="{size}" '
        f'fill="{ink}">Y a-t-il un cercle ? <tspan fill="{accent}" font-weight="600">No</tspan></text>',
        f'<text class="shift" x="{shift_x}" y="{high + (low - high) / 2 + size * .5:.0f}" text-anchor="middle" font-family="{serif}" '
        f'font-size="{size * 1.5:.0f}" font-style="italic" fill="{accent}">Shift</text>',
    ]
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" xmlns="http://www.w3.org/2000/svg" {attributes} '
        'aria-label="The same image and the same question, in English then in French. The model answers yes, then no: '
        'the gap between the two is a shift.">'
        f'{"".join(parts)}</svg>'
    )


def banner() -> str:
    """The README header. It carries its own background, so it reads on any page."""
    return (
        f'<svg viewBox="0 0 1280 320" role="img" aria-label="VLMShift. {TAGLINE}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="1" y="1" width="1278" height="318" rx="14" fill="{PAPER}" stroke="{ACCENT}" stroke-opacity=".25" stroke-width="2"/>'
        + logo('x="64" y="78" width="64" height="64"')
        + f'<text x="146" y="132" font-family="{SERIF}" font-size="60" fill="{INK}">VLMShift</text>'
        f'<text x="66" y="208" font-family="{SERIF}" font-size="30" fill="{INK}">A simple kit for measuring shift</text>'
        f'<text x="66" y="248" font-family="{SERIF}" font-size="30" fill="{INK}">in vision-language models.</text>'
        + scene('x="700" y="38" width="520" height="244"', compact=True)
        + "</svg>"
    )


STEPS = [
    ("1", "Data", "Check your table", "and link versions", "table.parquet"),
    ("2", "Audit", "Count what you have", "before any model", "counts.parquet"),
    ("3", "Variants", "Write the exact", "text models receive", "variants.parquet"),
    ("4", "Inference", "Ask every model", "about every variant", "results.parquet"),
    ("5", "Statistics", "Measure each shift", "and its uncertainty", "metrics.parquet"),
    ("6", "Report", "Tables, matrices", "and a summary", "report.md"),
]


def pipeline(clickable: bool = False) -> str:
    """The six steps, what each one writes, and which file drives it.

    If clickable, each step is a link to the anchor step-<number> of the same page.
    """
    ink, accent, card_fill = color("ink", INK), color("accent", ACCENT), color("card", CARD)
    sans, serif, mono = font("sans", SANS), font("serif", SERIF), font("mono", MONO)
    width, height, card, gap, left, top = 1280, 330, 188, 14, 25, 84
    parts = [f'<rect width="{width}" height="{height}" rx="14" fill="{accent}" fill-opacity=".06"/>']

    def bracket(first: int, last: int, text: str, y: int = 44) -> None:
        x1 = left + first * (card + gap)
        x2 = left + last * (card + gap) + card
        parts.append(f'<path d="M{x1},{y + 14} V{y + 6} H{x2} V{y + 14}" fill="none" stroke="{accent}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x1}" y="{y - 4}" font-family="{mono}" font-size="17" fill="{ink}">{text}</text>')

    bracket(0, 3, "experiment.yaml")
    bracket(4, 5, "analysis.yaml")
    line_y = top + 66
    for index, (number, name, first, second, output) in enumerate(STEPS):
        x = left + index * (card + gap)
        hot = number == "4"
        card_parts = [
            f'<rect x="{x}" y="{top}" width="{card}" height="178" rx="8" fill="{card_fill}"/>',
            f'<rect class="step-card" x="{x}" y="{top}" width="{card}" height="178" rx="8" fill="{accent}" '
            f'fill-opacity="{".14" if hot else "0"}" stroke="{accent}" stroke-opacity="{1 if hot else .3}" '
            f'stroke-width="{2 if hot else 1.2}"/>',
            f'<text x="{x + 15}" y="{top + 44}" font-family="{serif}" font-size="34" fill="{accent}">{number}</text>',
            f'<text x="{x + 44}" y="{top + 42}" font-family="{sans}" font-size="21" font-weight="600" fill="{ink}">{name}</text>',
        ]
        for row, words in enumerate((first, second)):
            card_parts.append(f'<text x="{x + 15}" y="{top + 84 + row * 22}" font-family="{sans}" font-size="16.5" fill="{ink}">{words}</text>')
        card_parts.append(f'<text x="{x + 15}" y="{top + 156}" font-family="{mono}" font-size="14.5" fill="{accent}">{output}</text>')
        drawn = "".join(card_parts)
        parts.append(f'<a href="#step-{number}" aria-label="Step {number}, {name}: see its files">{drawn}</a>' if clickable else drawn)
        if index < 5:
            parts.append(
                f'<path d="M{x + card + 2},{line_y} h{gap - 7} l-4,-4 m4,4 l-4,4" fill="none" '
                f'stroke="{accent}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    notes = [
        (left, ink, "Same content, same fingerprint: finished work is found again."),
        (left + 3 * (card + gap), accent, "Calls the models"),
        (left + 4 * (card + gap), ink, "Rerun freely: reads the disk, needs no model"),
    ]
    for x, colour, words in notes:
        parts.append(f'<text x="{x}" y="{top + 216}" font-family="{sans}" font-size="15.5" fill="{colour}">{words}</text>')
    return (
        f'<svg class="pipeline" viewBox="0 0 {width} {height}" role="group" xmlns="http://www.w3.org/2000/svg" '
        'aria-label="Six steps: data, audit, variants, inference, statistics, report. '
        'experiment.yaml drives steps 1 to 4 and analysis.yaml drives steps 5 and 6.">'
        f'{"".join(parts)}</svg>'
    )


def asset_files() -> dict[Path, str]:
    """The image files. As plain images they use fixed colours, which every viewer can read."""
    drawings = {"logo": logo(), "banner": banner(), "pipeline": pipeline()}
    return {ASSETS / f"{name}.svg": fixed(drawing) + "\n" for name, drawing in drawings.items()}
