"""The figures of the report, drawn without any plotting library: each one as SVG and as PDF.

Two figures per measure: a heatmap of balanced accuracy by level, and the shifts between levels
with their corrected intervals. Every cell prints its value and its number of units: the color
only helps the eye, it never carries the meaning alone. A run made with the fake models of the
tests is stamped as synthetic, on the figure itself.
"""
from __future__ import annotations

import zlib

import pandas as pd

FONT = "Helvetica, Arial, sans-serif"
INK, MUTED, GRID, MISSING = "#1a1a1a", "#5f6368", "#c8ccd0", "#ececec"
SYNTHETIC = "SYNTHETIC DATA: NOT FOR PUBLICATION"
MOST_ROWS = 40  # a taller figure is not read: the table has every row
CHAR = 0.52  # mean width of a Helvetica character, as a share of the font size


def rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def shade(value: float) -> str:
    """From white (0) to a dark blue (1): light enough everywhere for a black or white number."""
    value = min(max(value, 0.0), 1.0)
    light, dark = (247, 251, 255), (8, 69, 148)
    return "#" + "".join(f"{round(a + (b - a) * value):02x}" for a, b in zip(light, dark))


class Drawing:
    """Rectangles, lines and texts, written as SVG or as a one-page PDF."""

    def __init__(self, width: float, height: float, title: str):
        self.width, self.height, self.title, self.items = width, height, title, []

    def rect(self, x, y, w, h, fill, stroke=None):
        self.items.append(("rect", x, y, w, h, fill, stroke))

    def line(self, x1, y1, x2, y2, stroke=INK, width=1.0):
        self.items.append(("line", x1, y1, x2, y2, stroke, width))

    def text(self, x, y, content, size=11, anchor="start", bold=False, fill=INK):
        self.items.append(("text", x, y, str(content), size, anchor, bold, fill))

    def svg(self) -> str:
        def escape(text: str) -> str:
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width:g} {self.height:g}" '
                 f'width="{self.width:g}" height="{self.height:g}" role="img" font-family="{FONT}">',
                 f"<title>{escape(self.title)}</title>", f'<rect width="100%" height="100%" fill="#ffffff"/>']
        for item in self.items:
            if item[0] == "rect":
                _, x, y, w, h, fill, stroke = item
                border = f' stroke="{stroke}"' if stroke else ""
                parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}"{border}/>')
            elif item[0] == "line":
                _, x1, y1, x2, y2, stroke, width = item
                parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{width:g}"/>')
            else:
                _, x, y, content, size, anchor, bold, fill = item
                weight = ' font-weight="bold"' if bold else ""
                parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" text-anchor="{anchor}" fill="{fill}"{weight}>{escape(content)}</text>')
        return "\n".join([*parts, "</svg>"]) + "\n"

    def pdf(self) -> bytes:
        """A PDF with the two standard fonts. A character outside Latin-1 is written as a question mark."""
        def literal(text: str) -> str:
            safe = text.encode("latin-1", "replace").decode("latin-1")
            return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        ops = []
        for item in self.items:
            if item[0] == "rect":
                _, x, y, w, h, fill, stroke = item
                ops.append("%.3f %.3f %.3f rg" % rgb(fill))
                if stroke:
                    ops.append("%.3f %.3f %.3f RG 0.5 w" % rgb(stroke))
                ops.append(f"{x:.2f} {self.height - y - h:.2f} {w:.2f} {h:.2f} re {'B' if stroke else 'f'}")
            elif item[0] == "line":
                _, x1, y1, x2, y2, stroke, width = item
                ops.append("%.3f %.3f %.3f RG" % rgb(stroke) + f" {width:.2f} w {x1:.2f} {self.height - y1:.2f} m {x2:.2f} {self.height - y2:.2f} l S")
            else:
                _, x, y, content, size, anchor, bold, fill = item
                shift = {"start": 0, "middle": 0.5, "end": 1}[anchor] * len(content) * size * CHAR
                ops.append("BT /F%d %g Tf %.3f %.3f %.3f rg %.2f %.2f Td (%s) Tj ET"
                           % (2 if bold else 1, size, *rgb(fill), x - shift, self.height - y, literal(content)))
        stream = zlib.compress("\n".join(ops).encode("latin-1"))
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:g} {self.height:g}] /Contents 4 0 R "
             "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>").encode(),
            b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        ]
        out, offsets = bytearray(b"%PDF-1.4\n"), []
        for number, body in enumerate(objects, 1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
        start = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, start)
        return bytes(out)


def row_label(row) -> str:
    return row["model"] if pd.isna(row["split"]) else f"{row['model']}, {row['split']}"


TOP = 92  # where the rows start, under the title, the note and the stamp


def frame(title: str, subtitle: str, labels: list[str], body_width: float, row_height: float, synthetic: bool):
    """An empty figure with its title, wide enough for its texts, and the left margin its row labels need."""
    left = min(max(len(label) for label in labels) * 11 * CHAR + 24, 420)
    texts = max(len(title) * 14 * 0.6, len(subtitle) * 10 * CHAR, len(SYNTHETIC) * 10 * 0.6) + 32
    drawing = Drawing(max(left + body_width + 24, texts), TOP + len(labels) * row_height + 34, title)
    drawing.text(16, 24, title, size=14, bold=True)
    drawing.text(16, 42, subtitle, size=10, fill=MUTED)
    if synthetic:
        drawing.text(16, 60, SYNTHETIC, size=10, bold=True, fill="#b00020")
    return drawing, left


def heatmap(metrics: pd.DataFrame, measure: str, metric: str, synthetic: bool) -> Drawing | None:
    """One row per model and cell, one column per level: the metric, and the units it rests on."""
    part = metrics[(metrics["measure"] == measure) & (metrics["metric"] == metric)
                   & (metrics["mode"] == "valid") & metrics["level_b"].isna()]
    if part.empty:
        return None
    part = part.assign(row=[row_label(row) for row in part.to_dict("records")])
    rows, levels = list(dict.fromkeys(part["row"]))[:MOST_ROWS], sorted(part["level_a"].unique())
    cell_w, cell_h = max(64, max(len(str(level)) for level in levels) * 7 + 16), 34
    note = f"{metric} by {measure}, valid answers. Each cell: value, then units." + (
        f" First {MOST_ROWS} rows of {part['row'].nunique()}: all are in metrics.csv." if part["row"].nunique() > MOST_ROWS else "")
    drawing, left = frame(f"{metric} by {measure}", note, rows, len(levels) * cell_w, cell_h, synthetic)
    values = part.set_index(["row", "level_a"])
    for j, level in enumerate(levels):
        drawing.text(left + j * cell_w + cell_w / 2, TOP - 10, level, size=10, anchor="middle", bold=True)
    for i, name in enumerate(rows):
        y = TOP + i * cell_h
        drawing.text(left - 10, y + cell_h / 2 + 4, name, size=11, anchor="end")
        for j, level in enumerate(levels):
            x = left + j * cell_w
            found = values.loc[(name, level)] if (name, level) in values.index else None
            if found is None or pd.isna(found["value"]):
                drawing.rect(x, y, cell_w - 2, cell_h - 2, MISSING)
                drawing.text(x + cell_w / 2, y + cell_h / 2 + 3, "not estimable" if found is not None else "", size=8, anchor="middle", fill=MUTED)
                continue
            ink = "#ffffff" if found["value"] > 0.45 else INK
            drawing.rect(x, y, cell_w - 2, cell_h - 2, shade(found["value"]))
            drawing.text(x + cell_w / 2, y + 15, f"{found['value']:.3f}", size=11, anchor="middle", bold=True, fill=ink)
            drawing.text(x + cell_w / 2, y + 27, f"n={int(found['n_units'])}", size=8, anchor="middle", fill=ink)
    return drawing


def forest(metrics: pd.DataFrame, measure: str, metric: str, synthetic: bool) -> Drawing | None:
    """The shifts of a measure with their corrected intervals, around the line of no shift."""
    part = metrics[(metrics["measure"] == measure) & (metrics["metric"] == metric)
                   & (metrics["mode"] == "valid") & metrics["value"].notna()]
    if part.empty:
        return None
    total = len(part)
    part = part.reindex(part["value"].abs().sort_values(ascending=False).index).head(MOST_ROWS)
    labels = [f"{row_label(row)}: {row['level_a']} to {row['level_b']}" for row in part.to_dict("records")]
    lowest, highest = min(part["ci_low"].min(), 0), max(part["ci_high"].max(), 0)
    span = (highest - lowest) or 1.0
    low, high = lowest - 0.05 * span, highest + 0.05 * span
    width, row_h = 360, 22
    note = f"{metric} with its corrected interval. * significant." + (
        f" The {MOST_ROWS} largest of {total}: all are in metrics.csv." if total > MOST_ROWS else "")
    drawing, left = frame(f"Shifts of {measure}", note, labels, width + 110, row_h, synthetic)

    def at(value: float) -> float:
        return left + (value - low) / (high - low) * width

    bottom = TOP + len(labels) * row_h
    drawing.line(at(0), TOP - 6, at(0), bottom, stroke=MUTED)
    drawing.text(at(0), bottom + 16, "0", size=9, anchor="middle", fill=MUTED)
    for tick in (lowest, highest):
        if abs(tick) > 0.12 * span:  # far enough from 0 for the two labels not to touch
            drawing.text(at(tick), bottom + 16, f"{tick:+.2f}", size=9, anchor="middle", fill=MUTED)
    for i, (label, row) in enumerate(zip(labels, part.to_dict("records"))):
        y = TOP + i * row_h + row_h / 2
        drawing.text(left - 10, y + 4, label, size=10, anchor="end")
        drawing.line(left, y, left + width, y, stroke=GRID, width=0.5)
        drawing.line(at(row["ci_low"]), y, at(row["ci_high"]), y, width=1.6)
        drawing.rect(at(row["value"]) - 3, y - 3, 6, 6, INK)
        star = " *" if row["significant"] is True else ""
        drawing.text(left + width + 8, y + 4, f"{row['value']:+.3f}{star} (n={int(row['n_units'])})", size=10, bold=bool(star))
    return drawing


def make_figures(metrics: pd.DataFrame, synthetic: bool) -> dict[str, Drawing]:
    """The figures of every measure: name → drawing."""
    figures = {}
    for measure in sorted(metrics["measure"].unique()):
        found = {f"fig_{measure}_ba": heatmap(metrics, measure, "ba", synthetic),
                 f"fig_{measure}_shifts": forest(metrics, measure, "delta_ba", synthetic)}
        figures |= {name: drawing for name, drawing in found.items() if drawing is not None}
    return figures
