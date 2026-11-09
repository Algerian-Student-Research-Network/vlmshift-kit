"""The file guide: a map drawn from the real code, whose links come from reading the imports.

Importing this module stops with an error if a file is not placed, if an import cannot
be drawn, or if an explanation no longer matches the code. So the map cannot go stale.
It gives `svg` (the drawing), `items` (the text of each box) and `W` (its width).
"""
import ast
import html
from pathlib import Path

from explain import AD, BLOCKS, EXPLAIN, LIMIT, M, S

ROOT = Path(__file__).parents[2]

ROLES = {
    S + "cli.py": "The commands: init, import, audit, prepare, check, run, analyze, all",
    S + "data/starter.py": "Makes a table and two YAML files from a folder of images",
    S + "config/load.py": "Reads the YAML files, checks them, fingerprints them",
    S + "config/schema.py": "The rules of the two configuration files",
    S + "extensions.py": "The directory of models: a name gives a class",
    S + "common.py": "Fingerprints, Parquet writing, issues",
    S + "__init__.py": "The version number of the package",
    S + "data/table.py": "Runs step 1 and writes the table",
    S + "data/io.py": "Reads CSV and Parquet, everything as text",
    S + "data/ids.py": "Computes sample_id and the id of each row",
    S + "data/checks.py": "Standard names, image fingerprints, checks",
    S + "audit/counts.py": "Counts rows, units and labels by level",
    S + "audit/samples.py": "Variants per sample, missing levels",
    S + "variants/plan.py": "Runs step 3 and writes the variants",
    S + "variants/prompts.py": "Fills the templates: the exact text sent",
    S + "tasks/__init__.py": "Finds the task type by its name",
    S + "tasks/yes_no.py": "The yes/no task and reading by list",
    S + "tasks/base.py": "The Task interface and template holes",
    S + "inference/runner.py": "Sends batches; handles resuming, failures, results",
    S + "inference/check.py": "Asks each model a few questions before a long run",
    S + "inference/run_log.py": "Writes the run log",
    S + "tasks/llm_reader.py": "Small LLM: reads answers, validated by language",
    S + "inference/base.py": "The Model interface and the ModelError error",
    AD + "vlmevalkit.py": "Generative models through VLMEvalKit",
    AD + "contrastive.py": "Contrastive models (CLIP) through Hugging Face",
    AD + "huggingface.py": "Generative models through Hugging Face, with no VLMEvalKit",
    AD + "fake.py": "Fake models and fake LLM of the test dataset",
    S + "stats/measures.py": "Runs the measures, writes metrics and probes",
    S + "stats/pairing.py": "Chooses the rows, pairs the samples",
    S + "stats/shifts.py": "Computes the shifts Δ and the answer change",
    S + "stats/probes.py": "Probe: can a column be read in the vectors?",
    S + "stats/metrics.py": "Sens, Spec, BA, yes rate, INVALID rate, change rates",
    S + "stats/bootstrap.py": "Draws the units, computes intervals, significance",
    S + "report/export.py": "Writes the tables, the matrices and report.md",
    M + "run.sh": "Runs check, prepare, then paper.py",
    M + "check.py": "Checks raw data, versions, libraries, weights",
    M + "cite.py": "Writes SOURCES.md and references.bib",
    M + "prepare/__main__.py": "Prepares all the sources",
    M + "prepare/__init__.py": "Finds the source scripts, checks their cards",
    M + "prepare/<source>.py": "11 scripts, one per source: card, raw files, table",
    M + "label_rules/<source>.py": "11 rules: raw label to positive, negative, unknown",
    M + "prepare/template.py": "The script to copy for a new source",
    M + "label_rules/template.py": "The label rule to copy for a new source",
    M + "prepare/common.py": "Reads layouts, samples 300 patients, adds languages",
    M + "paper.py": "Runs Core, then writes the tables of the paper",
    M + "adapters/__init__.py": "Registers the models specific to Med",
    M + "adapters/open_clip_model.py": "Models of the open_clip library: BiomedCLIP",
    M + "adapters/chexzero.py": "CheXzero: OpenAI CLIP weights read by open_clip",
}


def family(folder: str, apart: set) -> list:
    """The files of a folder that share one shape, drawn as one row."""
    return sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob("*.py") if p.stem not in apart)


# One row for many files of the same shape: the 11 sources and their 11 label rules.
GROUPS = {
    M + "prepare/<source>.py": family(M + "prepare", {"__init__", "__main__", "common", "template"}),
    M + "label_rules/<source>.py": family(M + "label_rules", {"__init__", "template"}),
}
MEMBER = {member: group for group, members in GROUPS.items() for member in members}

STEP = "mem"
BANDS = [
    dict(num="", title="Start", sub="A command starts the tool, which reads and checks the configuration", call="",
         rows=[S + "cli.py", S + "data/starter.py", S + "config/load.py", S + "config/schema.py", S + "extensions.py", S + "common.py", S + "__init__.py"],
         reads=[("file", "experiment.yaml"), ("file", "analysis.yaml"), ("note", "answer lists, if separate"), ("note", "folder of images, for init")],
         writes=[("note", "with init only:"), ("file", "table.csv"), ("file", "experiment.yaml"), ("file", "analysis.yaml")]),
    dict(num="1", title="Data", sub="Make the table safe and link the variants", call="do_import() → build_table()",
         rows=[S + "data/table.py", S + "data/io.py", S + "data/ids.py", S + "data/checks.py"],
         reads=[("dir", "data: key of experiment.yaml"), ("file", "table.csv or .parquet"), ("dir", "images it names"), ("file", "images/*.png")],
         writes=[("dir", "tables/<t>/"), ("file", "table.parquet"), ("file", "issues.parquet")]),
    dict(num="2", title="Audit", sub="Count what you have, without changing anything", call="do_audit() → run_audit()",
         rows=[S + "audit/counts.py", S + "audit/samples.py"],
         reads=[(STEP, "Step of the table (in memory)")],
         writes=[("dir", "tables/<t>/audit/"), ("file", "counts.parquet, .csv"), ("file", "samples.parquet, .csv"), ("file", "issues.parquet")]),
    dict(num="3", title="Variants", sub="Write the exact text sent to the models", call="do_prepare() → prepare_variants()",
         rows=[S + "variants/plan.py", S + "variants/prompts.py", S + "tasks/__init__.py", S + "tasks/yes_no.py", S + "tasks/base.py"],
         reads=[(STEP, "Step of the table (in memory)")],
         writes=[("dir", "tables/<t>/variants/<v>/"), ("file", "variants.parquet, .csv"), ("file", "issues.parquet")]),
    dict(num="4", title="Inference", sub="Ask every model about every variant", call="do_run() → run_inference()",
         rows=[S + "inference/runner.py", S + "inference/check.py", S + "inference/run_log.py", S + "tasks/llm_reader.py", S + "inference/base.py",
               AD + "vlmevalkit.py", AD + "huggingface.py", AD + "contrastive.py", AD + "fake.py"],
         reads=[(STEP, "Step of the variants (in memory)"), ("dir", "tables/<t>/variants/<v>/"), ("file", "issues.parquet"),
                ("dir", "tables/<t>/audit/"), ("file", "counts, samples, issues"),
                ("dir", "cache/<m>/"), ("file", "outputs.jsonl"), ("file", "vectors.jsonl"), ("file", "sessions.jsonl"),
                ("file", "errors.jsonl"),
                ("dir", "reader_cache/"), ("file", "<h>.jsonl"), ("dir", "reader_llm.validation"), ("file", "annotated answers (CSV)")],
         writes=[("dir", "cache/<m>/"), ("file", "outputs.jsonl"), ("file", "vectors.jsonl"),
                 ("file", "sessions.jsonl"), ("file", "errors.jsonl"), ("dir", "reader_cache/"), ("file", "<h>.jsonl"), ("dir", "runs/<r>/"),
                 ("file", "experiment.yaml (copy)"), ("file", "audit/ (copy)"), ("file", "results.parquet"), ("file", "embeddings.parquet"),
                 ("file", "reader_validation.parquet"), ("file", "issues.parquet"), ("file", "run_log.json")]),
    dict(num="5", title="Statistics", sub="Measure the shifts and their uncertainty", call="do_analyze() → run_analysis()",
         rows=[S + "stats/measures.py", S + "stats/pairing.py", S + "stats/shifts.py", S + "stats/probes.py", S + "stats/metrics.py", S + "stats/bootstrap.py"],
         reads=[("dir", "runs/<r>/"), ("file", "experiment.yaml"), ("file", "run_log.json"), ("file", "results.parquet"), ("file", "embeddings.parquet"),
                ("dir", "tables/<t>/"), ("file", "table.parquet"), ("file", "variants/<v>/variants.parquet")],
         writes=[("dir", "runs/<r>/analyses/<a>/"), ("file", "analysis.yaml (copy)"), ("file", "metrics.parquet"), ("file", "probes.parquet"), ("file", "issues.parquet")]),
    dict(num="6", title="Report", sub="Present the numbers and the coverage", call="do_analyze() → write_report()",
         rows=[S + "report/export.py"], writes_mid=True,
         reads=[("dir", "runs/<r>/analyses/<a>/"), ("file", "analysis.yaml"), ("file", "metrics.parquet"), ("file", "probes.parquet"), ("file", "issues.parquet"),
                ("dir", "runs/<r>/"), ("file", "experiment.yaml, run_log.json"), ("file", "results.parquet, issues.parquet"), ("file", "reader_validation.parquet"),
                ("file", "audit/counts.parquet"), ("file", "audit/samples.parquet"), ("file", "audit/issues.parquet"),
                ("dir", "tables/<t>/"), ("file", "table.parquet, issues.parquet"), ("file", "variants/<v>/variants.parquet")],
         writes=[("dir", "runs/<r>/analyses/<a>/report/"), ("file", "report.md"), ("file", "metrics, probes, coverage"), ("file", "audit_counts, issues"),
                 ("file", "reader_share, reader_validation"), ("file", "matrix_<measure> (one per measure)"), ("note", "each table in .csv and .parquet")]),
    dict(num="", title="Med package", sub="Built on Core: 11 raw sources become tables, then the tables of the paper",
         call="run.sh → paper.py → cli.py: do_run(), do_analyze()", med=True,
         rows=[M + "run.sh", M + "check.py", M + "cite.py", M + "prepare/__main__.py", M + "prepare/__init__.py", M + "prepare/<source>.py",
               M + "label_rules/<source>.py", M + "prepare/template.py", M + "label_rules/template.py", M + "prepare/common.py",
               M + "paper.py", M + "adapters/__init__.py", M + "adapters/open_clip_model.py", M + "adapters/chexzero.py"],
         reads=[("dir", "$MED_DATA/"), ("file", "<source>/: raw files"), ("file", "converted/: DICOM as PNG"),
                ("dir", "packages/med/configs/"), ("file", "questions.yaml"), ("file", "experiment.yaml"), ("file", "analysis.yaml")],
         writes=[("dir", "packages/med/data/tables/"), ("file", "<source>.parquet"), ("note", "becomes data: in experiment.yaml"),
                 ("dir", "runs/<r>/paper/"), ("file", "tables of the paper, paper.md"),
                 ("dir", "packages/med/"), ("file", "SOURCES.md, references.bib")]),
]

# Tools used by many steps: their links are listed when you click, not drawn.
SHARED = {S + "common.py", S + "config/load.py", S + "config/schema.py", S + "extensions.py",
          S + "data/checks.py", S + "tasks/__init__.py", S + "tasks/yes_no.py", S + "variants/prompts.py", AD + "contrastive.py"}

# Calls with no import: they happen at run time.
RUN_LINKS = [
    (S + "inference/runner.py", AD + "vlmevalkit.py", "generate"),
    (S + "inference/runner.py", AD + "huggingface.py", "generate"),
    (S + "inference/runner.py", AD + "contrastive.py", "score, embed"),
    (S + "inference/runner.py", AD + "fake.py", "generate, score, embed"),
    (M + "run.sh", M + "check.py", "python check.py"),
    (M + "run.sh", M + "prepare/__main__.py", "python -m prepare"),
    (M + "run.sh", M + "paper.py", "python paper.py"),
    (M + "prepare/__init__.py", M + "prepare/<source>.py", "importlib"),
]
# Labels easier to read than the raw list of imported names.
LABELS = {
    (S + "data/table.py", S + "data/checks.py"): "apply_roles, check_*",
    (S + "stats/shifts.py", S + "stats/metrics.py"): "stat_of, counter…",
    (M + "prepare/__main__.py", M + "prepare/common.py"): "write_source…",
}


def internal_imports(path: Path) -> dict:
    found = {}

    def resolve(parts, names):
        base = ROOT.joinpath(*parts)
        for name in names:
            if base.joinpath(name + ".py").exists():  # from . import checks
                target = base / (name + ".py")
                found.setdefault(target.relative_to(ROOT).as_posix(), [])
                continue
            target = base / "__init__.py" if base.is_dir() else base.with_suffix(".py")
            found.setdefault(target.relative_to(ROOT).as_posix(), []).append(name)

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in ("adapters", "prepare", "label_rules"):
                    resolve(("packages", "med", *alias.name.split(".")), [alias.name])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = path.parents[node.level - 1]
                module = node.module.split(".") if node.module else []
                parts = (*base.relative_to(ROOT).parts, *module)
            else:
                module = node.module.split(".")
                if module[0] == "vlmshift":
                    parts = ("src", *module)
                elif module[0] in ("prepare", "label_rules", "adapters"):
                    parts = ("packages", "med", *module)
                else:
                    continue
            resolve(parts, [alias.name for alias in node.names])
    return found


def symbols(path: str) -> list:
    """Functions, classes and methods of a file, read from the code. For a group, those of its files."""
    if path in GROUPS:
        return sorted({name for member in GROUPS[path] for name in symbols(member)})
    if not path.endswith(".py"):
        return []
    found = []
    for node in ast.parse((ROOT / path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef):
            found.append(node.name)
        elif isinstance(node, ast.ClassDef):
            found.append(node.name)
            found += [f"{node.name}.{m.name}" for m in node.body
                      if isinstance(m, ast.FunctionDef) and not m.name.startswith("__")]
    return found


# --- 1. Place everything, or stop --------------------------------------------------

placed = {path: (b, r) for b, band in enumerate(BANDS) for r, path in enumerate(band["rows"])}
assert set(placed) == set(ROLES), set(placed) ^ set(ROLES)
on_disk = {MEMBER.get(path, path) for path in (p.relative_to(ROOT).as_posix() for base in ("src", "packages")
                                                for p in (ROOT / base).rglob("*.py") if p.stat().st_size)}
on_disk.add(M + "run.sh")
assert on_disk == set(placed), sorted(on_disk ^ set(placed))

imports = {}
for path in sorted(on_disk):
    for member in GROUPS.get(path, [path]):
        if member.endswith(".py"):
            for target, names in internal_imports(ROOT / member).items():
                key = (path, MEMBER.get(target, target))
                imports[key] = list(dict.fromkeys([*imports.get(key, []), *names]))

cli, orchestrators = S + "cli.py", {band["rows"][0] for band in BANDS} | {S + "inference/check.py"}  # check has its own command
spine, branches, arcs = [], [], []


def same_band(a, b):
    return placed[a][0] == placed[b][0]


def classify(a, b, label, kind):
    is_orch = a == BANDS[placed[a][0]]["rows"][0]
    if a == cli and b in orchestrators:
        spine.append((a, b))
    elif a.startswith(M) and not b.startswith(M):
        pass  # Med uses Core (paper.py calls cli.py, written in the header of the band): listed when you click
    elif same_band(a, b) and is_orch:
        branches.append((a, b, label, kind))
    elif b in SHARED or a in SHARED and not same_band(a, b):
        pass  # shared tools: listed when you click, not drawn
    elif same_band(a, b):
        arcs.append((a, b, label, kind))
    else:
        raise SystemExit(f"Cannot draw this link: {a} -> {b}")


links = []
for (a, b), names in sorted(imports.items()):
    full = LABELS.get((a, b), ", ".join(names))
    classify(a, b, full, "import")
    links.append((a, b, ", ".join(names) or full, "import"))
for a, b, label in RUN_LINKS:
    classify(a, b, label, "run")
    links.append((a, b, label, "run"))

# --- 2. Coordinates ----------------------------------------------------------------

RAIL_X, RAIL_W = 28, 262
CARD_X, SPINE_X, TRUNK_X = 326, 348, 372
HELP_X, HELP_R = 552, 924
LANE_X, LANE_STEP, LABEL_X = 932, 7, 970
W = LABEL_X + 184
ROW_H, CARD_H, HEADER, LINE = 58, 50, 62, 16
SLOTS = {1: [-7], 2: [-7, 7], 3: [-14, -7, 7], 4: [-14, -7, 7, 14], 5: [-14, -7, 0, 7, 14]}


def block_height(items):
    return 26 + len(items) * LINE + (10 if items else 0)


y = 14
for band in BANDS:
    band["y"] = y
    band["top"] = y + HEADER
    reads_h = block_height(band["reads"])
    writes_h = block_height(band["writes"]) if band["writes"] and not band.get("writes_mid") else 0
    rail_h = reads_h + (writes_h + 10 if writes_h else 0)
    band["h"] = HEADER + max(len(band["rows"]) * ROW_H, rail_h) + 12
    y += band["h"] + 14
H = y


def row_y(path):
    b, r = placed[path]
    return BANDS[b]["top"] + r * ROW_H


def cy(path):
    return row_y(path) + CARD_H // 2


def card_x(path):
    return CARD_X if path in orchestrators else HELP_X


def esc(text):
    return html.escape(str(text), quote=True)


def text(x, y, content, cls, anchor=None, extra=""):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text class="{cls}" x="{x}" y="{y}"{a}{extra}>{esc(content)}</text>'


def pair(a, b):
    return f'data-a="{esc(a)}" data-b="{esc(b)}"'


parts = {k: [] for k in "bands spine branch arc rail card label".split()}

# Bands and their headers
for band in BANDS:
    cls = "band med" if band.get("med") else "band"
    anchor = f'step-{band["num"]}' if band["num"] else band["title"].split()[0].lower()  # target of the six steps
    parts["bands"].append(f'<rect class="{cls}" id="{anchor}" x="14" y="{band["y"]}" width="{W - 22}" height="{band["h"]}" rx="10"/>')
    title = f'{band["num"]} · {band["title"]}' if band["num"] else band["title"]
    parts["bands"].append(text(RAIL_X, band["y"] + 34, title, "t-band"))
    parts["bands"].append(text(SPINE_X + 14, band["y"] + 34, band["sub"], "t-sub"))
    if band["call"]:
        parts["bands"].append(text(W - 22, band["y"] + 34, band["call"], "t-call", "end"))

# The thread of the steps
stops = [band["rows"][0] for band in BANDS[:7]]
for prev, stop in zip(stops, stops[1:]):
    x, y1, y2 = SPINE_X, row_y(prev) + CARD_H, row_y(stop)
    parts["spine"].append(f'<path class="link spine" {pair(cli, stop)} d="M{x},{y1} V{y2 - 2}" marker-end="url(#ah-flow)"/>')


def split_label(label):
    if len(label) <= 23 or ", " not in label:
        return [label]
    names = label.split(", ")
    half = (len(names) + 1) // 2
    return [", ".join(names[:half]) + ",", ", ".join(names[half:])]


# Calls from the conductor of a step
for a, b, label, kind in branches:
    y0, y1 = row_y(a) + CARD_H, cy(b)
    dash = " run" if kind == "run" else ""
    parts["branch"].append(f'<path class="link call{dash}" {pair(a, b)} d="M{TRUNK_X},{y0} V{y1} H{HELP_X - 2}" marker-end="url(#ah-link)"/>')
    lines = split_label(label)
    parts["label"].append(text(TRUNK_X + 8, y1 - 5, lines[0], "t-fn link", extra=" " + pair(a, b)))
    if len(lines) > 1:
        parts["label"].append(text(TRUNK_X + 8, y1 + 13, lines[1], "t-fn link", extra=" " + pair(a, b)))

# Calls between files of the same step
for band_index in range(len(BANDS)):
    local = sorted((arc for arc in arcs if placed[arc[0]][0] == band_index),
                   key=lambda item: abs(placed[item[0]][1] - placed[item[1]][1]))  # shortest first
    lanes = []  # row ranges taken in each lane
    written = {}
    ends = {}
    for arc in local:
        ends.setdefault(arc[0], []).append(arc)
        ends.setdefault(arc[1], []).append(arc)
    for arc in local:
        a, b, label, kind = arc
        low, high = sorted((placed[a][1], placed[b][1]))
        for index, used in enumerate(lanes):
            if all(high < u[0] or low > u[1] for u in used):
                used.append((low, high))
                lane = index
                break
        else:
            lanes.append([(low, high)])
            lane = len(lanes) - 1
        x = LANE_X + lane * LANE_STEP
        ya = cy(a) + SLOTS[len(ends[a])][ends[a].index(arc)]
        yb = cy(b) + SLOTS[len(ends[b])][ends[b].index(arc)]
        dash = " run" if kind == "run" else ""
        parts["arc"].append(f'<path class="link call{dash}" {pair(a, b)} d="M{HELP_R},{ya} H{x} V{yb} H{HELP_R + 2}" marker-end="url(#ah-link)"/>')
        written.setdefault(a, []).append((label, b))
    for a, labels in written.items():  # one-line labels, evenly spaced
        for index, (label, b) in enumerate(labels):
            yy = cy(a) + 3.5 + (index - (len(labels) - 1) / 2) * 13
            parts["label"].append(text(LABEL_X, yy, label, "t-fn link", extra=" " + pair(a, b)))

# Files on disk
BLOCK_TITLES = {}
for index, band in enumerate(BANDS):
    name = f'Step {band["num"]}, {band["title"]}' if band["num"] else band["title"]
    for side, word in (("reads", "what it reads"), ("writes", "what it writes")):
        if band[side]:
            BLOCK_TITLES[f"{index}:{side}"] = f"{name}: {word}"


def block(x, y, width, heading, items, id):
    h = block_height(items)
    label = esc(f"{heading}: {BLOCK_TITLES[id]}")
    out = [f'<g class="node data" data-id="{id}" tabindex="0" role="button" aria-label="{label}">',
           f'<rect class="rail" x="{x}" y="{y}" width="{width}" height="{h}" rx="6"/>', text(x + 10, y + 17, heading, "t-head")]
    yy = y + 34
    for kind, content in items:
        cls = {"dir": "t-dir", "file": "t-file", "note": "t-note", STEP: "t-mem"}[kind]
        out.append(text(x + (22 if kind == "file" else 10), yy, content, cls))
        yy += LINE
    out.append("</g>")
    return "".join(out), h


for index, band in enumerate(BANDS):
    top, center = band["top"], band["top"] + CARD_H // 2
    orch = band["rows"][0]
    content, reads_h = block(RAIL_X, top, RAIL_W, "READS", band["reads"], f"{index}:reads")
    parts["rail"].append(content)
    edge = RAIL_X + RAIL_W
    read_link = pair(f"{index}:reads", orch)
    write_link = pair(f"{index}:writes", orch)
    parts["rail"].append(f'<path class="link disk" {read_link} d="M{edge},{center - 8} H{CARD_X - 2}" marker-end="url(#ah-disk)"/>')
    if band.get("writes_mid"):
        yy = top + ROW_H + 8
        content, _ = block(HELP_X, yy, HELP_R - HELP_X, "WRITES", band["writes"], f"{index}:writes")
        parts["rail"].append(content)
        parts["rail"].append(f'<path class="link disk" {write_link} d="M{HELP_X + 40},{top + CARD_H} V{yy - 2}" marker-end="url(#ah-disk)"/>')
    elif band["writes"]:
        yy = top + reads_h + 10
        content, _ = block(RAIL_X, yy, RAIL_W, "WRITES", band["writes"], f"{index}:writes")
        parts["rail"].append(content)
        parts["rail"].append(f'<path class="link disk" {write_link} d="M{CARD_X},{center + 8} H{edge + 18} V{yy + 18} H{edge + 2}" marker-end="url(#ah-disk)"/>')

# Files of the code
for path, role in ROLES.items():
    x, yy = card_x(path), row_y(path)
    width = HELP_R - x
    kind = "orch" if path in orchestrators else "shared" if path in SHARED else "helper"
    folder, name = path.rsplit("/", 1)
    parts["card"].append(
        f'<g class="node {kind}" data-id="{esc(path)}" tabindex="0" role="button" aria-label="{esc(path)}: {esc(role)}">'
        f'<rect x="{x}" y="{yy}" width="{width}" height="{CARD_H}" rx="6"/>'
        f'<text class="t-path" x="{x + 12}" y="{yy + 20}"><tspan class="dir">{esc(folder)}/</tspan><tspan class="name">{esc(name)}</tspan></text>'
        f'{text(x + 12, yy + 39, role, "t-role")}</g>'
    )

aria = "Map of the VLMShift files, from the command to the report, then the Med package."
svg = (
    f'<svg id="map" viewBox="0 0 {W} {H}" role="group" aria-label="{esc(aria)}" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    + "".join(
        f'<marker id="ah-{n}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="{s}" markerHeight="{s}" orient="auto-start-reverse">'
        f'<path class="m-{n}" d="M0,0 L8,4 L0,8 z"/></marker>'
        for n, s in (("flow", 4.5), ("link", 7), ("disk", 7))
    )
    + '</defs>'
    + "".join("".join(group) for group in parts.values())
    + '</svg>'
)

# --- 3. Link data ------------------------------------------------------------------

files = {path: {"u": [], "v": []} for path in placed}  # what each file uses, and what uses it
for a, b, label, kind in links:
    files[a]["u"].append([b, label, kind])
    files[b]["v"].append([a, label, kind])


# Explanations: each one is compared to the code before being published.
assert set(EXPLAIN) == set(files), sorted(set(EXPLAIN) ^ set(files))
assert {f"{b}:{side}" for b, side in BLOCKS} == set(BLOCK_TITLES), sorted({f"{b}:{side}" for b, side in BLOCKS} ^ set(BLOCK_TITLES))

items = {}
disk_links = {}
for (band_index, side), info in BLOCKS.items():
    id = f"{band_index}:{side}"
    for _, entries in info["cols"]:
        for path, _ in entries:
            assert path in files, path
            disk_links.setdefault(path, []).append([BLOCK_TITLES[id], "", "disk", id])
    items[id] = dict(
        t=BLOCK_TITLES[id], g=info["goal"], h="Contents", n=None,
        l=[[name, words] for name, words in info["items"].items()],
        c=[[heading, [[path, label, "import"] for path, label in entries]] for heading, entries in info["cols"]],
    )

for path, info in EXPLAIN.items():
    assert path.endswith("run.sh") or info["goal"], path
    found, given = symbols(path), info.get("fn", {})
    assert set(given) <= set(found), (path, sorted(set(given) - set(found)))
    if len(found) <= LIMIT:
        assert set(given) == set(found), (path, sorted(set(found) - set(given)))
    else:
        assert not given, path
    has_class = any(name[0].isupper() and "." not in name for name in found)
    columns = [["USES", files[path]["u"]], ["USED BY", files[path]["v"]]]
    if path in disk_links:
        columns.append(["ON DISK", disk_links[path]])
    items[path] = dict(
        t=path, g=info["goal"], h="Functions and classes" if has_class else "Functions",
        l=[[name, words] for name, words in given.items()],
        n=f"{len(found)} functions and classes: too many to list here." if len(found) > LIMIT else None,
        c=columns,
    )
