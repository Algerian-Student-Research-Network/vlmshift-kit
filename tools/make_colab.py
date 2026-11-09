"""Write examples/pope_mini/colab.ipynb, the notebook for Google Colab and Kaggle.

Run after changing a cell:  python tools/make_colab.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "examples" / "pope_mini" / "colab.ipynb"
REPOSITORY = "https://github.com/Algerian-Student-Research-Network/vlmshift-kit.git"

CELLS = [
    ("md", """# VLMShift on POPE mini

This notebook runs the demo of VLMShift on Google Colab or Kaggle, with no NVIDIA card of your own.

1. **CLIP**, on 100 photos of POPE: works on any runtime, in a few minutes.
2. **InstructBLIP-13B**, a model of the POPE paper, to compare with its published results: needs a GPU runtime.

**Before you start**, choose a GPU. On Colab: *Runtime > Change runtime type > T4 GPU*. On Kaggle: *Settings > Accelerator > GPU T4 x2*.

Run the cells one by one, from the top. If the runtime disconnects, run them again: answers already obtained are kept, as long as the workspace is on Google Drive (cell 2)."""),
    ("md", "## 1. Install VLMShift"),
    ("code", f"""import pathlib

REPOSITORY = "{REPOSITORY}"

if not pathlib.Path("examples/pope_mini").exists():  # not in the project folder yet
    if not pathlib.Path("VLMShift").exists():
        !git clone --depth 1 {{REPOSITORY}} VLMShift
    %cd VLMShift
# torch and transformers are already there: only the rest is installed.
%pip install -q -e ".[local]"
"""),
    ("md", """## 2. Keep the answers on Google Drive (optional)

A Colab runtime is erased when it disconnects. With `KEEP_ON_DRIVE = True`, the workspace, where every answer is stored as soon as it arrives, goes to your Google Drive: run the notebook again and it resumes where it stopped. It takes a few MB. The model weights stay on the runtime."""),
    ("code", """KEEP_ON_DRIVE = False

if KEEP_ON_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    kept = pathlib.Path("/content/drive/MyDrive/VLMShift/pope_mini_workspace")
    kept.mkdir(parents=True, exist_ok=True)
    workspace = pathlib.Path("examples/pope_mini/workspace")
    if not workspace.exists():
        workspace.symlink_to(kept, target_is_directory=True)
    print("Workspace ->", kept)
"""),
    ("md", "## 3. Fetch the photos and the questions\n\n100 photos of COCO and their 1,800 POPE questions: 18 MB, from their original sources."),
    ("code", "!python examples/pope_mini/fetch.py"),
    ("md", """## 4. Run CLIP

CLIP says which of two sentences fits a photo better: "a photo of a bird." or "a photo.". The first time, it downloads CLIP (600 MB)."""),
    ("code", "!vlmshift all examples/pope_mini/experiment.yaml examples/pope_mini/analysis.yaml"),
    ("code", """from IPython.display import Markdown, display
from ruamel.yaml import YAML


def last_report(model: str) -> pathlib.Path:
    \"\"\"The report.md of the last analysis of a run that asked this model.\"\"\"
    runs = pathlib.Path("examples/pope_mini/workspace/runs")
    found = [
        report for report in runs.glob("*/analyses/*/report/report.md")
        if model in [entry["name"] for entry in YAML(typ="safe").load(report.parents[3] / "experiment.yaml")["models"]]
    ]
    return max(found, key=lambda report: report.stat().st_mtime)


clip_report = last_report("clip")
display(Markdown(clip_report.read_text(encoding="utf-8")))
"""),
    ("md", "Compare with the paper: how many points of accuracy each model loses from random to the harder modes."),
    ("code", "!python examples/pope_mini/compare.py {clip_report}"),
    ("md", """## 5. Run InstructBLIP-13B, a model of the paper

This part needs a GPU runtime. The first time, it downloads the weights: 57 GB, so expect a long wait.

- **A T4 (15 GB)**, free on Colab: the model is shrunk to 4 bits to fit, as `paper.yaml` asks (`quantize: 4bit`). Its answers can differ a little from the full model.
- **A card of 32 GB or more**, such as an A100: the next cell removes `quantize` and runs the full model, as in the paper.
- **Kaggle with two T4**: same as a T4. The full model could be split over both cards: set `FULL_MODEL = True`."""),
    ("code", """import torch

FULL_MODEL = None  # None: decided from the card. True or False to choose.

if not torch.cuda.is_available():
    print("No GPU. On Colab: Runtime > Change runtime type > T4 GPU, then run the notebook again.")
else:
    memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 2**30
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    full = memory >= 32 if FULL_MODEL is None else FULL_MODEL
    print(f"Cards: {names}, {memory:.0f} GB in all ->", "full model" if full else "model in 4 bits")
    paper = pathlib.Path("examples/pope_mini/paper.yaml").read_text(encoding="utf-8")
    if full:
        paper = paper.replace("    params: {quantize: 4bit}\\n", "")
    else:
        %pip install -q bitsandbytes
    pathlib.Path("examples/pope_mini/paper_here.yaml").write_text(paper, encoding="utf-8")
"""),
    ("md", "First ask four real questions, to see the answers and how long the full run would take:"),
    ("code", """if torch.cuda.is_available():
    !vlmshift check examples/pope_mini/paper_here.yaml
"""),
    ("md", "If the answers look right (\"Yes\" or \"No\"), run everything: 1,029 questions."),
    ("code", """if torch.cuda.is_available():
    !vlmshift all examples/pope_mini/paper_here.yaml examples/pope_mini/analysis.yaml
    paper_report = last_report("instructblip-13b")
    display(Markdown(paper_report.read_text(encoding="utf-8")))
"""),
    ("md", """Each measure next to the published one: accuracy, recall, specificity and yes ratio, in each mode. `inside` says whether the published value is in the likely range. Expect a few points of difference: the paper used 500 photos, read the answers in its own way, and ran the full model."""),
    ("code", """if torch.cuda.is_available():
    !python examples/pope_mini/compare.py {paper_report}
"""),
]


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(text) if kind == "md" else new_code_cell(text.rstrip("\n")) for kind, text in CELLS]
    for number, cell in enumerate(cells):
        cell.id = f"cell-{number}"  # nbformat draws random ids: the same notebook must give the same file
    notebook = new_notebook(cells=cells)
    notebook.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


def main() -> None:
    notebook = build()
    nbformat.validate(notebook)
    TARGET.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8", newline="\n")
    print(f"Notebook -> {TARGET}")


if __name__ == "__main__":
    main()
