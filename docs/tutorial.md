# Tutorial: your first experiment

You will install VLMShift, run a demo, read a report, then set it up on your own images. No programming is needed.

**What you need:** Python 3.10 or later, and a terminal. To check your version:

```bash
python --version
```

## 1. Install

Download the project and enter its folder:

```bash
git clone https://github.com/Algerian-Student-Research-Network/vlmshift-kit.git VLMShift
```

```bash
cd VLMShift
```

Create a virtual environment. It keeps the libraries of VLMShift apart from the rest of your computer:

```bash
python -m venv .venv
```

Activate it. On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install VLMShift. The `local` part adds PyTorch and Transformers, which run CLIP for the demo on your processor:

```bash
pip install -e ".[local]"
```

Check that it works:

```bash
vlmshift --help
```

You should see the list of commands.

## 2. Run the demo

The demo uses POPE (Li et al., EMNLP 2023), a published benchmark of object hallucination: a model says an object is there when it is not. POPE asks "Is there a bird in the image?" about photos of COCO. For each photo, it asks about 3 objects that are in it and 3 that are not, and it picks the absent objects in three ways, the `mode` column. Here is one of the photos, with a bird, a person and a boat:

| Mode | How the absent object is picked | Absent objects asked about |
|---|---|---|
| `random` | Any object | fire hydrant, bed, bench |
| `popular` | Among the most common objects | dining table, car, chair |
| `adversarial` | Among those often seen with the objects of the photo | handbag, car, chair |

A model that answers from what it expects, rather than from what it sees, says yes more often as the absent object becomes more likely. The paper found this in most of the models it tested. The demo checks whether CLIP does the same, on 100 of the 500 photos of the paper.

First fetch the photos and the questions, once. The photos follow the terms of Flickr, so they are not stored in the project:

```bash
python examples/pope_mini/fetch.py
```

```text
Images: 100/100
Table: 1800 questions about 100 images -> ...\examples\pope_mini\data\pope_mini.csv
```

Then run every step. The first time, it downloads CLIP (600 MB). Then it takes a few minutes on a normal processor. To run it in your browser instead, open the notebook `examples/pope_mini/colab.ipynb` in Google Colab: see the [Colab and Kaggle guide](colab.md).

```bash
vlmshift all examples/pope_mini/experiment.yaml examples/pope_mini/analysis.yaml
```

Each line of the output is one step that finished:

```text
Table: 1800 rows, 1800 samples -> ...
Audit -> ...
Variants: 1800 rows -> ...
  clip: 312 of 1029, 0 errors, about 2 min left
  clip: 640 of 1029, 0 errors, about 74 s left
  clip: 960 of 1029, 0 errors, about 13 s left
Run -> ...
Report -> ...\report\report.md
```

Only 1,029 of the 1,800 questions reach CLIP: the questions about the objects of the photo are the same in the three modes, and an answer already obtained is never asked again. A line starting with `!` would be a warning; there is none here.

CLIP does not write answers. It says which of two sentences fits the photo better: "a photo of a bird." or "a photo.". The first one means yes. `experiment.yaml` sets these two sentences.

The last line gives the path of your report. Open `report.md` in any text editor.

## 3. Read the report

The **Summary** tells you what moved, in plain sentences, model by model:

```text
clip

- mode, random to adversarial: the model says yes more often, by 0.098 (likely between 0.062 and 0.135, 100 units).
- mode, random to adversarial: balanced accuracy falls by 0.098 (likely between -0.135 and -0.062, 100 units).
```

Piece by piece:

| Piece | Meaning |
|---|---|
| `clip` | The model |
| `mode, random to adversarial` | We compare the modes. `random` is the reference, set in `analysis.yaml` |
| `says yes more often, by 0.098` | CLIP says yes to about 10 more questions out of 100. Rates go from 0 to 1 |
| `balanced accuracy falls by 0.098` | The model does worse. Each mode has as many present objects as absent ones, so this is the accuracy of the paper |
| `likely between -0.135 and -0.062` | The likely range of the true value |
| `100 units` | The number of independent units behind this number: the 100 photos. The 18 questions about one photo are not independent, and `experiment.yaml` says so with `unit: image_id` |

Only significant shifts are listed. From random to popular, CLIP loses 0.030, but its likely range reaches 0: it is not listed, and `metrics.csv` keeps it.

The two numbers are equal for a reason. The questions about present objects are the same in every mode, so CLIP finds as many of them, and every extra yes is a mistake on an absent object.

### Compare with the paper

Give `compare.py` the path of `report.md`:

```bash
python examples/pope_mini/compare.py <path of report.md>
```

It shows how many points of accuracy each model loses from random, measured for CLIP and published for the five models of the paper (its table 3):

```text
         model   source        mode  drop  low  high
          clip measured     popular   3.0  0.3   5.7
          clip measured adversarial   9.8  6.5  13.0
  InstructBLIP    paper     popular   7.4
  InstructBLIP    paper adversarial  14.4
         LLaVA    paper     popular   2.0
         LLaVA    paper adversarial   3.7
     MiniGPT-4    paper     popular   9.5
     MiniGPT-4    paper adversarial  11.2
MultiModal-GPT    paper     popular   0.0
MultiModal-GPT    paper adversarial   0.0
     mPLUG-Owl    paper     popular   2.7
     mPLUG-Owl    paper adversarial   2.6
```

CLIP behaves like the models of the paper: it loses a little on popular, more on adversarial. LLaVA, mPLUG-Owl and MultiModal-GPT lose little for another reason: they say yes to almost everything, 95 to 100% of the time in the paper, so they are near chance in every mode.

To check the published numbers themselves, run `examples/pope_mini/paper.yaml` on an NVIDIA card: it asks the same questions to InstructBLIP-13B, a model of the paper. No card? The notebook `examples/pope_mini/colab.ipynb` does it on Google Colab: see the [Colab and Kaggle guide](colab.md). `compare.py` then puts each measure next to the published one, and says whether the published value is inside the likely range. Expect a few points of difference: the paper used 500 photos, reads the answers in its own way, and ran the full model where a small card needs it shrunk to 4 bits.

The next sections of the report give:

- **Audit**: what your data contains, level by level.
- **Results by measure**: where to find every number.
- **Latent space probe**: whether a column can be read in the vectors of a model.
- **Reading the answers**: how raw answers became yes or no.
- **Coverage limits**: what is missing, and every issue found.

The same folder holds all the tables as CSV files, which open in any spreadsheet.

## 4. Set up your own images

Create a new folder anywhere, and put your images in an `images` folder inside it. If you know the true answer, sort them into `positive` and `negative`:

```text
my_experiment/
└── images/
    ├── positive/        the answer is yes
    │   ├── photo1.png
    │   └── photo2.png
    └── negative/        the answer is no
        ├── photo3.png
        └── photo4.png
```

Images placed directly in `images/` get the label `unknown`. They still count for answer changes, but not for accuracy. Most formats work, such as PNG, JPEG, WebP and TIFF. Other formats, such as DICOM, must be converted first.

Enter your folder:

```bash
cd my_experiment
```

Then give the question once per condition. Here we compare English and French:

```bash
vlmshift init images --question "EN=Is there a dog in this image? Answer yes or no." --question "FR=Y a-t-il un chien dans cette image ? Répondez par oui ou non."
```

The tool tells you what it found:

```text
Table: 8 rows, 4 images (2 positive, 2 negative) -> table.csv
Next: read experiment.yaml, then run: vlmshift prepare experiment.yaml
```

Three files appear next to `images`. An existing file is never replaced.

| File | What it holds |
|---|---|
| `table.csv` | One row per image and per condition. It opens in any spreadsheet |
| `experiment.yaml` | What to test: the table, the condition, the model |
| `analysis.yaml` | What to compute on the results |

### Adapt the files

Open `experiment.yaml` and add the words for yes and no in your languages:

```yaml
task:
  type: yes_no
  answers:
    yes: ["yes", "oui"]
    no: ["no", "non"]
```

Open `analysis.yaml` and add these lines at the end:

```yaml
stats:
  min_units: 2
```

Below 30 units, numbers show as "not estimable" by default. We lower this limit only to try the tool on four images. For real results, remove these lines and use at least 30 units per level.

**Good to know:** in `table.csv`, the `subject` column says which images belong together. By default each image is its own subject. If several images come from one person or one object, give them the same value: they are not independent, and the statistics need to know.

## 5. Check your data, with no model

Run the first three steps. No model is called:

```bash
vlmshift prepare experiment.yaml
```

```text
Table: 8 rows, 4 samples -> ...
Audit -> ...
  ! low_count (2): language = EN: 4 units (threshold 30)
Variants: 8 rows -> ...
```

Here, 8 rows and 4 samples: four images, each with two languages. `low_count` is normal on four images.

Each line gives a folder. Two files are worth opening, in any spreadsheet:

- `counts.csv`, in the `audit` folder: how many images you have per level. `vlmshift audit experiment.yaml` also prints it in the terminal.
- `variants.csv`, in the variants folder: the exact text each model will read.

If something is wrong, you find out now:

| Message | What it means | Fix |
|---|---|---|
| `! image_missing` | An image was not found: its row was removed | Check the path in `table.csv` |
| `Error: File not found: ...table.csv` | The table is not where `data` says | Check the name in `experiment.yaml` |
| `Error: Columns not found for ...` | A column named in the configuration is absent | Check the column names of your table |
| `Error: Table rejected ... Invalid label` | A label is empty or misspelled | Use `positive`, `negative` or `unknown` |
| `Error: ... is not valid` | The YAML has an unknown key or a wrong value | Read the message: it names the key |

## 6. Run a real model

The model in `experiment.yaml` comes from the VLMEvalKit catalogue. Most of its local models, including this one, **need an NVIDIA graphics card**.

> Generative models have only run on a processor with this version, never on an NVIDIA card. Treat your first run as a test, and check a few raw answers yourself.

**Working on a remote server?** Follow the [NVIDIA server guide](server.md), which covers every step from copying your files to bringing the report back.

**On a computer with an NVIDIA card?** From the `VLMShift` folder, install the library:

```bash
pip install -r environments/vlmevalkit.txt
```

Then, from `my_experiment`, ask the model a few real questions:

```bash
vlmshift check experiment.yaml
```

It shows the graphics card, the answers, how they are read, and how long the full run would take. If they look right, run everything:

```bash
vlmshift all experiment.yaml analysis.yaml
```

The first run downloads the model, which can take a while. If it stops for any reason, run the same command again: answers already obtained are kept.

**No NVIDIA card?** Use the free cards of Google Colab or Kaggle: the [Colab and Kaggle guide](colab.md) gives every step. Contrastive models such as CLIP also run on a normal processor: see [contrastive models](documentation.md#contrastive-models).

## 7. Read your results

Open the `report.md` printed on the last line, as in [step 3](#3-read-the-report). Start with **Coverage limits**: it tells you whether each model answered everything, at each level. Then read the **Summary**. To see what a model really wrote, open `results.csv` in the run folder.

If you see `not estimable`, the number has too little data behind it. It is never replaced by 0.

To change the analysis, edit `analysis.yaml` and run the same command. Finished work is reused, so no model is called again.

## Where to go next

| I want to | Do this |
|---|---|
| Compare another condition | Add a column to `table.csv` and a line under `axes` |
| Test a second model | Add a line under `models`. The first model is not run again |
| Use a remote server | Follow the [NVIDIA server guide](server.md) |
| Compare within groups | Add `split_by` to a measure in `analysis.yaml` |
| Understand a metric | Read [what is measured](documentation.md#9-what-is-measured) |
| Use my own model | Read [add a model](documentation.md#add-a-model) |

The [documentation](documentation.md) describes every option. The [file guide](files.html) shows every file on a clickable map.
