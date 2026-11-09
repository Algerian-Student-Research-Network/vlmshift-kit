# VLMShift documentation

Everything the tool can do, one topic at a time. New here? Start with the [tutorial](tutorial.md).

**Contents**

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [Words to know](#2-words-to-know)
3. [Your data](#3-your-data)
4. [Describing an experiment](#4-describing-an-experiment-experimentyaml)
5. [Models](#5-models)
6. [Reading the answers](#6-reading-the-answers)
7. [Commands](#7-commands)
8. [Describing an analysis](#8-describing-an-analysis-analysisyaml)
9. [What is measured](#9-what-is-measured)
10. [The latent space probe](#10-the-latent-space-probe)
11. [What the tool writes](#11-what-the-tool-writes)
12. [Issues](#12-issues)
13. [Resuming and reproducing](#13-resuming-and-reproducing)
14. [Extending the tool](#14-extending-the-tool)
15. [Limits of this version](#15-limits-of-this-version)

## 1. The idea in one minute

You show a model the same image with the same question, and you change one thing. Here, the language:

| Image | English | French | Moved? |
|---|---|---|---|
| `img1.png` | yes | no | yes |
| `img2.png` | no | no | no |
| `img3.png` | yes | yes | no |

VLMShift turns this into numbers:

- **Does the model do worse?** The balanced accuracy goes from 0.90 to 0.75.
- **Does it change its mind?** 12% of its answers moved.
- **How sure are we?** Each number comes with an interval.
- **What is missing?** The report says what could not be measured, and why.

## 2. Words to know

| Word | Meaning | Example |
|---|---|---|
| **Sample** | An image with its question, before any change | `img1.png` and its question |
| **Variant** | One version of a sample | The French version of that question |
| **Axis** | What you compare | `language`, `site`, `quality` |
| **Level** | A value of an axis | `EN`, `FR` |
| **Controlled axis** | You changed the same sample on purpose, so samples can be compared face to face | The question was translated |
| **Observed axis** | The groups already exist in the data, so samples are different | The image comes from site A or site B |
| **Unit** | A group of rows that are not independent of each other | All the images of one subject |
| **Measure** | One comparison you ask for | `language`, or `language` within each `site` |
| **Run** | One execution of `experiment.yaml` | |
| **Analysis** | One set of computations on a run, described by `analysis.yaml` | |

## 3. Your data

Your data is one table, in CSV or Parquet, with **one row per variant**. Column names are free: `roles` says which column plays which part.

### Start from a folder of images

You do not have to write the table by hand. `vlmshift init` makes it from a folder, with a starting `experiment.yaml` and `analysis.yaml` next to it:

```bash
vlmshift init images --question "EN=Is there a dog in this image? Answer yes or no." --question "FR=Y a-t-il un chien dans cette image ? Répondez par oui ou non."
```

- Give one `--question` per level to compare, at least two. Each one reads `LEVEL=text`.
- `--axis` names the condition. Default: `language`.
- A first subfolder named `positive`, `negative` or `unknown` gives the label of its images. Any other image gets `unknown`.
- Each image starts as its own subject. Edit the `subject` column if several images belong together.
- An existing file is never replaced.

### Writing the table yourself

```csv
image,question,label,language,site,subject
images/img1.png,Is there a circle in this image? Answer yes or no.,positive,EN,A,subj1
images/img1.png,Y a-t-il un cercle dans cette image ? Répondez par oui ou non.,positive,FR,A,subj1
images/img2.png,Is there a circle in this image? Answer yes or no.,negative,EN,A,subj1
images/img2.png,Y a-t-il un cercle dans cette image ? Répondez par oui ou non.,negative,FR,A,subj1
```

| Column | Required | What it holds |
|---|---|---|
| Image | Yes | Path of the image, relative to the table file |
| Question | For generative models | The text of the question, or use `task.template` |
| Label | Yes | `positive`, `negative` or `unknown` |
| One axis | Yes, at least one | The levels you compare |
| Unit | No | Which rows belong together. Default: each sample on its own |
| Positive and negative text | For contrastive models | Two sentences, or use `task.contrastive` |
| Sample ID | No | Links variants. Computed if absent |
| Any other column | No | Kept. If a template uses it, it is part of the question and may change between the variants (a word translated, for example). Otherwise it is part of the sample: the rows of one sample must have the same value in it |

Good to know:

- **Label `unknown`**: the row is left out of accuracy, and kept for answer change. Never write `negative` when you do not know.
- **Sample ID**: the tool links two rows when all their values are the same, apart from the question, the sentences and the controlled axes. Give your own `sample_id` column if variants use different image files, such as an image you blurred yourself. When several questions are asked about one image, such as one per concept, the columns of the template tell them apart and stay in the sample.
- **Columns named after a role**: a column called `sample_id`, `unit` or `original_id` takes that role without being declared. An `id` column of your dataset is kept as `original_id`, since VLMShift writes its own `id`. One column can serve two roles, such as a case number for `sample_id` and `unit`.
- **Numeric column as an axis**: an observed axis with more than 4 values, such as an age, is cut into four quartiles, in the audit, the coverage and the measures. Each one is named by its rank then its interval, such as `Q1 (4.999, 16.25]`: write that name to choose it as `reference`. A controlled axis keeps its own levels.
- **Several files**: `data` can be a list. The files are put end to end and must have the same columns: the import stops otherwise.
- **CSV from a spreadsheet**: commas, or semicolons as written by a spreadsheet set to French; UTF-8, or Windows-1252 as written by Excel. The first line tells the separator.

### What is checked at import

| Problem | What happens |
|---|---|
| Image not found | The row is removed, with a warning |
| Two identical rows | One is removed, with a warning |
| Invalid or empty label | The table is refused |
| One sample with several units or labels | The table is refused |
| Two questions of one sample at the same levels of the controlled axes | The table is refused: their pairs would be chosen at random |
| Image with more than 8 bits per pixel, common for X-rays | A warning. Contrastive models scale it; convert it to 8 bits for the others |
| Column named in the configuration but absent | The tool stops |

## 4. Describing an experiment: `experiment.yaml`

This file says what to test. Changing the data, the question or the models means asking the models again. Only what really changed is recomputed.

A complete example:

```yaml
schema_version: 1
data: my_table.csv
workspace: ./workspace

roles:
  unit: subject

axes:
  language: {type: controlled}
  site:     {type: observed}

task:
  type: yes_no
  answers:
    yes: ["yes", "oui"]
    no:  ["no", "non"]

models:
  - {name: qwen, adapter: vlmevalkit, id: Qwen2.5-VL-7B-Instruct}

inference:
  temperature: 0
  max_new_tokens: 16
```

All paths are relative to the YAML file. An unknown key is an error, so a typo never goes unnoticed.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `schema_version` | Yes | | Always `1` |
| `data` | Yes | | One path, or a list of paths |
| `workspace` | No | `./workspace` | Where results go |
| `roles.image`, `.question`, `.label` | No | Same name | Column for each role |
| `roles.unit` | No | Sample ID | Column of the independent unit |
| `roles.sample_id` | No | Computed | Column linking variants |
| `roles.original_id` | No | None | Your own row identifier, kept and checked |
| `roles.positive_text`, `.negative_text` | No | None | Columns of contrastive sentences |
| `axes.<column>.type` | Yes, one axis at least | | `controlled` or `observed` |
| `task.type` | Yes | | `yes_no`, the only type for now |
| `task.template` | No | Question column | Text with `{column}` holes |
| `task.answers` | No | See [section 6](#6-reading-the-answers) | Accepted words, or a path to a YAML file with the same shape |
| `task.reader_llm` | No | None | Small model that reads answers |
| `task.contrastive` | For contrastive models without sentence columns | None | `positive_template` and `negative_template` |
| `models` | Yes | | See [section 5](#5-models) |
| `inference.temperature` | No | `0` | `0` means repeatable answers |
| `inference.max_new_tokens` | No | `16` | Maximum answer length |
| `inference.retries` | No | `2` | Attempts before the `error` status |

Using a template instead of a question column:

```yaml
task:
  type: yes_no
  template: "Is there a {concept} in this image? Answer yes or no."
```

The tool writes the exact text for each row at step 3 and stores it. You can always check what a model really read in `variants.parquet`.

## 5. Models

Each entry of `models` has a unique `name`, shown in the results, and an `adapter`, which says how to call it.

| Key | Required | Meaning |
|---|---|---|
| `name` | Yes | Your name for this model |
| `adapter` | Yes | `huggingface`, `vlmevalkit`, `contrastive`, or a name you registered |
| `id` | No | Catalogue or Hugging Face name. Default: `name` |
| `revision` | No | Exact version of the weights |
| `params` | No | Overrides `inference` for this model |
| `embeddings` | No | `true` keeps image vectors for the probe |

### Generative models, through Hugging Face

These models answer with text, and the transformers library runs them by itself: InstructBLIP, LLaVA, Qwen-VL, SmolVLM and the other models of the Hugging Face hub tagged *image-text-to-text*. Nothing more to install on Google Colab or Kaggle:

```yaml
models:
  - {name: instructblip, adapter: huggingface, id: Salesforce/instructblip-vicuna-13b, revision: 91271ed4920607b735c9925dfa495de6d012d16d}
```

Install `environments/huggingface.txt` first, or `pip install -e ".[local]"`.

| Setting in `params` | Values | Default |
|---|---|---|
| `quantize` | `8bit` or `4bit`: the weights take 2 or 4 times less memory. Needs an NVIDIA card and `bitsandbytes` | Full weights |
| `dtype` | `float16`, `bfloat16` or `float32` | `float16` on a card, `float32` on a processor |

Things to know:

- **The question**: a model with a chat template (LLaVA, Qwen, SmolVLM) gets the question inside it, as it was trained. A model without one (InstructBLIP) gets the question as it is.
- **Several cards**: a model too large for one card is spread over all of them.
- **Shrunk weights**: answers can differ a little from the full model, and a `model_note` issue says so in the report.
- **Without a card**: it works on a processor, but a large model is very slow. `vlmshift check` gives the time of the full run.
- **No card of your own?** See [Colab and Kaggle](colab.md).

### Generative models, through VLMEvalKit

These models answer with text. Any model of the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) catalogue works by its name, with no code to write:

```yaml
models:
  - {name: qwen, adapter: vlmevalkit, id: Qwen2.5-VL-7B-Instruct}
```

Install the environment first. The [NVIDIA server guide](server.md) gives every step:

```bash
pip install -r environments/vlmevalkit.txt
```

Things to know:

- **Graphics card**: many local models of the catalogue require an NVIDIA card.
- **Exact question**: VLMShift calls the model directly, so it reads exactly the text written at step 3.
- **Settings**: `temperature` and `max_new_tokens` are only passed on if the model can take them. If a model refuses a setting, or receives it without declaring it, a `model_note` issue says so. Check it before trusting a run.
- **Versions**: write `revision` for a result you want to publish. Without it, the weights can change between two runs. A model that cannot be pinned gives an error.
- **API models**: their failure sentence is treated as a real failure, so it gives `error`, never `INVALID`.

### Contrastive models

Models such as CLIP do not write text. They compare an image to two sentences, and the closest sentence gives the decision.

```yaml
task:
  type: yes_no
  contrastive:
    positive_template: "a photo showing a {concept}"
    negative_template: "a photo with no {concept}"

models:
  - {name: clip, adapter: contrastive, id: openai/clip-vit-base-patch32, embeddings: true}
```

To vary the language, use the `positive_text` and `negative_text` columns, with one row per language. Install `environments/contrastive.txt` first.

### Try the models before a long run

```bash
vlmshift check experiment.yaml
```

This asks each model up to four real questions, with different levels of the axes. It prints:

- the graphics cards and the main library versions
- each question, the raw answer and how it is read
- any `model_note`, and answers the reader does not understand
- an estimate of the length of the full run

No answer is kept, so the check changes no result. The command ends with an error if a model gave no answer. See [what to look for](server.md#4-check-before-the-long-run).

A long run then prints one line per minute for each model, with the errors so far and the time left.

### Models that need different environments

Different model families can need different library versions. Run each one in its own environment. Both commands fill the same run:

```bash
vlmshift run experiment.yaml --only model-a
```

```bash
vlmshift run experiment.yaml --only model-b
```

`run_log.json` keeps the library versions of each model. Until every model is done, the report opens with "Partial run".

### Fake models

`fake_generative` and `fake_contrastive` follow a fixed rule and need no GPU. They power the tests, with the tiny dataset of `tests/fixtures/toy/`, and only know its eight images.

## 6. Reading the answers

A generative model writes free text. A reader turns it into `yes` or `no`.

| You declare | How answers are read |
|---|---|
| `answers` only | By the list |
| `reader_llm` only | By the small LLM |
| Both | The list first, then the small LLM for what remains |
| Neither | By a default list: `yes` and `no` |

**The list** ignores case, and spaces and punctuation at both ends. So `Yes.` matches `yes`, and `Yes, there is one` does not.

**The small LLM** is a local language model that classifies an answer as `yes`, `no` or `unclear`:

```yaml
task:
  type: yes_no
  answers: {yes: ["yes"], no: ["no"]}
  reader_llm:
    model: <Hugging Face name>
    validation: annotated_answers.csv
```

A reader could understand one language less well, and that would look like a shift. Three safeguards exist:

- The `decision_source` column says whether each decision came from the list or the LLM.
- The report shows the share of decisions made by the LLM.
- `validation` measures its accuracy language by language, on answers annotated by hand (columns `response`, `language`, `expected`). Without this file, the report says the reader is not validated.

Each row ends with one of three statuses:

| Status | Meaning |
|---|---|
| `valid` | The answer was read |
| `INVALID` | The model answered, but nothing readable came out |
| `error` | A technical failure: the model never answered |

Raw answers are stored, so changing the reader does not call the models again.

## 7. Commands

| Command | What it does |
|---|---|
| `vlmshift init <folder> --question LEVEL=text ...` | Makes a table and two YAML files from a folder of images |
| `vlmshift import experiment.yaml` | Step 1: table and checks |
| `vlmshift audit experiment.yaml` | Steps 1 and 2: prints the counts, no model needed |
| `vlmshift prepare experiment.yaml` | Steps 1 to 3: exact texts, no model needed |
| `vlmshift check experiment.yaml` | Steps 1 to 3, then a few real questions to each model. Keeps no answer |
| `vlmshift run experiment.yaml` | Steps 1 to 4: asks the models |
| `vlmshift check` or `run`, with `--only <model>` | The same, for some models only |
| `vlmshift analyze analysis.yaml --run <folder>` | Steps 5 and 6: statistics and report |
| `vlmshift all experiment.yaml analysis.yaml` | Everything |

Each command redoes the previous steps if needed and can be run again safely. Each step prints its issues, one line per kind, starting with `!`.

## 8. Describing an analysis: `analysis.yaml`

This file says what to compute on a run. Rerun it as often as you like: it never calls a model.

```yaml
schema_version: 1
run: runs/to-replace

axes:
  language: {reference: EN}

measures:
  - {name: language, compare: language}
  - {name: site, compare: site}
  - {name: language_by_site, compare: language, split_by: [site]}

stats:
  bootstrap: 2000
  min_units: 30
```

`run` is replaced by `vlmshift all`, or by `--run` with `vlmshift analyze`.

| Key | Default | Meaning |
|---|---|---|
| `run` | Required | The run to analyse |
| `axes.<axis>.reference` | None: all pairs | The level the others are compared to |
| `axes.<axis>.levels` | All levels | Which levels to keep, in the measures that compare or split by this axis. Levels on an axis that no measure uses are an error |
| `measures` | One per axis | The comparisons you want |
| `measures[].compare` | Required | The axis to compare |
| `measures[].split_by` | Nothing | Redo the comparison within each level of these axes |
| `stats.bootstrap` | `2000` | Number of random draws, at least 100 |
| `stats.alpha` | `0.05` | Risk accepted for significance |
| `stats.seed` | `0` | Same seed, same numbers |
| `stats.min_units` | `30` | Below this number of units: not estimable |
| `latent.targets` | None | Columns to probe, see [section 10](#10-the-latent-space-probe) |
| `latent.folds` | `5` | Number of cross-validation folds |
| `report.notice` | None | A sentence shown at the top of the report |

## 9. What is measured

### In each cell

A cell is one level, or one combination of levels.

| Name in `metrics.csv` | Metric | Plain meaning |
|---|---|---|
| `sens` | Sensitivity | Among real positives, how many does the model find? |
| `spec` | Specificity | Among real negatives, how many does it recognise? |
| `ba` | Balanced accuracy | The average of the two |
| `yes_rate` | Yes rate | How often it says yes, right or wrong |
| `inv_rate` | INVALID rate | How often its answer cannot be read |

### Between two levels

| Name in `metrics.csv` | Question | Needs labels | Works on |
|---|---|---|---|
| `delta_ba` (`delta_sens`, `delta_spec`) | Does the model do worse? | Yes | Both axis types |
| `delta_yes_rate` | Does it say yes more often? | No | Both axis types |
| `delta_inv_rate` | Does it stop following the format? | No | Both axis types |
| `chg` (`chg_yes_no`, `chg_no_yes`) | Does it change its mind on the same samples? | No | Controlled axes |

A delta is the metric at level b minus the metric at level a. A negative `delta_ba` means the model does worse at level b.

### Two modes

Each number is computed twice, so an unreadable answer cannot hide a problem:

| Mode | An `INVALID` answer counts as |
|---|---|
| `valid` | Left out |
| `strict` | A wrong answer, or a change |

`error` rows are never counted as answers.

### Fair comparisons

On a **controlled axis**, only samples present at both levels are compared, face to face. The samples left out are counted in `n_excluded`. On an **observed axis**, the groups contain different images, so there is no face-to-face comparison and no answer change.

### How sure is a number?

The tool draws **units** at random, with replacement, and recomputes each number 2000 times. It draws units and not rows, because two images of one subject are not independent. This gives an interval around each number.

A shift is **significant** if its corrected interval does not contain 0. The correction (Bonferroni) accounts for the number of comparisons `m`. Without it, comparing many levels would produce false alarms. `ci_low` and `ci_high` are corrected, and `ci95_low` and `ci95_high` are uncorrected and only exploratory.

A number is **not estimable** when it has nothing to rest on: no positive case, or fewer units than `min_units`. Each number counts only the units it rests on: a sensitivity counts the units with a positive case, a balanced accuracy the units with a positive case and those with a negative one, whichever are fewer. A cell of 40 units with a single positive image gives a sensitivity on 1 unit, so not estimable. It is shown as such, and never as 0.

## 10. The latent space probe

The probe asks: can a column, such as the site, be guessed from the way a model represents an image? If so, the model holds this information and might use it.

```yaml
latent:
  targets: [site, label]
  folds: 5
```

It needs a model with `embeddings: true`. A small classifier trains on some units and is tested on the others. Its score is compared to a noise floor, obtained on shuffled values, and only the part above this floor counts. A probe that succeeds shows that the information is there, not that the model uses it.

## 11. What the tool writes

```text
workspace/
├── tables/<table>/
│   ├── table.parquet                 The checked table
│   ├── issues.parquet                Issues of step 1
│   ├── audit/
│   │   ├── counts.parquet, .csv      Counts by level
│   │   ├── samples.parquet, .csv     Variants of each sample
│   │   └── issues.parquet            Issues of step 2
│   └── variants/<variants>/
│       ├── variants.parquet, .csv    The exact texts sent to the models
│       └── issues.parquet            Issues of step 3
├── cache/<model fingerprint>/        Raw answers and failures of a model, reused by every table and run
├── reader_cache/                     Answers already read by the small LLM
└── runs/<run>/
    ├── experiment.yaml               Frozen copy
    ├── audit/                        The audit of this experiment, copied from its table
    ├── results.parquet, .csv         One row per variant and model: raw answer, decision, status
    ├── embeddings.parquet            Image vectors, if requested
    ├── reader_validation.parquet     Accuracy of the small LLM, if a validation set is given
    ├── run_log.json                  Versions, dates, settings, fingerprints
    ├── issues.parquet                Issues of steps 3 and 4
    └── analyses/<analysis>/
        ├── analysis.yaml             Frozen copy
        ├── metrics.parquet           All the numbers
        ├── probes.parquet            Probe results
        ├── issues.parquet            Issues of step 5
        └── report/                   What you read
```

The summary of `report.md` gives the main significant shifts in plain sentences, largest first, and warns when they rest on fewer than 30 units. It names the levels where fewer than half of the answers can be used, such as a language the model does not speak. When nothing could be tested, it says so instead of reporting no shift. The `report/` folder also holds these tables, each in CSV and Parquet:

| Table | Content |
|---|---|
| `metrics` | Every number, with its interval and counts |
| `matrix_<measure>` | One measure as a cross table, ready to read |
| `coverage` | For each model, in total and by level of each axis: rows expected, valid, INVALID, failed, missing |
| `audit_counts` | The counts of step 2 |
| `probes` | Probe results |
| `reader_share` | Share of decisions made by the small LLM |
| `reader_validation` | Its accuracy by language |
| `issues` | Every issue of every step |

## 12. Issues

The tool never fixes a problem silently. It prints it in the terminal, then lists it in `issues.csv`. An `error` stops the step, and a `warning` lets it continue.

| Code | Step | Level | Meaning | What to do |
|---|---|---|---|---|
| `label_invalid` | 1 | error | A label is empty or unknown | Fix the label, or write `unknown` |
| `sample_many_units` | 1 | error | One sample has several units | Check the `unit` column, or give a `sample_id` |
| `sample_many_labels` | 1 | error | One sample has several labels | Check the labels |
| `id_collision` | 1 | error | Two different rows got the same identifier | Check the table |
| `image_missing` | 1 | warning | Image not found: row excluded | Check the path |
| `duplicate_line` | 1 | warning | Identical row removed | Nothing, or clean the table |
| `original_id_invalid` | 1 | warning | Your identifier is empty or repeated | Fix this column |
| `same_levels` | 1 | error | Two rows of one sample share the levels of the controlled axes | Add a column that tells the questions apart, such as `concept_id` |
| `image_16bit` | 1 | warning | Images have more than 8 bits per pixel | Convert them to 8 bits, unless every model is contrastive |
| `axis_single_level` | 1 | warning | An axis has one level: no comparison possible | Check the axis |
| `low_count` | 2 | warning | A level has fewer than 30 units | Add data, or read this number carefully |
| `empty_template_value` | 3 | warning | A template hole has no value: row excluded | Fill the column |
| `model_error` | 4 | warning | A row failed after all attempts | Rerun: failed rows are tried again |
| `embed_error` | 4 | warning | A vector could not be computed | Rerun |
| `model_note` | 4 | warning | A setting was refused or may be ignored | Check the model before trusting the run |
| `model_pending` | 4 | warning | A model has not finished | Run it, or accept a partial result |
| `vectors_pending` | 4 | warning | Image vectors are missing for a model with `embeddings: true` | Run it again: only the missing vectors are computed |
| `csv_locked` | 2 to 6 | warning | A CSV copy is open in another program and was not updated | Close it and run the command again |
| `constant_model` | 5 | warning | A model always gives the same answer | Check its answers |
| `probe_skipped` | 5 | warning | A probe could not run | Read the message |
| `probe_not_estimable` | 5 | warning | Too little data for a probe | Add data |

## 13. Resuming and reproducing

**Resuming.** Each answer enters a cache as soon as it arrives. After a crash, run the same command again: finished work is skipped, and rows in `error` are tried again. If every variant of a model fails, `run` and `all` end with an error: the run folder is written, but there is nothing to analyse.

**Fingerprints.** A table, a set of variants, a run and an analysis each get a fingerprint computed from their content. Same content, same fingerprint, same folder. Change one value and you get a new folder, and the old one stays.

**What is reused.** The answers of a model are kept in `workspace/cache/`, one folder per model, version and settings. Inside, each answer is keyed by the question itself: the image and the exact text. So adding a model does not ask the others again; correcting a label, adding an axis or importing another table asks only the new questions; and changing the answer reader asks no model at all.

**What is recorded.** `run_log.json` holds library versions, dates, settings and fingerprints, per model. `experiment.yaml` and `analysis.yaml` are copied next to their results.

For a result to publish, fix the `revision` of each model and install the pinned files of `environments/`.

## 14. Extending the tool

### Add a model

Write a class that follows `Model`, register it under a name, and start the command line from your own script:

```python
# my_run.py
from vlmshift.cli import main
from vlmshift.extensions import register_adapter
from vlmshift.inference.base import Model


class MyModel(Model):
    kind = "generative"

    def generate(self, images, prompts, params):
        return ["yes" for _ in images]


register_adapter("my_model", MyModel)
main()
```

```yaml
models:
  - {name: mine, adapter: my_model}
```

```bash
python my_run.py all experiment.yaml analysis.yaml
```

| To write | When |
|---|---|
| `load()` | To load weights |
| `generate(images, prompts, params)` | Generative model: returns one text per image |
| `score(images, positives, negatives)` | Contrastive model: returns two scores per image |
| `embed(images)` | With `supports_embed = True`: returns one vector per image |

Raise `ModelError` on a technical failure. The engine tries again, then sets the `error` status.

### Add a package

A package is a folder of `packages/` that prepares data and configurations on top of Core. `packages/med/` is the example: it reproduces a medical image benchmark, 11 chest X-ray sources asked about 35 concepts in 7 languages by 10 models, with one script per source, label rules, its own models and the tables of the paper. Copy `prepare/template.py` to add a source.

Data you may not share stays out of the repository. Each source script carries a card, `SOURCE`: name and version, official page, access level, steps, licence, SHA-256 of the raw file or folder when it is known, a note, BibTeX. `check.py` uses it to say where to get a missing source and to warn about another version. `cite.py` turns the cards into `SOURCES.md`, to find the data, and `references.bib`, to cite it.

### Rebuild the site

The site in `docs/` is generated from the code and from these Markdown files, and a test fails when it is out of date:

```bash
python tools/site/build.py
```

## 15. Limits of this version

- **Generative models have not run on an NVIDIA card yet.** SmolVLM-256M ran through the real VLMEvalKit on a processor only, after changing one line of VLMEvalKit for that test. The small LLM is only tested through tiny random models. Contrastive models are tested on the real CLIP. Do a first real run before trusting a generative result.
- **One task type**: yes/no questions.
- **Long format only**: one row per variant.
- **Probe**: contrastive models only, because VLMEvalKit gives no access to the vectors of generative models.
- **Med package**: its 11 sources are tested on fake raw folders shaped like the real ones. Only Montgomery has run on the real data, and none of its 10 models on its real weights. Its 35 concepts and their translations are rebuilt, not those of the paper.

The [design](design.md) describes what is planned.
