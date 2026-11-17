# Med package

Medical image benchmark built on VLMShift: does the language of the question change the decisions of medical vision-language models, across chest X-ray domains? It only uses the interfaces of Core.

> Research benchmark. It is not a medical device: no number it gives supports a diagnosis.

- **35 concepts** of the paper, `normal` included, with their RadLex identifier, in **7 languages**: English, French, Spanish, Hindi, Chinese, Arabic, German ([configs/questions.yaml](configs/questions.yaml)). English is the reference.
- **The wordings of the paper**: the canonical question, two paraphrases, the negated question, the pairs of sentences for contrastive models, four answer formats, and the questions in Latin letters ([configs/romanized.yaml](configs/romanized.yaml)). Nothing is reconstructed: a test rebuilds the 1,493 distinct questions of the October 2026 runs from these two files.
- **11 sources** of chest X-rays, from North America, Europe, Asia, Africa and Oceania ([SOURCES.md](SOURCES.md)).
- **10 models**, each pinned to a commit ([configs/experiment.yaml](configs/experiment.yaml)). The 5 generative ones answer in the format asked; anything else is INVALID, never NO. The 5 contrastive ones compare the image with a positive and a negative sentence.

## The terms are drafts

Each concept of `questions.yaml` has a `status`. The terms were drafted for the paper and no clinician who speaks each language has reviewed them yet, so all say `needs_validation`. While one does, `paper.py` and `replay.py` refuse to run and say why. With `--allow-unvalidated` they run, and the report opens with **DRAFT TERMS**. Once a concept is reviewed, set its `status` to `validated` and write who and when in `reviewed`.

## Three ways to run it

**The sample of the paper.** The study froze what it sent to the models in two evaluation plans. They name patients and images of sources that cannot be shared again, so they are not in this repository. Copy `eval_plan_core.csv` and `eval_plan_core_supp.csv` (folder `manifests/` of the drive of the study) to `data/paper/`, then:

```bash
cd packages/med
export PYTHONPATH=.
python plans.py                       # one experiment per arm, in data/paper/arms/
python paper.py data/paper/arms/main/experiment.yaml data/paper/arms/main/analysis.yaml --allow-unvalidated
```

`plans.py` writes each text again from the bank and compares it with the plan: one difference stops everything. The plans give the path of each image on the machine of the study; `data/paper/image_roots.json` says where these folders are on yours, such as `{"/content/images/NIH": "D:/nih"}`.

| Arm ([configs/arms.yaml](configs/arms.yaml)) | What changes | Questions |
|---|---|---|
| `main` | The language, on 11 sources | 513 × 7 languages |
| `formats` | The answer format: YES/NO, 1/0, A/B | 228 × 7 × 3 |
| `negation` | The plain question and its negation, answered by 1/0 | 513 negated, 228 paired |
| `controls` | The image: blank, another patient, cropped border, standardized contrast, another site with the same label | 80 × 7 × 6 |
| `quality` | The image blurred or noised at five severities | 80 × 3 × 11 |
| `factorial` | The sentence and the term each in English or another language | 200 × 19 |
| `permutation` | A concept that is not that of the image | 228 × 7 × 2 |
| `transliteration` | Hindi, Chinese and Arabic in their script and in Latin letters | 228 × 3 × 2 |

All arms share one workspace: a question asked in one arm is not asked again in another.

**The numbers of the paper, without a GPU.** The study kept every answer of every model. Copy the files `pred__<model>__core.csv` and `pred__<model>__core_supp.csv` (folder `preds/by_model/`) to `data/paper/preds/`, then:

```bash
python replay.py main factorial --allow-unvalidated
```

`replay.py` puts the recorded answers where a run would have written them and runs the analysis and the report on them. It reads no image. Replayed on the supplementary run of CLIP ViT-L/14 and LLaVA-Med, it gives the frame, term and interaction effects of table 11 of the paper: -0.077, -0.074, 0.088 and -0.018, -0.018, 0.012.

**From the public sources.** `run.sh` prepares the 11 sources from their raw files, 300 patients each with one image, half with a positive label when there are enough, and runs the benchmark:

```bash
MED_DATA=/path/to/the/data bash packages/med/run.sh --allow-unvalidated
```

This is a new sample, not the one of the paper, and the raw files do not carry everything the annotated workbook of the study does: for JSRT, the public file names only say whether there is a nodule, while the paper asks about lung cancer, granuloma and hamartoma.

## What is written

Core writes its report in `runs/<run>/analyses/<analysis>/report/` (see the [documentation](../../docs/documentation.md)). `paper.py` adds the tables of the paper in `runs/<run>/paper/`:

| File | Content |
|---|---|
| `models.csv` | Per model: accuracy, F1, balanced accuracy, range between its best and worst language |
| `languages.csv` | Per language: accuracy and its gap with English, with a 95% interval from 10,000 paired draws; the same for balanced accuracy |
| `domain_language.csv` | Accuracy per source and language |
| `rankings.csv` | Kendall τ between the model rankings of two languages, in each source |
| `matched_cells.csv`, `cells.csv` | The cells kept by the complete-case rule, and every number per cell |

Only the cells where every model has an answer in every language are compared. Balanced accuracy has its own core: the cells with both classes. To rewrite the tables of a finished run without any model: `python paper.py --run <run folder>`.

Two audits of the paper are in `audits.py`: `python audits.py congruence <analysis folder>` gives the shift from English to the language of the country (Spanish in Spain, Chinese in China), and `python audits.py semantic <workbook.csv>` counts the distinct texts of each language column of the annotated workbook, which shows that these columns carry report details.

## Data

The data is not in this repository, and must never be: each user asks for it at its source and accepts its licence. [SOURCES.md](SOURCES.md) says, for each source, where to ask, how, which file was used and how to cite it. [references.bib](references.bib) gathers the citations.

`python check.py configs/experiment.yaml` says what is missing and where to get it, and warns when a raw file is not the version of the benchmark. Only the code, the cards and the aggregated results go online: `packages/med/data/` and `workspace/` are ignored by git.

The CheXzero weights are not on Hugging Face: put `best_64_5e-05_original_22000_0.864.pt`, from the folder linked by github.com/rajpurkarlab/CheXzero, in `data/weights/chexzero/`.

## Labels

Each source codes its labels in its own way: `label_rules/<source>.py` turns that code into positive, negative or unknown, for the concepts of the paper and with the label policy the paper gives to that source:

| Policy | Sources | Rule |
|---|---|---|
| Classes that exclude each other | Montgomery, Shenzhen, TB Pakistan, Bangladesh, Nigeria | Positive for the class of the image, negative for the others. A class outside the paper (COVID-19 in Nigeria) is unknown |
| Exhaustive findings | NIH, RANZCR | A finding or device not listed is negative. NIH labels were mined from reports: negative means "not reported" |
| Positives, and normal studies | PadChest-GR | A reported finding is positive. A normal study is negative for everything else. In an abnormal study, what is not reported is unknown |
| Positives only | PTX-498 | No negative: accuracy but no balanced accuracy |
| Votes | RICORD | Typical, indeterminate and atypical appearances are lung opacity; "negative for pneumonia" is its absence; a tie is unknown |

Unknown is never turned into negative, and unknown is never asked. The docstring of each rule gives its choices.

## Add a source

1. Copy `prepare/template.py` and `label_rules/template.py` under the name of the source.
2. Adapt `RAW`, `SOURCE` (its card: name and version, page, access, steps, licence, BibTeX), `prepare()` and the label rule.
3. Put the raw file in place and run `python check.py configs/experiment.yaml`: it gives the SHA-256 to write in `SOURCE`.
4. Add its table to `data:` in `configs/experiment.yaml`, its country to `COUNTRIES` in `prepare/common.py`, and its concepts to `configs/questions.yaml`.
5. Run `python cite.py`: it writes `SOURCES.md` and `references.bib` from the cards.
