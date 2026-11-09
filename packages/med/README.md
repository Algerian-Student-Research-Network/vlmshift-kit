# Med package

Medical image benchmark built on VLMShift: does the language of the question change the decisions of medical vision-language models, across chest X-ray domains? It only uses the interfaces of Core.

- **11 sources** of chest X-rays, from North America, Europe, Asia, Africa and Oceania ([SOURCES.md](SOURCES.md)). Each gives 300 patients, one image each, half with a positive label when there are enough.
- **35 concepts** with their RadLex identifier, asked in **7 languages**: English, French, Spanish, German, Arabic, Hindi, Chinese ([configs/questions.yaml](configs/questions.yaml)). English is the reference.
- **10 models**, each pinned to a commit ([configs/experiment.yaml](configs/experiment.yaml)). The 5 generative ones answer YES or NO in English, whatever the language. Anything else is INVALID, never NO. The 5 contrastive ones compare the image with a positive and a negative sentence.

## Run

```bash
MED_DATA=/path/to/the/data bash packages/med/run.sh
```

`run.sh` checks that everything is there, prepares the 11 sources, runs the models through Core, then writes the tables of the paper in `runs/<run>/paper/`:

| File | Content |
|---|---|
| `models.csv` | Per model: accuracy, F1, balanced accuracy, range between its best and worst language |
| `languages.csv` | Per language: accuracy and its gap with English, with a 95% interval from 10,000 paired draws; the same for balanced accuracy |
| `domain_language.csv` | Accuracy per source and language |
| `rankings.csv` | Kendall τ between the model rankings of two languages, in each source |
| `matched_cells.csv`, `cells.csv` | The cells kept by the complete-case rule, and every number per cell |

Only the cells where every model has an answer in every language are compared. Balanced accuracy has its own core: the cells with both classes. To rewrite the tables of a finished run without any model: `python paper.py --run <run folder>`.

## Data

The data is not in this repository, and must never be: each user asks for it at its source and accepts its licence. [SOURCES.md](SOURCES.md) says, for each source, where to ask, how, which file was used and how to cite it. [references.bib](references.bib) gathers the citations.

`python check.py configs/experiment.yaml` says what is missing and where to get it, and warns when a raw file is not the version of the benchmark. Only the code, the cards and the aggregated results go online: `packages/med/data/` and `workspace/` are ignored by git.

The CheXzero weights are not on Hugging Face: put `best_64_5e-05_original_22000_0.864.pt`, from the folder linked by github.com/rajpurkarlab/CheXzero, in `data/weights/chexzero/`.

## Labels

Each source codes its labels in its own way: `label_rules/<source>.py` turns that code into positive, negative or unknown. Only what a source states becomes a negative: a finding it does not mention stays unknown, and unknown is never asked. The choices for each source are in [docs/decisions.md](../../docs/decisions.md).

## Add a source

1. Copy `prepare/template.py` and `label_rules/template.py` under the name of the source.
2. Adapt `RAW`, `SOURCE` (its card: name and version, page, access, steps, licence, BibTeX), `prepare()` and the label rule.
3. Put the raw file in place and run `python check.py configs/experiment.yaml`: it gives the SHA-256 to write in `SOURCE`.
4. Add its table to `data:` in `configs/experiment.yaml`, and its concepts to `configs/questions.yaml`.
5. Run `python cite.py`: it writes `SOURCES.md` and `references.bib` from the cards.
