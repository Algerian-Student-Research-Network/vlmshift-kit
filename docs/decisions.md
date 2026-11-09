# Decision log

Choices made during V1 development, where the design leaves some room.

## To be validated

| # | Topic | Situation | What was done |
|---|---|---|---|
| 1 | The 11 Med sources | The paper names them (table 1), not its harmonization code | One script per source, written from the paper and the official pages. The layouts of PTX-498, Bangladesh and JSRT were read in the directory of their archives; those of NIH, Shenzhen, Montgomery, Pakistan TB and Nigeria in their official listings. Those of PadChest-GR, RANZCR and RICORD come from third-party code: a wrong name stops with the columns found. **Only Montgomery has run on the real data**: 138 images, 80 negative and 58 positive, through Core's import, audit and preparation |
| 2 | Example "EN → FR change = 0.25" (7.5) | Impossible with the fake model as described: one `INVALID` and one failure remove 2 samples out of 8, so 2/6 | Fake model with faults: 1/3. Without faults (`faults: false`): 0.25, as in the example. Both are tested |
| 3 | Translations of `questions.yaml` | 35 concepts and one wording in 7 languages (EN FR ES DE AR HI ZH), written by an AI | **To be proofread by a doctor for each language** before any result is published |
| 4 | ID of the Med models | The 10 models of the paper, each pinned to a commit checked on Hugging Face. MedSigLIP and MedGemma are gated: accept their terms first. BiomedCLIP goes through open_clip, which cannot pin a commit | To check with `vlmshift check` before the first real run |
| 5 | The 35 Med concepts | The paper gives their number and their RadLex basis, not their list | Rebuilt from the labels of the sources: the 14 findings of NIH, tuberculosis, COVID-19, the 4 devices of RANZCR, and the 15 most frequent other findings of PadChest-GR. 20 have a RadLex term (checked on radlex.org, RadLex 4.3), 8 combine two terms, 7 have none. To replace with the bank of the paper if it is published |
| 6 | Nigeria source | Its 2,600 file names and sizes are those of two Kaggle sets of other origins (COVID-19 Radiography Database, Tuberculosis (TB) Chest X-ray Database). The pixels were not compared | Kept, as in the paper, and said in its card. Its results may not describe a Nigerian domain |
| 7 | CheXzero weights | The paper does not say which file | `best_64_5e-05_original_22000_0.864.pt`, one of the 10 files of the CheXzero folder. The CheXzero paper averages the 10 |

## Interpretations of the design

| Topic | Choice | Why |
|---|---|---|
| Strict mode on a controlled axis | Pairs built on the rows outside `error`, not on E_ab | By definition, E_ab contains no `INVALID`: strict mode would have no effect |
| Δ_Inv on a controlled axis | Same pairs as strict mode | On E_ab, the `INVALID` rate would always be 0 |
| Several controlled axes | The pair is (sample, other controlled axes) | "The other axes being fixed": blurred EN against blurred FR, never against original FR |
| Base metrics in `metrics.parquet` | `level_b` empty, `m = 1`, `significant` empty | They describe a cell. They are not comparisons, Bonferroni does not apply to them |
| m (Bonferroni) | Pairs × cells, without counting the metrics or the two modes. The cells are those of the variants: a model that has not reached a cell yet already gets the m of the finished run | Follows the example in section 5.5 (6 × 11 = 66). An unfinished run is never corrected more lightly |
| `mode` column added to `metrics.parquet` | `valid` or `strict` | Each metric is computed twice. `yes_rate` exists only in `valid`, `inv_rate` only in `strict` |
| Detail yes → no and no → yes | `valid` mode only | An `INVALID` pair has no direction of change |
| `data` as a list | Files put end to end, same columns | The simplest case of "split into several files" |
| `id` column in the dataset | Read as the original ID | The example in section 3 has one, without declaring it in `roles` |
| Row `id` and image | The image fingerprint is part of the `id` computation | Replacing an image changes the table. The full path is not part of it: moving the dataset changes nothing |
| Audit of the cells | Each axis alone, then all axes crossed | An empty cell exists only in a crossing. The measures are not known yet at step 2 |
| `min_units` on a shift | The smallest number of units of the two levels | The figure is no more reliable than its smallest level |
| Fewer than 2 units | Always not estimable, even with `min_units: 1` | A single unit cannot be drawn at random |
| Probe on a column that varies for the same image | Ignored and flagged | Section 5.7: it makes no sense |
| Probe: split | `StratifiedGroupKFold` | Keeps units whole and balances the classes. Without it, a fold can have only one class |
| Levels checked on the variants, not on the results | | An interrupted run must give a report, even with no result at the reference level |

## Technical choices

| Topic | Choice | Why |
|---|---|---|
| Bootstrap | Counts per unit, weighted by draw | All metrics are ratios of counts. One formula for the value and the draws, and a very fast computation |
| Commands | Each command runs the previous ones again | Nothing to remember between two commands, and everything is idempotent. Cost: images are read again at each command |
| `analysis.yaml` and `run` | `vlmshift all` and `--run` replace the key | The run fingerprint is not known when the file is written |
| Paths | Relative to the YAML file; images relative to the data file | The toy dataset works from any folder |
| Model cache | JSONL, one line appended per variant | An append at each batch survives an interruption. Parquet does not allow appending |
| Rows in `error` | Left out of the cache, so run again at the next launch | A failure is not an answer |
| Units of each number | Each number counts the units it rests on: units with a positive case for a sensitivity, with both for a balanced accuracy, with a pair for an answer change. Below `min_units`: not estimable | A cell of 41 units with one positive image gave a balanced accuracy "on 41 units", significant, with an interval of zero width |
| Numeric axes | An observed numeric axis with more than 4 values is cut into quartiles in the audit, the coverage and the measures. Each quartile is named by its rank then its interval, such as `Q1 (4.999, 16.25]`, so that sorted as text the levels keep the order of the numbers. A controlled axis keeps its levels | The measures compared every raw age with every other. The levels of a controlled axis are chosen by the experimenter |
| Columns of the templates | Part of the question, not of the sample | The documented `{concept}`, translated in each language, split every sample in two: no pair was left. Like the question, these columns may change between the variants. All declared templates count, whatever the models, so adding a model never changes the samples |
| No row left for the models | `prepare` stops and names the template columns to check | With every value empty, the run used to crash |
| Configuration errors | One line per problem, with the key and the value written | A key that accepts two forms (answer lists or a file name) also printed the error of the form not used |
| Name of a new axis (`init --axis`) | Any name Python accepts: letters of any alphabet, digits and `_` | `langue_é` was refused, while a table accepts it |
| Cache place | The model cache belongs to the table, keyed on the texts its kind of model receives | Two experiments on one table, one generative and one contrastive, wrote different texts into the same variants folder. Now each set of templates has its own folder, and adding a CLIP model keeps the answers of the generative ones |
| Failures with `--only` | Saved in the cache (`errors.jsonl`) and rewritten at each launch of the model | Running a second model alone made the failures of the first look like rows not asked yet |
| A model that never answers | `run` and `all` stop with an error, after writing the run folder | All rows in `error` used to end with exit code 0, and a script went on to analyse nothing |
| Cut cache line | Closed before the next line is added | After a hard stop, the first new answer was glued to the cut line and lost |
| Clear messages | Broken YAML, a template brace without a column name, an axis name taken by the table, a batch size of 0, a column of the tool as a probe target: each stops with a sentence, not a traceback | Found by a test campaign on hand-made mistakes |
| Table read from a spreadsheet | Semicolons accepted; several files must have the same columns; a level written as a number is read as text | A French spreadsheet saves CSV with semicolons |
| `inference.retries` and the cache | Not part of the model fingerprint | More attempts cannot change an answer. Raising them after a failed run used to ask every question again |
| Failed batch | Run again row by row | One faulty image does not lose its whole batch |
| Frozen copy of `experiment.yaml` | The completed configuration, not the source file | The copy no longer depends on any other file (e.g. answer lists) |
| Med labels | Only what a source states. NIH: a finding listed is positive, "No Finding" is negative for the 14, a finding not listed next to another is unknown. RANZCR: a device with no column at 1 is absent. PadChest-GR: a label in a negative sentence is negative, a report with no positive sentence is negative for its concepts. Nigeria: an image of one class is unknown for the concepts of the others. RICORD: the class chosen by most readers, typical → positive, negative for pneumonia → negative, the rest unknown | Section 4.2: unknown is dropped, never turned into a negative |
| Med sample | 300 patients per source, with one image each (the first by path), half with a positive label when there are enough, seed 0. Unknown labels are dropped first | The paper takes 300 patients per source, balanced at patient level: read as half ill, half not |
| Med sources without patient numbers | Each image is its own patient: Pakistan TB, Bangladesh, Nigeria, Shenzhen, Montgomery, PTX-498, JSRT | They give none |
| Version of a raw folder | SHA-256 of the sorted lines `<folder>/<name> <SHA-256 of the file>` | The place where the archive is extracted does not matter, and a file moved to another class changes it. For Pakistan TB, computed from the SHA-256 that Mendeley publishes for each file |
| Med core of the paper | The complete-case rule is applied as written: a cell stays if every model has an answer in every language | In the paper, the core is 12 pairs because some models ran on fewer cells. Here every model runs on every cell, so the core is larger |
| Unknown adapter | Error at step 3, not at load time | `analyze` does not need the adapters: Med can be analyzed without importing Med |
| Registration of the Med adapters | `paper.py` and `check.py` import `adapters` before calling Core | Generic loading is a bonus. Core is not modified |
| Report | Tables and `report.md` | HTML is a bonus. `report.md` carries the sections required at step 6 |
| `fake.py` in `inference/adapters/` | Addition compared to the 6.2 folder tree | 7.5 wants normal adapters, usable from `tests/fixtures/toy/` |
| `common.py` | Addition compared to the 6.2 folder tree | Fingerprints, Parquet and issues are used by all steps |
| Hooking up VLMEvalKit | Direct call to the `supported_VLM` catalog, without going through its full pipeline | The text sent stays exactly the one from step 3. Its pipeline can rewrite the question depending on the dataset and asks for a base64 TSV |
| Settings sent to VLMEvalKit | According to what each model declares. Not declared: sent with a caveat. Refused, or API: not sent. Each case is noted (`model_note`) | Each model has its own settings. Out of 553 catalog entries, 13 to 23 local models did not accept ours and could not load. The log must not claim a setting that was not applied |
| VLMEvalKit batches | 1 image | The library has no batches. Each answer goes into the cache right away |
| Image and text sent to VLMEvalKit | Explicit types (`image`, `text`) | Otherwise, an invalid path is read as text: the model answers without an image and without an error |
| Several environments | `run --only`, then a trace per model in `sessions.jsonl` and `run_log.json` | VLMEvalKit recommends different `transformers` versions for different families. Each launch fills the same run |
| Partial run | Analysis possible, with a "Partial run" banner and the detail of what is missing | An incomplete report must not look like a complete one |
| Project site | A home page and five pages in `docs/`, generated by `tools/site/build.py` from the imports of the code and from the Markdown files. A test fails if they are out of date | A hand-drawn map goes stale without anyone noticing. Plain files publish with GitHub Pages or open from the disk |
| Visual identity | One sage green and its transparent tints on a white page, Newsreader for titles and Inter for text. The logo is a circle cut in two: one half has moved, and faded | Calm and easy to read, with one colour to learn. The moved half is the shift the project measures. One file, `tools/site/art.py`, draws the logo, banner and diagram |
| Words in Core | "Image", "subject", "unit", with no field-specific term | The tool works on any images. Field-specific words stay in `packages/med/` |
| `vlmshift init` | Makes the table and two YAML files from a folder. Subfolders give the labels, each image starts as its own subject, and an existing file is never replaced | Writing a table by hand was the first obstacle. A file already adapted must never be lost |
| Issues in the terminal | Each step prints one line per kind of issue | A missing image removed a row without a word on screen |
| Report summary | The main significant shifts in plain sentences, with a warning under 30 units. The full table stays in `metrics.csv` | The summary must be readable without knowing the metric names |
| `vlmshift check` | Asks each model up to 4 real questions, then prints the card, the answers, how they are read and an estimated duration. Keeps no answer | On a server, a problem otherwise appears after a long wait. Keeping nothing means the check cannot change a result |
| Progress of a run | At most one line per minute for each model, with the errors so far and the time left | A run can last hours. A broken model must be seen early |
| `transformers` version | `5.22.2`, the version pip chooses with the pinned VLMEvalKit commit | The 43 imports VLMEvalKit makes at load time all exist in this version. Older families can use their own environment |
| Coverage by level | The coverage table also gives each level of each axis. The summary names the levels where fewer than half of the answers can be used | In the first real run, SmolVLM-256M repeated every French question instead of answering. The total, 50%, hid that French had none |
| Summary when nothing is tested | "No shift could be tested", never "no significant shift" | With every number not estimable, "no significant shift" suggested that something had been measured |
| OpenCV on a server | `opencv-python` replaced by `opencv-python-headless` after the install | VLMEvalKit loads OpenCV, which looks for a screen library (`libGL.so.1`) that servers rarely have. The headless version needs no administrator |
| CSV next to the audit and the variants | `counts`, `samples` and `variants` are also written in CSV | They are meant to be read before any model runs, without Python |
| Cache keyed by question | One folder per model and settings in `workspace/cache/`. Each output is keyed by the image and the exact text | The cache belonged to one table and one set of templates: a corrected label, a new axis or an added CLIP model asked every question again. Two models without `id` shared one cache |
| Audit kept with the run | The audit is copied into `runs/<run>/audit/`, and the report reads it there | Two experiments on one table with other axes wrote over each other's audit: a report could show the other one |
| 16-bit images | The contrastive adapter scales them to 8 bits, and the import warns (`image_16bit`) | X-rays are often 16-bit. Converted directly, almost every pixel became white |
| CSV from a spreadsheet | UTF-8 or Windows-1252, commas or semicolons, read from the first line. A copy open in Excel is skipped with a warning (`csv_locked`) | French Excel saves with `;` and Windows-1252. An open CSV stopped the whole step after the models had run |
| Two questions at the same levels | An error (`same_levels`) when there are controlled axes, and the template columns then stay in the sample ID | The pairs of such rows would be chosen at random. Without controlled axes there is nothing to pair |
| Levels of an axis no measure uses | An error in `analysis.yaml` | `axes.site.levels: [A]` reads as "only site A", but it only restricts the measures on site. Ignoring it would give numbers on every site |
| Demo on POPE mini | 100 photos of POPE (COCO val2014) and their 1,800 questions, fetched by `examples/pope_mini/fetch.py` and asked to CLIP. The toy dataset moved to `tests/fixtures/toy/` | Circles and fake models taught nothing about real models. POPE is public, small, runs without GPU, and its paper publishes results to compare with (table 3) |
| POPE `mode` is observed, the photo is the unit | `mode: {type: observed}`, `unit: image_id` | The absent object differs from one mode to the next: no question can be paired with another. The 18 questions about a photo are not independent |
| The 100 photos | The first 100 by name among those asked about in all three modes | The same choice on every machine, with no random draw to record |
| CLIP's two sentences | "a photo of a bird." against "a photo." | Tried on POPE mini before writing the example, among five pairs. Sentences with a negation ("a photo without a bird.") did worse: CLIP reads negation poorly. Chosen on these same photos, so the numbers of CLIP illustrate the tool: they are not a benchmark result |
| Adapter `huggingface` | Generative models that transformers runs by itself, one image at a time, without sampling. The question goes inside the model's chat template if it has one, as it is otherwise. `params.quantize` shrinks the weights to 8 or 4 bits | VLMEvalKit takes a long installation (219 packages, 31 minutes), and some of its models more by hand: its InstructBLIP needs LAVIS and the path of Vicuna-13B written in a file of the library, its MiniGPT-4 needs its own repository and Vicuna-13B v0, rebuilt from LLaMA. On Colab and Kaggle, torch and transformers are already there |
| `paper.yaml` | InstructBLIP-13B only, `Salesforce/instructblip-vicuna-13b` at a pinned commit, in 4 bits. The Colab notebook runs it in full on a card of 32 GB or more | The weights of the paper, packaged for transformers. MiniGPT-4 v1 has no such package. In float16 the model needs about 28 GB, and a free T4 has 15 |
| Colab notebook | `examples/pope_mini/colab.ipynb`, written by `tools/make_colab.py`. A test fails if they differ | A notebook edited by hand drifts from the project. The cells that need a card are skipped without one |
| Versions of Colab | Tested with Python 3.13, pandas 2.2.3, numpy 2.4.6, scikit-learn 1.6.1, scipy 1.16.3, torch 2.14.1 and transformers 5.22.1, read from [googlecolab/backend-info](https://github.com/googlecolab/backend-info) | Colab pins older versions, pandas 2.2 above all, that no other test environment had |
| Site on GitHub Pages | `.github/workflows/pages.yml` rebuilds `docs/` from the sources and publishes it at each push to `main` | The published site always matches the code of the same commit |

## Known limits

| Limit | Effect | Lead |
|---|---|---|
| VLMEvalKit | Installed for real on a fresh Linux machine (Debian, Python 3.11, no card) by following the [server guide](server.md): 31 minutes, 219 packages, 7.2 GB. It needed `git`, `rouge-score` (imported but not listed by VLMEvalKit) and OpenCV for servers. SmolVLM-256M then answered 8 questions on the processor, after changing `device_map="cuda"` to `"cpu"` in VLMEvalKit for that test only. Never run on an NVIDIA card | Run `vlmshift check` on a server |
| Importing VLMEvalKit | It turns off certificate checks for Python's built-in HTTPS client in that process, and logs an error about a missing `.env` file. Hugging Face downloads use another client | Keep it in mind if your own code runs in the same process |
| Graphics card | Many local VLMEvalKit models require an NVIDIA card in their code (92 files out of 186, including SmolVLM and Qwen) | Plan for a machine with a GPU, or an API model |
| Settings sent without guarantee | A model can receive them without declaring them, then ignore them. The `model_note` issue flags it, without being able to settle it | Check the model class before a real run |
| Small LLM | Tested on tiny random models created on the spot, never on real weights | Same |
| Adapter `huggingface` | Tested on a tiny random LLaVA and a tiny random InstructBLIP created on the spot, and SmolVLM-256M answered real POPE questions on a processor. Never on a card, never with `quantize` | Run the Colab notebook on a T4 |
| Folds of the probe | The same seed gives other folds with scikit-learn 1.6 (Colab) and 1.8, so a probe score on very few units can change between the two | Compare probe scores within one environment. `run_log.json` keeps the versions |
| Contrastive models | Run for real with `openai/clip-vit-base-patch32` (pinned commit) on a processor. Scores and vectors match the standard forward pass of the model within 0.0000002. Run from start to finish on the 100 photos of POPE mini. `vlmshift check` with CLIP also works with the versions a server would install (Python 3.11, pandas 3.1, torch 2.15) | Run `examples/pope_mini/paper.yaml` on an NVIDIA card, to compare with published numbers |
| `revision` with VLMEvalKit | The weights of this version are downloaded, then given to the model through `model_path`. Without `model_path` (API): error. Without `revision`: the weights can change between two runs. Never tested on real weights | Always write `revision` for a result to be published |
| CheXzero and BiomedCLIP | Read through open_clip. CheXzero is tested on a random state dict of the same shape as its weights, BiomedCLIP not at all. Neither has run on its real weights | Run `vlmshift check` with both |
| Med sources | Tested on fake raw folders shaped like the real ones. Only Montgomery has run on the real data | Run `python check.py` then `python -m prepare`, and compare the counts with table 1 of the paper |
| Med overlaps | RANZCR images come from NIH, under other names. NIH lists pixel-identical images, some under two patient numbers. Pakistan TB has 26 byte-identical pairs. Bangladesh version 3 has duplicates with opposite labels (removed in version 6) | Kept as in the paper. A cross-source check by image fingerprint would find them |
| Large-scale bootstrap | Memory = draws × units (2000 × 100,000 ≈ 1.6 GB per comparison) | Draw in batches if needed |
| Bootstrap with very low counts | Intervals sometimes reduced to a point, hence "significant". Visible on the toy dataset of the tests, where `min_units` is lowered | Keep `min_units: 30` on real data |
