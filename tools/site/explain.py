"""The text shown when a box is clicked. build.py compares it to the real code."""
S, M = "src/vlmshift/", "packages/med/"
AD = S + "inference/adapters/"

# Above this number, the functions of a file are not listed.
LIMIT = 10

EXPLAIN = {
    S + "cli.py": dict(
        goal="The front door. When you type a command such as vlmshift all, this file starts. It computes nothing itself: it starts the steps in the right order and prints where the results are. Each command first redoes the steps before it, so there is nothing to remember between two commands.",
        fn={
            "show_issues": "Prints the issues of a step, one line per kind, so that none stays hidden in a file.",
            "do_init": "Makes the starting files from a folder of images.",
            "do_import": "Runs step 1.",
            "do_audit": "Runs step 1, then the audit. The audit command also prints the counts.",
            "do_prepare": "Runs the audit, then prepares the variants.",
            "do_check": "Prepares the variants, then tries each model on a few questions.",
            "do_run": "Prepares the variants, then asks the models. --only limits it to some models.",
            "do_analyze": "Computes the statistics, then writes the report.",
            "parser": "Describes the commands and their arguments.",
            "main": "Reads the typed command and calls the right function. Prints a clear message on error.",
        }),
    S + "data/starter.py": dict(
        goal="The way in for your own images. From a folder, it writes a table with one row per image and per condition, plus a starting experiment.yaml and analysis.yaml. Subfolders named positive and negative give the labels. It never replaces a file that already exists.",
        fn={
            "parse_questions": "Turns EN=text into one question per level, and refuses a wrong form.",
            "find_images": "Lists the images of the folder and its subfolders.",
            "label_of": "Reads the label from the first subfolder, otherwise unknown.",
            "check_axis": "Refuses an axis name that is not a plain word or that the table already uses.",
            "experiment_text": "Writes the starting experiment.yaml.",
            "analysis_text": "Writes the starting analysis.yaml, with the first level as reference.",
            "write_starter": "Writes the three files next to the folder of images.",
        }),
    S + "config/load.py": dict(
        goal="Turns the YAML files into checked objects that every step uses. It also computes their fingerprint: a code that changes only when a real value changes. The fingerprint is how finished work is found again.",
        fn={
            "_yaml": "Sets up the YAML 1.2 reader, so that yes and no stay text.",
            "read_yaml": "Reads a YAML file.",
            "write_yaml": "Writes a YAML file, such as the frozen copies.",
            "problems": "Writes one line per problem: the key, what is wrong, the value written.",
            "_build": "Applies the rules of schema.py and turns an error into a clear message.",
            "load_experiment": "Loads experiment.yaml. If the answers are in a separate file, it copies them in.",
            "load_analysis": "Loads analysis.yaml. The command line can replace the run it names.",
            "run_path": "Finds the folder of the run to analyse.",
            "config_hash": "Computes the fingerprint of a configuration, ignoring paths.",
        }),
    S + "config/schema.py": dict(
        goal="The rule book of the two configuration files. Each YAML block is described here: its keys, their type, their default value. An unknown key or a forbidden value is an error as soon as the file is read, before any computation. It also checks rules across keys, such as two models with the same name."),
    S + "extensions.py": dict(
        goal="The directory of models. In experiment.yaml you write a name such as fake_generative, and this file finds the matching Python class. A package such as Med can add its own models without touching the rest of the code.",
        fn={
            "register_adapter": "Adds a model to the directory, under a name.",
            "get_adapter": "Gives the class for a name. Error if the name is unknown.",
            "adapter_kind": "Says whether a model is generative or contrastive, with no error if the name is unknown.",
        }),
    S + "common.py": dict(
        goal="The toolbox shared by almost all the code: fingerprints, Parquet writing, reading and writing JSONL caches, and the notebook of issues (Issues). It also defines Step, what one step hands to the next, VLMShiftError, the error that stops the tool with a clear message, and to_levels, which cuts an observed numeric axis into quartiles for the audit, the measures and the report."),
    S + "__init__.py": dict(
        goal="Marks the folder as a Python package and gives its version number. It has no link with the other files."),

    S + "data/table.py": dict(
        goal="The conductor of step 1. It turns your starting file into a reliable table: reading, standard names, identifiers, checks, then writing. If a serious error exists, it first writes the list of issues, then stops.",
        fn={"build_table": "Runs the whole step and returns the table, its folder and its fingerprint."}),
    S + "data/io.py": dict(
        goal="Reads the input files, in CSV or Parquet. Everything becomes text, and an empty cell becomes a missing value. All the data is then handled the same way.",
        fn={
            "read_file": "Reads one file.",
            "read_table": "Puts several files end to end and notes the folder of each, to find the images.",
        }),
    S + "data/ids.py": dict(
        goal="Gives rows their identifiers. sample_id links the versions of one image: the EN and FR versions get the same code. id tells each row apart. These codes come from the content, so shuffling the file does not change them.",
        fn={
            "text_of": "Writes the content of a row as one text, leaving some columns out.",
            "digest": "Turns that text into a fingerprint.",
            "content": "Gives the fingerprint of each row, leaving some columns out.",
            "sample_ids": "Computes the fingerprint without the question or the controlled axes. If two variants of a sample would then be the same, it keeps the question columns.",
            "row_ids": "Computes the fingerprint on the content of the row, language included.",
        }),
    S + "data/checks.py": dict(
        goal="The checks at the door, with one small function per check: missing image, invalid label, duplicate, two subjects in one sample... Depending on how serious it is, it removes the row, warns or records an error. It also holds two preparations: giving columns their standard names and computing image fingerprints.",
        fn={
            "require_columns": "Stops with a clear message if a column is absent.",
            "apply_roles": "Gives the columns their standard names.",
            "add_image_hash": "Adds the full path and the fingerprint of each image.",
            "drop_duplicates": "Keeps one copy of rows that are strictly identical.",
            "check_collisions": "Refuses two different rows with the same id.",
            "drop_missing_images": "Removes the rows whose image is not found, and reports them.",
            "check_labels": "Refuses an empty or unknown label, instead of guessing.",
            "check_original_id": "Warns when an original identifier is empty or used twice.",
            "check_samples": "Reports a sample with several units or labels, or two variants at the same levels.",
            "check_axes": "Warns when an axis has one level only: there is nothing to compare.",
        }),

    S + "audit/counts.py": dict(
        goal="The conductor of step 2. It counts what the table holds: rows, samples, units and labels, for each level of each axis, then for all axes crossed. It flags what falls under 30 units. It never changes the table.",
        fn={
            "count_cells": "Counts each combination of levels, even an empty one.",
            "run_audit": "Runs the step and writes the audit.",
        }),
    S + "audit/samples.py": dict(
        goal="Takes stock of the samples: how many variants each one has, and which levels it lacks. A sample with no FR version, for example, cannot be compared face to face.",
        fn={"sample_table": "Gives one row per sample: its variants and its missing levels.",
            "combinations": "Lists the level combinations present, such as language=FR, quality=blur."}),

    S + "variants/plan.py": dict(
        goal="The conductor of step 3. It checks that each axis exists, has the texts sent to the models written, then stores the result in a folder named after its fingerprint. Changing the answer lists creates no new variants, because the texts sent stay the same.",
        fn={"prepare_variants": "Runs the step and returns the variants."}),
    S + "variants/prompts.py": dict(
        goal="Writes the exact text each model will receive. The template \"Is there {concept}?\" becomes, for example, \"Is there a circle?\". This text is stored, never recomputed: you can always check what the model read. A row missing a value is excluded and flagged.",
        fn={
            "templates_for": "Picks the templates needed by the models present.",
            "add_prompts": "Fills the templates, row by row.",
        }),
    S + "tasks/__init__.py": dict(
        goal="Finds the task type from its name. In V1 there is only one: yes_no.",
        fn={"get_task": "Gives the task for a name."}),
    S + "tasks/yes_no.py": dict(
        goal="Everything specific to yes/no questions. It turns a raw answer such as \"Yes.\" into a yes or no decision, by comparing it to the answer lists. It also says whether a decision is right against the label.",
        fn={
            "is_noise": "Spots punctuation and spaces, in every language.",
            "clean": "Cleans an answer: lower case, no punctuation at the edges.",
            "YesNoTask": "The yes/no task.",
            "YesNoTask.read": "Looks for the answer in the yes and no lists.",
            "YesNoTask.is_correct": "Compares the decision to the label. Does not judge an unknown label.",
        }),
    S + "tasks/base.py": dict(
        goal="Sets what a task must be able to do: ask the question, read the answer, say whether it is right. The steps talk to this interface, so adding a task type does not change them.",
        fn={
            "holes": "Lists the columns a template asks for.",
            "check_template": "Refuses a template with a broken brace or a brace without a column name.",
            "Task": "The base of every task.",
            "Task.build_prompt": "Fills the question template.",
            "Task.build_contrastive": "Fills the two sentences of contrastive models.",
            "Task.read": "Reads an answer. Each task writes its own.",
            "Task.is_correct": "Says whether a decision is right. Each task writes its own.",
        }),

    S + "inference/runner.py": dict(
        goal="The conductor of step 4. It sends the variants to the models in batches and keeps each output in a cache as soon as it arrives: a power cut loses nothing, and a restart skips what is done. The cache of a model is keyed by the question itself, image and text: another table, other labels or another axis ask nothing twice. On failure it retries, then isolates the faulty row. It then turns raw outputs into decisions and writes the results table. With --only it runs some models and still rebuilds the full table from their caches. During a long run it prints one line per minute."),
    S + "inference/check.py": dict(
        goal="The rehearsal before a long run. It asks each model up to four real questions, then prints the graphics card, the raw answers, how they are read and how long the full run would take. A problem shows in minutes instead of hours later. No answer is kept, so the results do not change.",
        fn={
            "graphics_cards": "Lists the NVIDIA cards that torch can use.",
            "show_machine": "Prints the cards and the versions of the main libraries.",
            "sample_rows": "Chooses a few rows that differ on the axes.",
            "ask": "Sends one row, then prints the question, the answer and how it is read.",
            "check_model": "Loads a model, asks the questions and estimates the full run.",
            "check_models": "Tries each model in turn, and ends with an error if one gave no answer.",
        }),
    S + "inference/run_log.py": dict(
        goal="Writes the run log: library versions, models, settings, fingerprints and dates. This log says exactly how a result was obtained. Steps 5 and 6 also use it to find the table and the variants.",
        fn={
            "now": "Gives the date and time.",
            "library_versions": "Records the versions of Python and the libraries.",
            "write_run_log": "Writes run_log.json.",
        }),
    S + "tasks/llm_reader.py": dict(
        goal="A small local language model that reads the answers the list does not recognise, such as \"Yes, there is one\". It answers yes, no or unclear, and its answers are cached. Since it could understand one language less well, its accuracy is measured language by language on a hand-annotated set.",
        fn={
            "load_llm": "Loads the model. Stops everything if it cannot load.",
            "LLMReader": "The reader and its cache.",
            "LLMReader.read": "Classifies an answer. The same answer is only asked once.",
            "validate_reader": "Measures the accuracy of the reader, language by language.",
        }),
    S + "inference/base.py": dict(
        goal="Sets what a model must be able to do, whatever it is. The engine only knows this interface: it calls generate or score without knowing which model is behind them. That is what lets you add a model without touching the engine.",
        fn={
            "ModelError": "The error for a technical failure.",
            "guard": "Turns any library failure into a ModelError.",
            "Model": "The base of every adapter.",
            "Model.info": "Gives the name and version, for the log.",
            "Model.load": "Loads the model.",
            "Model.unload": "Frees the memory.",
            "Model.generate": "Answers a question about an image.",
            "Model.score": "Scores a positive and a negative sentence.",
            "Model.embed": "Turns an image into a vector.",
        }),
    AD + "vlmevalkit.py": dict(
        goal="Connects generative models, the ones that answer with text, through the VLMEvalKit library. A name from its catalogue is enough: there is no code per model. The text sent stays exactly the one written at step 3. The library is only imported when needed. Tested with the real library on a processor, never on an NVIDIA card.",
        fn={
            "split_settings": "Keeps the settings this model can receive, and notes the others.",
            "pinned_weights": "Downloads the exact version asked for with revision.",
            "VLMEvalKitModel": "The adapter.",
            "VLMEvalKitModel.load": "Creates the model from the catalogue, with the settings it accepts.",
            "VLMEvalKitModel.ask": "Asks one question. A failure sentence becomes a real error.",
            "VLMEvalKitModel.generate": "Asks the questions one by one.",
        }),
    AD + "huggingface.py": dict(
        goal="Connects generative models that the transformers library runs by itself, such as InstructBLIP, LLaVA or Qwen. Lighter than VLMEvalKit: nothing more to install on Google Colab or Kaggle. A large model can be shrunk to 8 or 4 bits to fit a small card.",
        fn={
            "wrap": "Puts the question in the chat template of the model, if it has one.",
            "new_tokens": "Keeps only the words of the answer, without the question.",
            "HuggingFaceModel": "The adapter.",
            "HuggingFaceModel.load": "Loads the weights and the processor, on the card if there is one.",
            "HuggingFaceModel.ask": "Asks one question about one image.",
            "HuggingFaceModel.generate": "Asks the questions one by one.",
        }),
    AD + "contrastive.py": dict(
        goal="Connects contrastive models such as CLIP. They do not answer with text: they compare an image to two sentences and say which one is closest. The score is a cosine similarity. It can also give the vector of an image, which the probe uses.",
        fn={
            "open_rgb": "Opens an image in colour. A 16-bit image, common for X-rays, is first scaled to 8 bits.",
            "ContrastiveModel": "The adapter.",
            "ContrastiveModel.load": "Loads the weights and the processor from Hugging Face.",
            "ContrastiveModel._unit_vectors": "Scales vectors to a length of 1.",
            "ContrastiveModel._image_vectors": "Turns images into vectors.",
            "ContrastiveModel._text_vectors": "Turns sentences into vectors.",
            "ContrastiveModel.score": "Scores the two sentences for each image.",
            "ContrastiveModel.embed": "Returns the vectors of the images.",
        }),
    AD + "fake.py": dict(
        goal="Fake models whose answers follow a fixed rule, to test the tool with no GPU. The fake generative model answers correctly in English; in French it flips two answers, says \"maybe\" once and fails once. The fake contrastive model encodes the site in its vectors on purpose, so the probe can find it. The fake LLM ignores Hindi, also on purpose."),

    S + "stats/measures.py": dict(
        goal="The conductor of step 5. For each model and each measure, it cuts the data into cells, has the metrics and shifts computed, and gathers everything in metrics.parquet. It first checks the rules that need the run, such as an axis that was never declared.",
        fn={
            "with_unit_codes": "Numbers the units for the computations.",
            "make_rows": "Turns statistics into rows, with intervals and counts.",
            "support": "Counts the units that really count in a number: those with something to divide by.",
            "base_rows": "Computes the metrics of one cell.",
            "shift_rows": "Computes the shifts between two levels.",
            "split_cells": "Cuts along split_by, for example by site.",
            "measure_rows": "Handles one measure for one model.",
            "axis_levels": "Picks the levels and checks that they exist.",
            "check_rules": "Checks consistency with the run.",
            "run_analysis": "Runs the whole step.",
        }),
    S + "stats/pairing.py": dict(
        goal="Chooses the rows of each comparison. For a controlled axis, it keeps only the samples present at both levels: the same image in EN and FR, face to face. The samples left out are counted.",
        fn={
            "pairing_keys": "Says what identifies a pair. Nothing for an observed axis.",
            "select": "Keeps the usable rows and counts the excluded pairs.",
        }),
    S + "stats/shifts.py": dict(
        goal="Computes the shift metrics: what changes between two levels. The shift Δ is the metric at level b minus the metric at level a. The answer change is the share of samples whose decision flips, from yes to no or from no to yes.",
        fn={
            "delta_of": "Prepares the computation of a shift.",
            "change_sums": "Counts the answer changes, per unit.",
        }),
    S + "stats/probes.py": dict(
        goal="The latent space probe. It checks whether a column, such as the site, can be guessed from the vector of an image. A small classifier learns on some units and is tested on the others. Its score is compared to a noise floor obtained on randomly shuffled values.",
        fn={
            "predict_by_folds": "Trains and tests by folds. A unit stays in one fold.",
            "balanced_accuracy": "Measures how well the probe does, class by class.",
            "probe_scores": "Computes the score, the noise floor and the interval.",
            "run_probes": "Runs the probe for each model and each column.",
        }),
    S + "stats/metrics.py": dict(
        goal="Defines the base metrics: sensitivity (Sens), specificity (Spec), balanced accuracy (BA), yes rate and INVALID rate, and the answer-change rates. All are ratios of counts. These counts are kept per unit, which makes the bootstrap very fast. A division by zero gives \"not estimable\", never 0.",
        fn={
            "rate": "Divides two numbers, never inventing a 0.",
            "counter": "Prepares a counter of rows per unit.",
            "unit_sums": "Counts, per unit, the right answers and the useful rows.",
            "weigh": "Adds up these counts according to a draw.",
            "stat_of": "Prepares the computation of a metric or a change rate.",
        }),
    S + "stats/bootstrap.py": dict(
        goal="Measures the uncertainty of each number. The idea: draw the units at random, with replacement, 2000 times by default, and see how the number varies. It draws units and not rows, because two images of the same subject are not independent. A shift is significant if its corrected interval does not contain 0.",
        fn={
            "draw_weights": "Makes the draws: how many times each unit is drawn.",
            "summarize": "Computes the value, the intervals and the significance.",
        }),

    S + "report/export.py": dict(
        goal="The conductor of step 6. It reads back the outputs of the previous steps and produces the report: tables in CSV and Parquet, one matrix per measure, and the report.md summary. It redoes no statistics. It only adds the coverage, in total and by level, and the share of decisions made by the small LLM. The summary tells the main shifts in plain sentences, names the levels where most answers cannot be used, and says so when nothing could be tested. A value that cannot be computed shows as \"not estimable\", and a partial run says so at the top."),

    M + "run.sh": dict(
        goal="Runs the whole Med benchmark with one command: it checks that everything is ready, prepares the 11 sources, then runs paper.py, which runs the models through Core and writes the tables of the paper. It stops at the first error."),
    M + "check.py": dict(
        goal="Checks before starting that the raw data is there, that the libraries of each model and source are installed, and that the CheXzero weights are in place. It lists everything that is missing, instead of letting the benchmark fail later. For a missing source, it says where to ask for it and where to put it, from the card of the source. It also warns when a raw file or folder is not the version of the card, and gives the fingerprint to write in a card that has none.",
        fn={
            "digest": "Computes the SHA-256 of a file, block by block.",
            "fingerprint": "Fingerprints a raw file, or a raw folder from the names and contents of its files.",
            "where_to_get": "Says how to get a missing source, from its card.",
            "problems": "Returns the list of what is missing.",
            "other_versions": "Lists the raw files that are not the version of their card.",
        }),
    M + "cite.py": dict(
        goal="Turns the cards of the sources into two files: SOURCES.md, to find each source, and references.bib, to cite it. A test fails when they are out of date.",
        fn={"texts": "Writes the content of the two files from the cards."}),
    M + "prepare/__main__.py": dict(
        goal="The entry point of python -m prepare. It goes through each source it finds, prepares it and writes its table."),
    M + "prepare/__init__.py": dict(
        goal="Finds the source scripts of the folder by itself. Adding a source means adding a file, with nothing to change elsewhere. It also says what a card must hold, and checks it.",
        fn={
            "sources": "Imports all the source scripts of the folder.",
            "name_of": "Gives the name of a source: that of its script.",
            "folder_of": "Says where the files of a source go: its raw folder, or the folder of its raw file.",
            "card_problems": "Lists what is missing from the card of a source.",
        }),
    M + "prepare/<source>.py": dict(
        goal="One script per source of the paper: NIH ChestX-ray14, RANZCR CLiP, PadChest-GR, the Pakistan TB set, the Bangladesh pneumonia set, the Nigeria set, RSNA-RICORD-1c, Shenzhen, PTX-498, JSRT and Montgomery. Each one names its raw file or folder (RAW) and holds its card (SOURCE): where to ask for the data, its access, its licence, its version and its citation, since the data itself is never shared. It reads the raw layout of its source and gives one row per image and concept: image, patient, site, concept, label. A source without patient numbers uses one patient per image.",
        fn={
            "prepare": "Reads the raw files of the source and builds its rows.",
            "votes": "RICORD only: the classes the radiologists gave to each study, from the MD.ai export.",
        }),
    M + "label_rules/<source>.py": dict(
        goal="Each source codes its labels in its own way: a column of findings, one column per device, the folder of an image, the end of its name, the vote of several radiologists. Each rule turns that code into positive, negative or unknown, concept by concept. Only what the source states becomes a negative: a finding that is not mentioned stays unknown, and unknown is never asked.",
        fn={"labels": "Gives the label of each concept the source states, for one image."}),
    M + "prepare/template.py": dict(
        goal="The script to copy for a new source. It reads a raw file and turns it into the table every source gives, with its card. It is not a source: python -m prepare skips it.",
        fn={"prepare": "Reads the raw file and builds the table."}),
    M + "label_rules/template.py": dict(
        goal="The label rule to copy for a new source. Doubt and empty values become unknown, never negative.",
        fn={"to_label": "Translates a raw value into a label."}),
    M + "prepare/common.py": dict(
        goal="The tools shared by the sources. It finds the images of a source wherever its archive put them, extracts a zip once, and turns DICOM images into PNG. It then keeps the positives and negatives, draws 300 patients per source with one image each, half of them ill when there are enough, adds one row per language with the texts of questions.yaml, and writes the table in the format Core expects."),
    M + "paper.py": dict(
        goal="Runs the models through Core, then writes the tables of the paper from the run. A cell is a source and a concept. Only the cells where every model has an answer in every language are kept: the complete-case rule. An unreadable answer counts as wrong. The gap of each language with English comes with a 95% interval from 10,000 paired draws of model and cell pairs.",
        fn={
            "load_lines": "Reads the answers of a run with their source, concept, language and label.",
            "cell_scores": "Computes accuracy, F1, balanced accuracy and the INVALID rate of each cell.",
            "complete": "Keeps the cells where a metric exists for every model and language.",
            "models_table": "Table 3: one row per model, with the range between its best and worst language.",
            "gaps": "Gives the gap of each language with English and its interval.",
            "languages_table": "Table 4: one row per language, accuracy and balanced accuracy.",
            "rankings": "Compares the order of the models between two languages, in each source (Kendall τ).",
            "tables": "Gathers all the tables.",
            "write_tables": "Writes each table in CSV, and paper.md that shows them together.",
            "main": "Runs and analyses, or only rereads a finished run.",
        }),
    M + "adapters/__init__.py": dict(
        goal="Registers the models specific to Med in the directory of Core. Importing this file is enough for the names open_clip and chexzero to work in experiment.yaml."),
    M + "adapters/open_clip_model.py": dict(
        goal="The contrastive models published for the open_clip library, such as BiomedCLIP, that Hugging Face transformers cannot load. open_clip gives the model, its image preprocessing and its tokenizer. The scoring is that of Core: the cosine similarity of the image with each sentence.",
        fn={
            "OpenClipModel": "The adapter: it inherits the scoring of ContrastiveModel.",
            "OpenClipModel.build": "Gives the model, its preprocessing and its tokenizer.",
            "OpenClipModel.load": "Loads the model on the GPU if there is one. It refuses a revision: open_clip cannot choose one.",
            "OpenClipModel._image_vectors": "Turns images into unit vectors.",
            "OpenClipModel._text_vectors": "Turns sentences into unit vectors.",
        }),
    M + "adapters/chexzero.py": dict(
        goal="The CheXzero model, a CLIP ViT-B/32 trained on chest X-rays and their reports. Its official weights are files of the OpenAI clip package, which open_clip reads as they are. The images are normalized as CheXzero was trained and padded to a square instead of cut.",
        fn={
            "CheXzeroModel": "The adapter: everything else comes from OpenClipModel.",
            "CheXzeroModel.build": "Builds the ViT-B/32 with the weights and the preprocessing of CheXzero.",
        }),
}

RUNNER, PLAN, TABLE, COUNTS = S + "inference/runner.py", S + "variants/plan.py", S + "data/table.py", S + "audit/counts.py"
MEASURES, EXPORT, LOG, READER = S + "stats/measures.py", S + "report/export.py", S + "inference/run_log.py", S + "tasks/llm_reader.py"

# One entry per pink block: (band number, "reads" or "writes").
BLOCKS = {
    (0, "reads"): dict(
        goal="The two configuration files, written by you. experiment.yaml describes the experiment before inference. analysis.yaml describes the computations wanted after it.",
        items={
            "experiment.yaml": "Data, roles, axes, question, answer reader, models. Changing it means running the models again.",
            "analysis.yaml": "Run to analyse, reference levels, measures, statistics, probe. Rerun it as often as you like.",
            "answer lists": "The words accepted for yes and no, if they are in a separate file. They are copied into the configuration.",
            "folder of images": "Only with vlmshift init, to start a table.",
        },
        cols=[("READ BY", [(S + "config/load.py", "load_experiment, load_analysis"), (S + "data/starter.py", "find_images")])]),
    (0, "writes"): dict(
        goal="Only with vlmshift init: a starting point next to your images, for you to adapt. A file that already exists is never replaced.",
        items={
            "table.csv": "One row per image and per condition.",
            "experiment.yaml": "A first description of the experiment.",
            "analysis.yaml": "A first description of the analysis.",
        },
        cols=[("WRITTEN BY", [(S + "data/starter.py", "write_starter")]), ("READ BACK BY", [(S + "data/io.py", "table.csv"), (S + "config/load.py", "the two YAML files")])]),
    (1, "reads"): dict(
        goal="Your starting dataset. This is the only place where the tool reads your raw data.",
        items={
            "table.csv or .parquet": "Your table: one row per image and per condition. The data: key gives its path.",
            "images/*.png": "The images named in the table. They are read to compute their fingerprint.",
        },
        cols=[("READ BY", [(S + "data/io.py", "read_file, read_table"), (S + "data/checks.py", "add_image_hash")])]),
    (1, "writes"): dict(
        goal="The clean table and the list of what was wrong. The folder carries the fingerprint of the table: change one value and it is a new folder.",
        items={
            "table.parquet": "The checked table, with id, sample_id, unit and image_hash. The next steps get it in memory: this file is a record.",
            "issues.parquet": "The issues of step 1.",
        },
        cols=[("WRITTEN BY", [(TABLE, "build_table")]), ("READ BACK BY", [(EXPORT, "issues.parquet, load_inputs")])]),
    (2, "reads"): dict(
        goal="This step reads no file. It gets the table in memory, in a Step object holding the table, its folder and its fingerprint.",
        items={"Step of the table": "Returned by build_table, then passed on by cli.py."},
        cols=[("COMES FROM", [(TABLE, "build_table")]), ("RECEIVED BY", [(COUNTS, "run_audit")])]),
    (2, "writes"): dict(
        goal="The audit of the table, stored in the audit/ folder. Its issues.parquet is a separate file from the one of step 1.",
        items={
            "counts.parquet, .csv": "The counts by level and by crossing of axes. The CSV opens in a spreadsheet.",
            "samples.parquet, .csv": "One row per sample: its variants and its missing levels.",
            "issues.parquet": "What falls under the threshold of 30 units.",
        },
        cols=[("WRITTEN BY", [(COUNTS, "run_audit")]), ("READ BACK BY", [(EXPORT, "load_inputs")])]),
    (3, "reads"): dict(
        goal="As for the audit, the table arrives in memory in a Step object. No file is read.",
        items={"Step of the table": "Returned by build_table, then passed on by cli.py."},
        cols=[("COMES FROM", [(TABLE, "build_table")]), ("RECEIVED BY", [(PLAN, "prepare_variants")])]),
    (3, "writes"): dict(
        goal="What will be sent to the models. The folder carries the fingerprint of the table and the templates, so several models are tested on exactly the same variants.",
        items={
            "variants.parquet, .csv": "The table, plus the exact texts: prompt, positive_prompt and negative_prompt. The CSV opens in a spreadsheet.",
            "issues.parquet": "The rows excluded for lacking a value to fill a template.",
        },
        cols=[("WRITTEN BY", [(PLAN, "prepare_variants")]),
              ("READ BACK BY", [(MEASURES, "variants.parquet"), (EXPORT, "variants.parquet"), (RUNNER, "issues.parquet")])]),
    (4, "reads"): dict(
        goal="The variants arrive in memory. The step also reads back its own caches: whatever is already there is not computed again.",
        items={
            "Step of the variants": "What to send to the models.",
            "issues.parquet": "The issues of step 3, carried into those of the run.",
            "counts, samples, issues": "The audit of this experiment, copied into the run.",
            "outputs.jsonl": "The answers and scores already obtained, one folder per model and settings.",
            "vectors.jsonl": "The image vectors already computed.",
            "sessions.jsonl": "Where and when each model ran, to fill in the run log.",
            "errors.jsonl": "The failures of the last launch of each model: a model not run this time keeps them.",
            "<h>.jsonl": "The answers already read by the small LLM.",
            "annotated answers": "The hand-annotated set, to validate the small LLM.",
        },
        cols=[("COMES FROM", [(PLAN, "prepare_variants"), (COUNTS, "audit")]), ("READ BY", [(RUNNER, "caches, audit and issues.parquet"), (READER, "cache and annotated set")])]),
    (4, "writes"): dict(
        goal="Two different things. The caches keep the raw outputs and make resuming possible: one folder per model, its version and its settings, shared by every table and experiment of the workspace. The run folder holds the result of this experiment.",
        items={
            "outputs.jsonl": "Filled after each batch, and reused by every run.",
            "vectors.jsonl": "One vector per image, for models with embeddings: true.",
            "sessions.jsonl": "One line each time a model is loaded: date, library versions, notes.",
            "errors.jsonl": "The questions that failed. They are asked again next time; the failures of another experiment are kept.",
            "<h>.jsonl": "The cache of the small LLM.",
            "experiment.yaml (copy)": "The complete configuration, frozen.",
            "audit/ (copy)": "The audit of this experiment: another experiment on the same table, with other axes, does not replace it.",
            "results.parquet": "One row per variant and per model: answer, decision, status. Also in CSV, to read the raw answers.",
            "embeddings.parquet": "The vectors, for the probe.",
            "reader_validation.parquet": "The accuracy of the small LLM by language.",
            "issues.parquet": "The issues of steps 3 and 4.",
            "run_log.json": "Versions, dates, settings and fingerprints.",
        },
        cols=[("WRITTEN BY", [(RUNNER, "caches, results, frozen copies"), (LOG, "run_log.json"), (READER, "cache of the LLM")]),
              ("READ BACK BY", [(MEASURES, "run, results, vectors"), (EXPORT, "run, results, validation")])]),
    (5, "reads"): dict(
        goal="The analysis starts from the disk, not from memory. That is why it can be rerun without calling the models.",
        items={
            "experiment.yaml": "To know the axes and the models.",
            "run_log.json": "To find the table and the variants.",
            "table.parquet": "The whole table: the quartiles of an observed axis are its own.",
            "results.parquet": "The decisions of each model.",
            "embeddings.parquet": "The vectors, if a probe is requested and they exist.",
            "variants.parquet": "The labels, axes and units of each row.",
        },
        cols=[("WRITTEN BY", [(RUNNER, "run and results"), (LOG, "run_log.json"), (TABLE, "table.parquet"), (PLAN, "variants.parquet")]), ("READ BY", [(MEASURES, "run_analysis")])]),
    (5, "writes"): dict(
        goal="The result of one analysis. A run can have several analyses, each in its own folder.",
        items={
            "analysis.yaml (copy)": "The analysis configuration, frozen.",
            "metrics.parquet": "One row per number, with intervals and counts.",
            "probes.parquet": "The results of the probe.",
            "issues.parquet": "The issues of step 5, such as a model that always gives the same answer.",
        },
        cols=[("WRITTEN BY", [(MEASURES, "run_analysis")]), ("READ BACK BY", [(EXPORT, "load_inputs")])]),
    (6, "reads"): dict(
        goal="The report reads back the outputs of every step. It runs no model.",
        items={
            "runs/<r>/analyses/<a>/": "The numbers, the probe and their issues.",
            "runs/<r>/": "The configuration, the log, the results, the audit of the experiment and the validation of the LLM.",
            "tables/<t>/": "The table, the variants and the issues of step 1.",
        },
        cols=[("WRITTEN BY", [(TABLE, "issues.parquet"), (COUNTS, "audit"), (PLAN, "variants.parquet"), (RUNNER, "results"), (LOG, "run_log.json"), (MEASURES, "metrics and probe")]),
              ("READ BY", [(EXPORT, "load_inputs")])]),
    (6, "writes"): dict(
        goal="The final result, the part you read. Each table exists in CSV and in Parquet.",
        items={
            "report.md": "The readable report, in six sections, opening on the main shifts in plain sentences.",
            "metrics": "All the numbers, with a column already formatted.",
            "probes": "The results of the probe.",
            "coverage": "Per model: rows planned, valid, INVALID, failed and missing.",
            "audit_counts": "The audit.",
            "issues": "All the issues together.",
            "reader_share": "The share of decisions made by the small LLM.",
            "reader_validation": "Its accuracy by language.",
            "matrix_<measure>": "One measure as a cross table, cell by cell.",
        },
        cols=[("WRITTEN BY", [(EXPORT, "write_report")])]),
    (7, "reads"): dict(
        goal="The raw data of each source and the concept bank. The data is not in the repository: everyone gets it under its own licence.",
        items={
            "<source>/": "The raw files of a source, as it gives them: a label file, folders of images, or archives kept as they are.",
            "converted/": "The DICOM images of RICORD and JSRT, written once as PNG.",
            "questions.yaml": "The 35 concepts in 7 languages, with their RadLex identifier, and one wording per language.",
            "experiment.yaml, analysis.yaml": "The configurations of the benchmark: 11 tables, 10 models, English as reference.",
        },
        cols=[("READ BY", [(M + "prepare/<source>.py", "RAW"), (M + "prepare/common.py", "questions.yaml"), (M + "check.py", "experiment.yaml")])]),
    (7, "writes"): dict(
        goal="The table of each source, in the format Core expects: each one is an entry of the data: key of experiment.yaml. After the run, paper.py writes the tables of the paper. Apart, cite.py writes from the cards where to find each source and how to cite it.",
        items={
            "<source>.parquet": "300 patients, one image each, one row per concept and language.",
            "tables of the paper": "Per model, per language with its interval, source × language, and rankings, in CSV and in paper.md.",
            "SOURCES.md, references.bib": "Where to ask for each source, and its citation.",
        },
        cols=[("WRITTEN BY", [(M + "prepare/common.py", "write_source"), (M + "paper.py", "write_tables"), (M + "cite.py", "texts")]),
              ("READ BACK BY", [(S + "data/io.py", "read_file")])]),
}
