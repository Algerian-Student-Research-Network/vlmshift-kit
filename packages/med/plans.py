"""The experiments of the paper, rebuilt from its evaluation plans.

Usage, from packages/med:
    python plans.py                     write the table and the two YAML files of each arm
    python plans.py --arm main          one arm only
    python paper.py data/paper/arms/main/experiment.yaml data/paper/arms/main/analysis.yaml

A plan is the list of what the study sent to the models: one row per image, concept and
condition (eval_plan_core.csv, and eval_plan_core_supp.csv for the supplementary run). The plans
name patients and images of sources that cannot be shared again, so they are not in this
repository: put them in data/paper/. This script turns them into the tables of Core, one per arm
of configs/arms.yaml.

Each text is written again from the bank (bank.py) and compared with the text of the plan: a
single difference stops everything. So a table that is written holds exactly the questions of
the paper.

The plans give the path of each image on the machine of the study. data/paper/image_roots.json
says where these folders are here: {"/content/images/NIH": "D:/nih", ...}.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from bank import answer_sets, load_bank, load_romanized, pair, question, romanized

from vlmshift.common import VLMShiftError, write_parquet
from vlmshift.config.load import read_yaml, write_yaml

HERE = Path(__file__).parent
PLANS = ("eval_plan_core.csv", "eval_plan_core_supp.csv")
KEY = ["sample_id", "pathology"]
# What a row of a table holds whatever its arm, then what each arm adds.
COMMON = ["image", "patient", "site", "country", "scanner", "concept", "label", "sample",
          "language", "question", "positive_text", "negative_text"]
EXTRA = {
    "main": [], "formats": ["answer_format"], "negation": ["answer_format", "negated"],
    "controls": ["image_condition"], "quality": ["image_condition"],
    "factorial": ["answer_format", "frame_language", "term_language"],
    "permutation": ["term_condition", "shown_concept"], "transliteration": ["script"],
}


def read_plans(folder: Path) -> pd.DataFrame:
    """The plans found in a folder, end to end. The first one is needed, the supplementary one is not."""
    if not (folder / PLANS[0]).exists():
        raise VLMShiftError(
            f"{folder / PLANS[0]} not found. The evaluation plans of the paper are on the drive of the study "
            f"(manifests/): copy {' and '.join(PLANS)} to {folder}"
        )
    frames = [pd.read_csv(folder / name, dtype=str, keep_default_na=False).replace("", pd.NA)
              for name in PLANS if (folder / name).exists()]
    plan = pd.concat(frames, ignore_index=True)
    for column, default in (("script_variant", "native"), ("term_condition", "true")):  # only the supplementary plan has them
        plan[column] = plan[column].fillna(default) if column in plan else default
    plan["shown_concept"] = plan["shown_concept"].fillna(plan["pathology"]) if "shown_concept" in plan else plan["pathology"]
    return plan


def texts_of(row: dict, bank: dict, latin: dict) -> tuple[str, str, str]:
    """The question and the pair of a row of a plan, written again from the bank."""
    if row["script_variant"] == "romanized":
        return romanized(latin, row["shown_concept"], row["language"])
    present, absent = pair(bank, row["shown_concept"], row["frame_language"], row["prompt_variant"], row["term_language"])
    asked = question(bank, row["shown_concept"], row["frame_language"], row["prompt_variant"], row["answer_format"], row["term_language"])
    return asked, present, absent


def check_texts(plan: pd.DataFrame, bank: dict, latin: dict) -> None:
    """Every text of the plan must be the one the bank writes: the proof that the bank is that of the paper."""
    kinds = plan.drop_duplicates(["shown_concept", "frame_language", "term_language", "prompt_variant", "answer_format",
                                  "script_variant", "language"])
    for row in kinds.to_dict("records"):
        if texts_of(row, bank, latin) != (row["prompt"], row["positive_text"], row["negative_text"]):
            raise VLMShiftError(
                f"The plan and the bank disagree on {row['shown_concept']} in {row['language']} "
                f"({row['prompt_variant']}, {row['answer_format']}).\n  plan: {row['prompt']}\n  bank: {texts_of(row, bank, latin)[0]}"
            )


def local_images(paths: pd.Series, roots: dict) -> list[str]:
    """Each image path of the study, with its folder replaced by the one it has here."""
    prefixes = sorted(roots, key=len, reverse=True)  # the most precise folder first
    found = []
    for path in paths:
        prefix = next((prefix for prefix in prefixes if str(path).startswith(prefix)), None)
        found.append(str(Path(roots[prefix] + str(path)[len(prefix):])) if prefix else str(path))
    return found


def as_table(plan: pd.DataFrame, roots: dict) -> pd.DataFrame:
    """The rows of a plan with the columns of Core. The label of a negated question is turned
    back to that of the finding: Core counts a yes to a negated question as a denial."""
    negated = plan["negated"].eq("1")
    present = plan["y_true"].eq("1") != negated
    return pd.DataFrame({
        "image": local_images(plan["resolved_image_path"], roots),
        "patient": plan["dataset"] + ":" + plan["patient_id"],
        "site": plan["dataset"], "country": plan["country"], "scanner": plan["scanner_type"],
        "concept": plan["pathology"], "label": present.map({True: "positive", False: "negative"}),
        "sample": plan["sample_id"] + ":" + plan["pathology"],
        "language": plan["language"], "question": plan["prompt"],
        "positive_text": plan["positive_text"], "negative_text": plan["negative_text"],
        "answer_format": plan["answer_format"], "negated": negated.map({True: "1", False: "0"}),
        "image_condition": plan["image_condition"], "frame_language": plan["frame_language"],
        "term_language": plan["term_language"], "term_condition": plan["term_condition"],
        "shown_concept": plan["shown_concept"], "script": plan["script_variant"],
    })


def arm_rows(plan: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The rows of the plans that make an arm. An arm that compares with the main questions takes
    them from the main arm, for its own samples only."""
    experiment, part = plan["experiment"], plan["arm"]
    main = plan[(part == "main") & (experiment == "core")]
    formats = plan[(part == "factorial") & (experiment == "core")]

    def with_main(rows: pd.DataFrame, languages=None) -> pd.DataFrame:
        same = main.merge(rows[KEY].drop_duplicates(), on=KEY)
        return pd.concat([same if languages is None else same[same["language"].isin(languages)], rows], ignore_index=True)

    if arm == "main":
        return main
    if arm == "formats":
        return with_main(formats)
    if arm == "negation":  # the negation was asked by 1/0: it pairs with the plain question asked by 1/0
        return pd.concat([formats[formats["answer_format"] == "numeric_10"], plan[experiment == "negation"]], ignore_index=True)
    if arm == "controls":
        return with_main(plan[experiment.isin(["visual_control", "label_swap"])])
    if arm == "quality":
        return plan[experiment == "corruption"]
    if arm == "factorial":
        return plan[experiment == "term_frame_factorial"]
    if arm == "permutation":
        return with_main(plan[experiment == "term_permutation"])
    rows = plan[experiment == "transliteration"]
    return with_main(rows, set(rows["language"]))


def arm_table(plan: pd.DataFrame, arm: str, axes: dict, roots: dict) -> pd.DataFrame:
    """The table of an arm: its rows once each, with the columns it needs."""
    table = as_table(arm_rows(plan, arm), roots)
    controlled = [axis for axis, kind in axes.items() if kind == "controlled"]
    table = table.drop_duplicates(["sample", *controlled])[[*COMMON, *EXTRA[arm]]]
    if "language" not in axes:  # the sentence and the term have each their language
        table = table.drop(columns="language")
    return table.reset_index(drop=True)


def arm_configs(arm: str, spec: dict, base: dict, table: Path, bank: dict, folder: Path) -> tuple[dict, dict]:
    """experiment.yaml and analysis.yaml of an arm: its axes and analyses, the models of the benchmark."""
    task = {"type": "yes_no", **spec.get("task", {})}
    if "answers_by" in task:
        task["answer_sets"] = answer_sets(bank)
    else:
        task["answers"] = base["task"]["answers"]
    experiment = {
        "schema_version": 1, "data": str(table), "workspace": str(folder.parent.parent / "workspace"),
        "roles": {"unit": "patient", "sample_id": "sample", "positive_text": "positive_text", "negative_text": "negative_text"},
        "axes": {axis: {"type": kind} for axis, kind in spec["axes"].items()},
        "task": task, "models": base["models"], **({"inference": base["inference"]} if "inference" in base else {}),
    }
    analysis = {"schema_version": 1, "run": "runs/to-replace", "stats": {"min_units": 5}, **spec["analysis"],
                "report": {"notice": f"Arm {arm} of the paper. {spec['about']}"}}
    return experiment, analysis


def write_arms(plans: Path, out: Path, only: list[str] | None = None) -> dict[str, int]:
    """Write every arm the plans hold. Return the number of rows of each."""
    bank, latin = load_bank(), load_romanized()
    plan = read_plans(plans)
    check_texts(plan, bank, latin)
    roots = json.loads((plans / "image_roots.json").read_text(encoding="utf-8")) if (plans / "image_roots.json").exists() else {}
    specs, base = read_yaml(HERE / "configs" / "arms.yaml"), read_yaml(HERE / "configs" / "experiment.yaml")
    if unknown := sorted(set(only or []) - set(specs)):
        raise VLMShiftError(f"Unknown arm {unknown}. Arms: {list(specs)}")
    written = {}
    for arm, spec in specs.items():
        if only and arm not in only:
            continue
        table = arm_table(plan, arm, spec["axes"], roots)
        if table.empty:  # the supplementary plan is not there
            continue
        folder = out / "arms" / arm
        write_parquet(table, folder / "table.parquet", arm)
        experiment, analysis = arm_configs(arm, spec, base, folder / "table.parquet", bank, folder)
        write_yaml(experiment, folder / "experiment.yaml")
        write_yaml(analysis, folder / "analysis.yaml")
        written[arm] = len(table)
    return written


if __name__ == "__main__":
    arguments = sys.argv[1:]
    chosen = [arguments[index + 1] for index, word in enumerate(arguments) if word == "--arm"]
    folder = HERE / "data" / "paper"
    try:
        for name, rows in write_arms(folder, folder, chosen or None).items():
            print(f"{name}: {rows} rows -> {folder / 'arms' / name}")
    except VLMShiftError as error:
        sys.exit(f"Error: {error}")
