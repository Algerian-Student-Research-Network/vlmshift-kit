"""Two audits of the paper that are about chest X-ray data, not about any model in general.

Usage, from packages/med:
    python audits.py congruence <analysis folder>     does a model do better in the language of the country?
    python audits.py semantic <workbook.csv>          are the translated columns of the workbook equivalent?

- congruence: for the countries whose language is one of the seven and is not English, the shift
  from English to that language on the images of that country. It reads the measure
  language_by_country of the main arm: nothing is computed again. A country is where the images
  were acquired: it describes a source, it does not explain a shift.
- semantic: the annotated workbook has one text column per language. For the same concept in the
  same source, a language with more distinct texts than English carries details of each report,
  not only the concept. That is why the questions of the paper use the short terms of the bank,
  never these columns.
"""
import sys
from pathlib import Path

import pandas as pd
from bank import LANGUAGES

from vlmshift.common import VLMShiftError

# The language of each country of the paper, when it is one of the seven. English-speaking
# countries have no contrast with English, and Hindi is not counted as the language of Pakistan.
LOCAL = {"Spain": "ES", "China": "ZH"}
SHOWN = ["delta_acc", "delta_ba", "chg"]


def congruence(metrics: pd.DataFrame) -> pd.DataFrame:
    """The shifts from English to the local language, on the images of each country that has one."""
    by_country = metrics[(metrics["measure"] == "language_by_country") & (metrics["mode"] == "valid")
                         & metrics["metric"].isin(SHOWN)]
    if by_country.empty:
        raise VLMShiftError("No measure language_by_country in this analysis: it is the one of the main arm")
    parts = []
    for country, language in LOCAL.items():
        local = by_country[(by_country["split"] == f"country={country}") & (by_country["level_a"] == "EN")
                           & (by_country["level_b"] == language)]
        parts.append(local.assign(country=country, local_language=language))
    columns = ["model", "country", "local_language", "metric", "value", "ci_low", "ci_high", "significant", "n_units", "n_samples"]
    return pd.concat(parts, ignore_index=True)[columns].sort_values(["country", "model", "metric"]).reset_index(drop=True)


def semantic_audit(workbook: pd.DataFrame) -> pd.DataFrame:
    """For each source and concept: the number of distinct texts in each language column, their
    mean length, and the ratio to English. A ratio far above 1 marks report details in the translation."""
    rows = []
    listed = workbook.assign(concept=workbook["radlex_libelle"].fillna("").str.split(";")).explode("concept")
    listed["concept"] = listed["concept"].str.strip()
    for (site, concept), part in listed[listed["concept"] != ""].groupby(["dataset", "concept"]):
        row = {"site": site, "concept": concept, "n_rows": len(part)}
        for language in LANGUAGES:
            texts = part.get(f"text_{language}", pd.Series(dtype=str)).dropna().str.strip()
            texts = texts[texts != ""]
            row[f"unique_{language}"] = texts.nunique()
            row[f"mean_chars_{language}"] = texts.str.len().mean() if len(texts) else None
        for language in LANGUAGES[1:]:
            row[f"ratio_{language}_vs_EN"] = row[f"unique_{language}"] / row["unique_EN"] if row["unique_EN"] else None
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    try:
        if len(argv) != 2 or argv[0] not in ("congruence", "semantic"):
            raise VLMShiftError("Usage: python audits.py congruence <analysis folder>, or python audits.py semantic <workbook.csv>")
        source = Path(argv[1])
        if argv[0] == "congruence":
            found, target = congruence(pd.read_parquet(source / "metrics.parquet")), source / "report" / "congruence.csv"
        else:
            found, target = semantic_audit(pd.read_csv(source, dtype=str)), source.with_name("semantic_equivalence.csv")
        found.to_csv(target, index=False)
        print(f"{len(found)} rows -> {target}")
    except VLMShiftError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
