"""Image conditions: the same questions on a changed image (blank, shuffled, blurred, swapped...).

`vlmshift conditions experiment.yaml --add blank shuffled blur` reads the table of an experiment
and writes a larger one next to the changed images: each row once per condition, with a new
column to declare as a controlled axis. The rows of one sample keep the same sample_id, so each
condition is compared with the original image on the same questions.

Every change is deterministic: the same table and the same seed give the same files. No lesion
or marker position is ever invented: a mask needs a file of boxes drawn by someone.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import DEEP_MODES, Issues, VLMShiftError, read_csv_text
from . import checks
from .ids import sample_ids
from .io import read_table

ORIGINAL = "real"
# Five severities per degradation, from light to strong.
DEGRADATIONS = {
    "blur": [0.5, 1.0, 1.8, 2.8, 4.0],          # radius of the Gaussian blur, in pixels
    "noise": [3, 7, 12, 20, 30],                # standard deviation of the Gaussian noise, out of 255
    "jpeg": [90, 70, 50, 30, 10],               # JPEG quality
    "brightness": [0.9, 0.8, 0.7, 0.6, 0.5],    # brightness kept
    "contrast": [0.9, 0.75, 0.6, 0.45, 0.3],    # contrast kept
    "downsample": [0.8, 0.6, 0.45, 0.3, 0.2],   # size kept before scaling back
}
SIMPLE = ("blank", "border_cropped", "contrast_standardized")
OTHER_IMAGE = ("shuffled", "same_label_swap")  # another image of the table: no file is written
BOX_COLUMNS = ["image", "box_type", "x1", "y1", "x2", "y2"]


def parse_conditions(words: list[str]) -> list[str]:
    """`blur` → its five severities, `blur:3` → one, `mask:marker` → marker_masked. Refuse the rest."""
    found = []
    for word in words:
        name, _, detail = word.partition(":")
        if name in DEGRADATIONS and (not detail or detail in "12345" and len(detail) == 1):
            found += [f"{name}_s{severity}" for severity in ([int(detail)] if detail else range(1, 6))]
        elif name == "mask" and detail.isidentifier():
            found.append(f"{detail}_masked")
        elif word in SIMPLE + OTHER_IMAGE:
            found.append(word)
        else:
            known = [*SIMPLE, *OTHER_IMAGE, *DEGRADATIONS, "blur:1 to blur:5", "mask:<box type>"]
            raise VLMShiftError(f"Unknown condition {word!r}. Known conditions: {', '.join(known)}")
    if repeated := sorted({name for name in found if found.count(name) > 1} | ({ORIGINAL} & set(found))):
        raise VLMShiftError(f"Condition given twice, or named like the original image: {repeated}")
    return found


def open_gray_or_color(path: str):
    """The image in 8 bits, gray or color as it is. A 16-bit image is scaled, not clipped."""
    from PIL import Image

    image = Image.open(path)
    if image.mode in DEEP_MODES:
        pixels = np.asarray(image, dtype=np.float64)
        low, high = pixels.min(), pixels.max()
        scaled = (pixels - low) / (high - low) * 255 if high > low else np.zeros_like(pixels)
        return Image.fromarray(scaled.round().astype(np.uint8))
    return image.convert("L" if image.mode in ("L", "1") else "RGB")


def stable_number(*parts) -> int:
    """A number that depends only on its parts: the same on every machine, at every run."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12], 16)


def changed(image, condition: str, seed: int, boxes: list[tuple] = ()):
    """One image under one condition."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

    if condition == "blank":  # no information at all: a mid-gray image of the same size
        return Image.new(image.mode, image.size, 127 if image.mode == "L" else (127, 127, 127))
    if condition == "border_cropped":  # the edges often carry the marks of a site
        dx, dy = max(1, image.width // 20), max(1, image.height // 20)
        return image.crop((dx, dy, image.width - dx, image.height - dy)).resize(image.size, Image.Resampling.BICUBIC)
    if condition == "contrast_standardized":
        return ImageOps.autocontrast(image)
    if condition.endswith("_masked"):  # filled with the median gray: the region no longer stands out
        fill = int(np.median(np.asarray(image.convert("L"))))
        masked = image.copy()
        for box in boxes:
            ImageDraw.Draw(masked).rectangle(box, fill=fill if image.mode == "L" else (fill, fill, fill))
        return masked
    kind, severity = condition.rsplit("_s", 1)
    level = DEGRADATIONS[kind][int(severity) - 1]
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "noise":
        pixels = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, level, pixels.shape)
        return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=level)
        return Image.open(io.BytesIO(buffer.getvalue())).convert(image.mode)
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(level)
    small = image.resize((max(1, int(image.width * level)), max(1, int(image.height * level))), Image.Resampling.BILINEAR)
    return small.resize(image.size, Image.Resampling.BILINEAR)


def shuffled(images: pd.DataFrame, within: list[str], seed: int) -> dict[str, str]:
    """For each image, another image of its group: the question no longer goes with what is shown.

    The images of a group are put in a random circle and each takes the next one, so none keeps
    its own. A group of one image has nothing to exchange. The circle depends on the content of
    the images, not on where they are stored.
    """
    rng = np.random.default_rng(seed)
    other = {}
    images = images.drop_duplicates("_path").sort_values("image_hash")
    for _, group in (images.groupby(within) if within else [(None, images)]):
        paths = group["_path"].tolist()
        if len(paths) > 1:
            order = rng.permutation(len(paths))
            other |= {paths[i]: paths[j] for i, j in zip(order, np.roll(order, -1))}
    return other


def swapped(table: pd.DataFrame, across: str, same: list[str], seed: int) -> dict[int, str]:
    """For each row, an image of another level of `across` (another site) with the same label
    and the same values of `same` (the concept). The answer should not change: what changes is
    where the image comes from. A row with no such image is left out."""
    other = {}
    match = [*same, "label"]
    pools = dict(iter(table.drop_duplicates(["_path", *match]).sort_values("image_hash").groupby(match)))
    for index, row in table.iterrows():
        pool = pools[tuple(row[match])]
        pool = pool[pool[across] != row[across]]
        if len(pool):
            other[index] = pool["_path"].iloc[stable_number(seed, row["image_hash"], *row[match]) % len(pool)]
    return other


def read_boxes(path: Path, table: pd.DataFrame) -> dict[tuple[str, str], list[tuple]]:
    """Boxes drawn by someone: image (as written in the table), box_type, x1, y1, x2, y2."""
    boxes = read_csv_text(path)
    checks.require_columns(boxes, BOX_COLUMNS, f"the boxes of {path.name}")
    paths = dict(zip(table["image"], table["_path"]))
    if unknown := sorted(set(boxes["image"]) - set(paths)):
        raise VLMShiftError(f"{path.name}: images that are not in the table, such as {unknown[0]}")
    found: dict[tuple[str, str], list[tuple]] = {}
    for row in boxes.itertuples():
        found.setdefault((paths[row.image], row.box_type), []).append(tuple(int(float(v)) for v in (row.x1, row.y1, row.x2, row.y2)))
    return found


def relative(path: Path | str, folder: Path) -> str:
    try:
        return Path(os.path.relpath(path, folder)).as_posix()
    except ValueError:  # another drive, on Windows
        return Path(path).resolve().as_posix()


def write_conditions(config, conditions: list[str], out: Path, axis: str = "image_condition", seed: int = 0,
                     within: list[str] = (), across: str | None = None, same: list[str] = (),
                     boxes: Path | None = None) -> tuple[pd.DataFrame, Issues]:
    """Write the changed images and the table that asks every question under every condition."""
    paths = [config.path(path) for path in (config.data if isinstance(config.data, list) else [config.data])]
    raw = read_table(paths)
    if axis in raw.columns:
        raise VLMShiftError(f"The table already has a column {axis}: choose another name with --axis")
    issues = Issues(step=1)
    table = checks.add_image_hash(checks.apply_roles(raw, config.roles), issues)
    checks.require_columns(table, [*within, *same, *([across] if across else [])], "the conditions")
    if "same_label_swap" in conditions and not across:
        raise VLMShiftError("same_label_swap needs --across, the column whose level must change (the site, for example)")
    missing = table["image_hash"].isna()
    for path in table.loc[missing, "_path"].unique():
        issues.warn("image_missing", f"Image not found, rows left out: {path}")
    table, raw = table[~missing], raw[~missing]
    if table.empty:
        raise VLMShiftError("No image of the table was found: nothing to change")

    # The identifier Core would give to each sample, written in the new table: the rows of the
    # changed images then stay in the sample of the original one.
    controlled = [name for name, spec in config.axes.items() if spec.type == "controlled"]
    computed = pd.Series(sample_ids(table, controlled, config.task.columns()), index=table.index, dtype="string")
    samples = table["sample_id"].fillna(computed) if "sample_id" in table else computed
    id_column = config.roles.sample_id or "sample_id"  # the column Core reads the sample from

    drawn = read_boxes(boxes, table) if boxes else {}
    exchange = shuffled(table, list(within), seed) if "shuffled" in conditions else {}
    swaps = swapped(table, across, list(same), seed) if "same_label_swap" in conditions else {}
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    image_column = config.roles.image

    def image_under(index, path: str, digest: str, condition: str) -> str | None:
        """The path of the image of a row under a condition, written if needed. None: left out."""
        if condition == "shuffled":
            return exchange.get(path)
        if condition == "same_label_swap":
            return swaps.get(index)
        marks = drawn.get((path, condition.removesuffix("_masked")), []) if condition.endswith("_masked") else []
        if condition.endswith("_masked") and not marks:
            return None
        target = images / f"{digest[:16]}_{condition}.png"
        if not target.exists():  # the same image under the same condition is written once
            changed(open_gray_or_color(path), condition, stable_number(seed, digest, condition) % 2 ** 32, marks).save(target)
        return str(target)

    parts = [raw.assign(**{image_column: [relative(path, out) for path in table["_path"]], axis: ORIGINAL,
                           f"{axis}_source": [relative(path, out) for path in table["_path"]], id_column: samples})]
    for condition in conditions:
        found = [image_under(index, path, digest, condition)
                 for index, path, digest in zip(table.index, table["_path"], table["image_hash"])]
        kept = [path is not None for path in found]
        if left := kept.count(False):
            issues.warn("condition_left_out", f"{condition}: {left} rows have no image under this condition and are left out")
        shown = [relative(path, out) for path in found if path is not None]
        source = shown if condition in OTHER_IMAGE else [relative(path, out) for path in table.loc[kept, "_path"]]
        parts.append(raw[kept].assign(**{image_column: shown, axis: condition, f"{axis}_source": source, id_column: samples[kept]}))
    result = pd.concat(parts, ignore_index=True).drop(columns="_root")
    result.to_csv(out / "table.csv", index=False, encoding="utf-8")
    return result, issues


def conditions_command(config, args) -> None:
    """The `vlmshift conditions` command: write the images and the table, then say how to use them."""
    out = Path(args.out).resolve() if args.out else config.path("conditions")
    table, issues = write_conditions(
        config, parse_conditions(args.add), out, args.axis, args.seed,
        args.within or [], args.across, args.same or [], Path(args.boxes) if args.boxes else None,
    )
    for code, group in issues.frame().groupby("code", sort=False):
        print(f"  ! {code} ({len(group)}): {group['message'].iloc[0]}")
    counts = table[args.axis].value_counts(sort=False)
    print(f"Table: {len(table)} rows ({', '.join(f'{count} {name}' for name, count in counts.items())}) -> {out / 'table.csv'}")
    print(f"Next: in experiment.yaml, set data to that table and add the axis {args.axis}: {{type: controlled}}. "
          f"In analysis.yaml, give it the reference {ORIGINAL}.")
