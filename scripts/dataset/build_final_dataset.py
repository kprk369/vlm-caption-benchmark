"""Build the private, reproducible 120-image benchmark from Unsplash Lite.

The final set contains 20 images in each of six metadata-assisted strata. The
existing 20-image pilot is preserved as items 1-20; 100 new photos are selected
with a fixed seed. Descriptions are used only for sampling, never as reference
captions or evaluation ground truth.

Unsplash Dataset Terms prohibit redistributing Dataset content. Keep the output
directory private and out of Git. Publish this script and the protocol, not the
downloaded images or manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover - environment check
    raise SystemExit("Pillow is required: python -m pip install Pillow") from exc


SEED = 20261012
TARGET_PER_CATEGORY = 20
CATEGORY_ORDER = (
    "objects_technology",
    "people_actions",
    "indoor_scene",
    "outdoor_scene",
    "animal",
    "complex_text_heavy",
)

# Deliberately narrow patterns reduce cross-category noise. These labels are
# sampling strata, not claims about everything visible in an image.
CATEGORY_PATTERNS = {
    "objects_technology": re.compile(
        r"\b(computer|laptop|keyboard|camera|smartphone|cell phone|phone|bicycle|bike|"
        r"headphones?|monitor|television|game console|robot|drone|microphone|"
        r"smartwatch|electronic device|circuit board|graphics card|gpu)\b",
        re.I,
    ),
    "people_actions": re.compile(
        r"(?:\b(man|woman|person|people|boy|girl|child|children|worker|athlete)\b.*"
        r"\b(running|walking|riding|holding|cooking|working|playing|jumping|climbing|"
        r"reading|writing|eating|drinking|shopping|dancing|driving|carrying|talking|"
        r"sitting|standing|swimming|skating|skiing)\b)|"
        r"(?:\b(running|walking|riding|holding|cooking|working|playing|jumping|climbing|"
        r"reading|writing|eating|drinking|shopping|dancing|driving|carrying|swimming)\b.*"
        r"\b(man|woman|person|people|boy|girl|child|children)\b)",
        re.I,
    ),
    "indoor_scene": re.compile(
        r"\b(interior|indoors?|living room|dining room|bedroom|bathroom|kitchen|"
        r"office|restaurant|cafe|library|classroom|workshop|warehouse|hallway|"
        r"museum|hotel room|studio apartment|inside a room)\b",
        re.I,
    ),
    "outdoor_scene": re.compile(
        r"\b(mountain|forest|beach|ocean|lake|river|waterfall|road|park|garden|"
        r"field|desert|valley|coast|cityscape|village|trail|bridge|skyline|canyon|"
        r"snowy landscape|countryside)\b",
        re.I,
    ),
    "animal": re.compile(
        r"\b(dog|cat|bird|horse|cow|sheep|goat|elephant|giraffe|zebra|bear|lion|"
        r"tiger|monkey|fox|deer|rabbit|fish|whale|dolphin|insect|butterfly|animal|"
        r"wildlife|penguin|duck|chicken|camel|squirrel)\b",
        re.I,
    ),
    "complex_text_heavy": re.compile(
        r"\b(street sign|road sign|traffic sign|signage|billboard|menu board|menu|"
        r"poster|newspaper|book page|storefront sign|shop sign|price tag|labels?|"
        r"graffiti|protest sign|neon sign|advertisement|notice board|departure board)\b",
        re.I,
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--photos",
        type=Path,
        default=Path("local-data/source/unsplash-lite/photos.tsv000"),
    )
    parser.add_argument("--pilot-dir", type=Path, default=Path("local-data/pilot/images"))
    parser.add_argument("--output", type=Path, default=Path("local-data/final-120"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--dataset-version", default="Unsplash Lite Dataset 1.4.1")
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="Optional UTF-8 file containing one rejected photo ID per line.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(text: str) -> list[str]:
    return [name for name, pattern in CATEGORY_PATTERNS.items() if pattern.search(text)]


def acceptable(row: dict[str, str], categories: list[str]) -> bool:
    try:
        width = int(row["photo_width"])
        height = int(row["photo_height"])
        aspect = float(row["photo_aspect_ratio"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        bool(categories)
        and bool(row.get("photo_image_url"))
        and min(width, height) >= 1200
        and 0.65 <= aspect <= 1.75
    )


def load_candidates(photos: Path, excluded_ids: set[str]) -> dict[str, list[dict[str, str]]]:
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    with photos.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["photo_id"] in excluded_ids:
                continue
            terms = normalize(row.get("ai_description"))
            categories = classify(terms)
            if not acceptable(row, categories):
                continue
            row["selection_terms"] = terms
            for category in categories:
                candidates[category].append(row.copy())
    return candidates


def choose_new_rows(
    candidates: dict[str, list[dict[str, str]]],
    pilot_counts: Counter[str],
    used_ids: set[str],
    used_photographers: set[str],
    seed: int,
) -> list[tuple[str, dict[str, str]]]:
    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, str]]] = []
    for category in CATEGORY_ORDER:
        need = TARGET_PER_CATEGORY - pilot_counts[category]
        if need < 0:
            raise RuntimeError(f"Pilot already exceeds the target for {category}.")
        pool = candidates[category][:]
        rng.shuffle(pool)
        chosen: list[dict[str, str]] = []
        for row in pool:
            username = normalize(row.get("photographer_username"))
            if row["photo_id"] in used_ids or not username or username in used_photographers:
                continue
            chosen.append(row)
            used_ids.add(row["photo_id"])
            used_photographers.add(username)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            raise RuntimeError(f"Only {len(chosen)} unique candidates for {category}; need {need}.")
        selected.extend((category, row) for row in chosen)
    return selected


def resized_url(base_url: str, width: int) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update({"fm": "jpg", "fit": "max", "w": str(width), "q": "85"})
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), "")
    )


def download(url: str, destination: Path, retries: int = 4) -> None:
    if destination.is_file() and destination.stat().st_size >= 10_000:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "VLM-caption-benchmark-independent-research/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if len(data) < 10_000:
                raise RuntimeError("Downloaded response was unexpectedly small.")
            temporary.write_bytes(data)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            if temporary.exists():
                temporary.unlink()
            if attempt == retries:
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            time.sleep(attempt * 2)


def pilot_rows(pilot_dir: Path) -> list[dict[str, str]]:
    manifest = pilot_dir / "pilot_manifest.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 pilot rows, found {len(rows)}.")
    return rows


def load_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise RuntimeError(f"Exclusions file not found: {path}")
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def copied_pilot_record(
    row: dict[str, str], index: int, source: Path, destination: Path
) -> dict[str, Any]:
    shutil.copy2(source, destination)
    return {
        "dataset_index": index,
        "filename": destination.name,
        "category": row["category"],
        "origin": "pilot",
        "photo_id": row["photo_id"],
        "photo_page_url": row["photo_page_url"],
        "source_image_url": row["source_image_url"],
        "downloaded_image_url": row["downloaded_image_url"],
        "photographer": row["photographer"],
        "photographer_username": row["photographer_username"],
        "dataset_name": row["dataset_name"],
        "dataset_url": row["dataset_url"],
        "source_width": row["source_width"],
        "source_height": row["source_height"],
        "selection_seed": row["selection_seed"],
        "sha256": sha256_file(destination),
        "photo_description_selection_only": row["photo_description_selection_only"],
        "ai_description_selection_only": row["ai_description_selection_only"],
        "review_status": "accepted_after_visual_review",
        "review_notes": "Pilot image retained; sampling stratum visually corrected where needed.",
    }


def new_record(
    row: dict[str, str], category: str, index: int, destination: Path,
    url: str, dataset_version: str, seed: int,
) -> dict[str, Any]:
    photographer = " ".join(
        value.strip() for value in (
            row.get("photographer_first_name", ""), row.get("photographer_last_name", "")
        ) if value.strip()
    )
    return {
        "dataset_index": index,
        "filename": destination.name,
        "category": category,
        "origin": "final_expansion",
        "photo_id": row["photo_id"],
        "photo_page_url": row["photo_url"],
        "source_image_url": row["photo_image_url"],
        "downloaded_image_url": url,
        "photographer": photographer,
        "photographer_username": row["photographer_username"],
        "dataset_name": dataset_version,
        "dataset_url": "https://unsplash.com/data",
        "source_width": row["photo_width"],
        "source_height": row["photo_height"],
        "selection_seed": seed,
        "sha256": sha256_file(destination),
        "photo_description_selection_only": row.get("photo_description", ""),
        "ai_description_selection_only": row.get("ai_description", ""),
        "review_status": "selected_after_metadata_screen",
        "review_notes": "",
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_contact_sheets(rows: list[dict[str, Any]], images_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_page = 5, 25
    thumb_w, thumb_h, label_h = 240, 170, 34
    for page_index, start in enumerate(range(0, len(rows), rows_per_page), start=1):
        page_rows = rows[start : start + rows_per_page]
        sheet = Image.new("RGB", (columns * thumb_w, 5 * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for position, row in enumerate(page_rows):
            x = (position % columns) * thumb_w
            y = (position // columns) * (thumb_h + label_h)
            with Image.open(images_dir / row["filename"]) as source:
                tile = ImageOps.contain(source.convert("RGB"), (thumb_w - 8, thumb_h - 8))
            tile_x = x + (thumb_w - tile.width) // 2
            tile_y = y + (thumb_h - tile.height) // 2
            sheet.paste(tile, (tile_x, tile_y))
            label = f"{int(row['dataset_index']):03d}  {row['category']}"
            draw.text((x + 6, y + thumb_h + 5), label, fill="black", font=ImageFont.load_default())
        sheet.save(output_dir / f"contact_sheet_{page_index:02d}.jpg", quality=90)


def validate(rows: list[dict[str, Any]], images_dir: Path) -> dict[str, Any]:
    categories = Counter(str(row["category"]) for row in rows)
    ids = [str(row["photo_id"]) for row in rows]
    filenames = [str(row["filename"]) for row in rows]
    hashes = [str(row["sha256"]) for row in rows]
    errors: list[str] = []
    if len(rows) != 120:
        errors.append(f"row_count={len(rows)}")
    for category in CATEGORY_ORDER:
        if categories[category] != TARGET_PER_CATEGORY:
            errors.append(f"{category}={categories[category]}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate_photo_id")
    if len(set(filenames)) != len(filenames):
        errors.append("duplicate_filename")
    if len(set(hashes)) != len(hashes):
        errors.append("duplicate_file_hash")
    for row in rows:
        path = images_dir / str(row["filename"])
        if not path.is_file() or path.stat().st_size < 10_000:
            errors.append(f"invalid_file:{path.name}")
            continue
        with Image.open(path) as image:
            image.verify()
    return {"valid": not errors, "errors": errors, "category_counts": dict(categories)}


def main() -> int:
    args = parse_args()
    if not args.photos.is_file():
        raise SystemExit(f"Missing Unsplash photos table: {args.photos}")
    if (args.output / "manifest.csv").exists():
        raise SystemExit(
            f"Output already contains a finished dataset: {args.output}. "
            "Choose a new --output directory to avoid accidental mixing."
        )
    pilot = pilot_rows(args.pilot_dir)
    pilot_counts = Counter(row["category"] for row in pilot)
    used_ids = {row["photo_id"] for row in pilot}
    rejected_ids = load_exclusions(args.exclusions)
    used_ids.update(rejected_ids)
    used_photographers = {normalize(row["photographer_username"]) for row in pilot}
    print("Loading and classifying Unsplash Lite metadata...", flush=True)
    candidates = load_candidates(args.photos, used_ids)
    if rejected_ids:
        print(f"Excluded {len(rejected_ids)} visually rejected photo IDs.", flush=True)
    for category in CATEGORY_ORDER:
        print(f"  {category}: {len(candidates[category])} eligible candidates", flush=True)
    selected = choose_new_rows(candidates, pilot_counts, used_ids, used_photographers, args.seed)
    print(f"Selected {len(selected)} new images with seed {args.seed}.", flush=True)
    if args.dry_run:
        return 0

    images_dir = args.output / "images"
    sheets_dir = args.output / "contact-sheets"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pilot, start=1):
        source = args.pilot_dir / row["filename"]
        destination = images_dir / f"image_{index:03d}.jpg"
        rows.append(copied_pilot_record(row, index, source, destination))
        print(f"[{index:03d}/120] copied pilot {row['photo_id']}", flush=True)

    for offset, (category, row) in enumerate(selected, start=21):
        destination = images_dir / f"image_{offset:03d}.jpg"
        url = resized_url(row["photo_image_url"], args.width)
        print(f"[{offset:03d}/120] downloading {category}: {row['photo_id']}", flush=True)
        download(url, destination)
        rows.append(new_record(row, category, offset, destination, url, args.dataset_version, args.seed))
        write_manifest(rows, args.output / "manifest.partial.csv")

    manifest = args.output / "manifest.csv"
    write_manifest(rows, manifest)
    partial = args.output / "manifest.partial.csv"
    if partial.exists():
        partial.unlink()
    make_contact_sheets(rows, images_dir, sheets_dir)
    report = validate(rows, images_dir)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["valid"]:
        return 1
    print(f"Created private final dataset at {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
