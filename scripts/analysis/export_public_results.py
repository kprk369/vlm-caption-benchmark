"""Export the final private run into a GitHub-safe, reproducible result bundle.

The exporter intentionally excludes images, Unsplash Dataset rows, source URLs,
image hashes, raw API responses, logs, secrets, pilots, and superseded runs.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "local-data" / "final-runs" / "revision-qwen38"
RESULTS = ROOT / "results"
CAPTIONS_IN = PRIVATE / "captions-120x4" / "benchmark.csv"
JUDGMENTS_IN = {
    "qwen38-pass-1.csv": PRIVATE / "judges" / "qwen38-pass-1" / "judgments.csv",
    "qwen38-pass-2.csv": PRIVATE / "judges" / "qwen38-pass-2" / "judgments.csv",
    "flash-next-pass-1.csv": PRIVATE / "judges" / "strata-combined-pass-1" / "judgments.csv",
    "flash-next-pass-2.csv": PRIVATE / "judges" / "strata-combined-pass-2" / "judgments.csv",
}
FINAL_MODELS = {
    "qwen3.8:27b-q4_K_M",
    "gemma4:12b-it-q8_0",
    "gemma3:4b-it-q8_0",
    "qwen3-vl:2b-instruct-q8_0",
}
PRIVATE_CAPTION_COLUMNS = {
    "image_sha256",
    "image_size_bytes",
    "image_width_px",
    "image_height_px",
}
PRIVATE_JUDGMENT_COLUMNS = {"image_sha256"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    caption_fields, captions = read_csv(CAPTIONS_IN)
    caption_keys = {
        f"{row['dataset_index']}|{row['image']}|{row['model']}" for row in captions
    }
    model_counts = Counter(row["model"] for row in captions)
    if len(captions) != 480 or len(caption_keys) != 480:
        raise RuntimeError("Expected 480 unique final captions.")
    if set(model_counts) != FINAL_MODELS or set(model_counts.values()) != {120}:
        raise RuntimeError(f"Unexpected final model counts: {dict(model_counts)}")
    if any(row["status"] != "ok" for row in captions):
        raise RuntimeError("Caption export contains failed rows.")

    public_caption_fields = [
        field for field in caption_fields if field not in PRIVATE_CAPTION_COLUMNS
    ]
    write_csv(RESULTS / "raw" / "captions.csv", public_caption_fields, captions)

    judge_metadata: list[dict[str, Any]] = []
    for public_name, source in JUDGMENTS_IN.items():
        fields, rows = read_csv(source)
        keys = {
            f"{row['dataset_index']}|{row['image']}|{row['caption_model']}"
            for row in rows
        }
        if len(rows) != 480 or len(keys) != 480 or keys != caption_keys:
            raise RuntimeError(f"Incomplete judgment pass: {public_name}")
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError(f"Judgment pass has failures: {public_name}")
        public_fields = [field for field in fields if field not in PRIVATE_JUDGMENT_COLUMNS]
        write_csv(RESULTS / "raw" / "judgments" / public_name, public_fields, rows)
        first = rows[0]
        judge_metadata.append({
            "file": f"raw/judgments/{public_name}",
            "pass_id": first["pass_id"],
            "judge_model": first["judge_model"],
            "judge_model_digest": first.get("judge_model_digest", ""),
            "seed": int(first["seed"]),
            "temperature": float(first["temperature"]),
            "judgments": len(rows),
        })

    first_caption = captions[0]
    model_metadata = []
    for model in sorted(FINAL_MODELS):
        row = next(item for item in captions if item["model"] == model)
        model_metadata.append({
            "model": model,
            "digest": row["model_digest"],
            "parameter_size": row["model_parameter_size"],
            "quantization": row["model_quantization_level"],
            "captions": model_counts[model],
        })
    metadata = {
        "schema_version": 1,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "images": 120,
            "categories": 6,
            "images_per_category": 20,
            "availability": "Images and Unsplash Dataset rows are not redistributed. See docs/DATA.md.",
        },
        "caption_generation": {
            "rows": len(captions),
            "prompt_sha256": first_caption["prompt_sha256"],
            "seed": int(first_caption["seed"]),
            "temperature": float(first_caption["temperature"]),
            "models": model_metadata,
        },
        "judging": judge_metadata,
        "hardware": {
            "gpu": first_caption["gpu_name"],
            "gpu_memory_mib": int(first_caption["gpu_total_memory_mib"]),
            "gpu_driver": first_caption["gpu_driver_version"],
            "ollama_version": first_caption["ollama_version"],
        },
        "publication_exclusions": sorted(PRIVATE_CAPTION_COLUMNS | PRIVATE_JUDGMENT_COLUMNS),
    }
    write_json(RESULTS / "experiment_metadata.json", metadata)

    workbook = ROOT / "outputs" / "qwen38-final-analysis" / "vlm_caption_benchmark_results.xlsx"
    if workbook.is_file():
        (RESULTS / "analysis").mkdir(parents=True, exist_ok=True)
        shutil.copy2(workbook, RESULTS / "analysis" / "vlm_caption_benchmark_results.xlsx")

    print(json.dumps({
        "captions": len(captions),
        "judgment_passes": len(JUDGMENTS_IN),
        "judgments": 480 * len(JUDGMENTS_IN),
        "models": dict(sorted(model_counts.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
