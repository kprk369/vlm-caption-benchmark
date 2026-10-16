"""Aggregate the four final judge passes and measured caption-generation telemetry."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
OUTPUT = RESULTS / "analysis"
CAPTIONS = RAW / "captions.csv"
PASSES = {
    "qwen_pass_1": RAW / "judgments" / "qwen38-pass-1.csv",
    "qwen_pass_2": RAW / "judgments" / "qwen38-pass-2.csv",
    "strata_pass_1": RAW / "judgments" / "flash-next-pass-1.csv",
    "strata_pass_2": RAW / "judgments" / "flash-next-pass-2.csv",
}
MODEL_LABELS = {
    "qwen3.8:27b-q4_K_M": "Qwen3.8 27B Q4_K_M",
    "gemma4:12b-it-q8_0": "Gemma 4 12B Q8_0",
    "gemma3:4b-it-q8_0": "Gemma 3 4B Q8_0",
    "qwen3-vl:2b-instruct-q8_0": "Qwen3-VL 2B Q8_0",
}
RATING_FIELDS = ("accuracy", "coverage", "clarity")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in order[index:end]:
            result[position] = average_rank
        index = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return numerator / denominator if denominator else None


def quadratic_weighted_kappa(left: list[int], right: list[int], low: int = 1, high: int = 5) -> float | None:
    if len(left) != len(right) or not left:
        return None
    size = high - low + 1
    observed = [[0.0] * size for _ in range(size)]
    left_hist = [0.0] * size
    right_hist = [0.0] * size
    for a, b in zip(left, right):
        observed[a - low][b - low] += 1
        left_hist[a - low] += 1
        right_hist[b - low] += 1
    n = len(left)
    weighted_observed = 0.0
    weighted_expected = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) / (size - 1)) ** 2
            weighted_observed += weight * observed[i][j] / n
            weighted_expected += weight * (left_hist[i] * right_hist[j] / (n * n))
    return 1 - weighted_observed / weighted_expected if weighted_expected else None


def cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = mean(float(a == b) for a, b in zip(left, right))
    p_left = mean(float(x) for x in left)
    p_right = mean(float(x) for x in right)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    return (observed - expected) / (1 - expected) if expected != 1 else None


def bootstrap_ci(values_by_image: dict[str, float], rng: random.Random, iterations: int = 5000) -> tuple[float, float]:
    keys = sorted(values_by_image)
    samples: list[float] = []
    for _ in range(iterations):
        samples.append(mean(values_by_image[rng.choice(keys)] for _ in keys))
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def paired_bootstrap_difference(
    left: dict[str, float], right: dict[str, float], rng: random.Random, iterations: int = 5000
) -> tuple[float, float, float]:
    keys = sorted(set(left) & set(right))
    differences = {key: left[key] - right[key] for key in keys}
    samples: list[float] = []
    for _ in range(iterations):
        samples.append(mean(differences[rng.choice(keys)] for _ in keys))
    samples.sort()
    return mean(differences.values()), percentile(samples, 0.025), percentile(samples, 0.975)


def aggregate() -> None:
    caption_rows = read_csv(CAPTIONS)
    if len(caption_rows) != 480:
        raise RuntimeError(f"Expected 480 captions, found {len(caption_rows)}")

    base: dict[str, dict[str, Any]] = {}
    for row in caption_rows:
        key = f"{row['dataset_index']}|{row['image']}|{row['model']}"
        if key in base:
            raise RuntimeError(f"Duplicate caption key: {key}")
        base[key] = {
            "source_key": key,
            "dataset_index": int(row["dataset_index"]),
            "image": row["image"],
            "category": row["category"],
            "caption_model": row["model"],
            "caption_model_label": MODEL_LABELS.get(row["model"], row["model"]),
            "caption": row["caption"],
            "candidate_format_valid": boolean(row["caption_format_valid"]),
        }

    pass_rows: dict[str, dict[str, dict[str, str]]] = {}
    integrity: dict[str, Any] = {"caption_rows": len(caption_rows), "passes": {}}
    for pass_name, path in PASSES.items():
        rows = read_csv(path)
        mapped: dict[str, dict[str, str]] = {}
        errors = 0
        invalid_confidence = 0
        for row in rows:
            key = f"{row['dataset_index']}|{row['image']}|{row['caption_model']}"
            if key in mapped:
                raise RuntimeError(f"Duplicate key in {pass_name}: {key}")
            if key not in base:
                raise RuntimeError(f"Unknown key in {pass_name}: {key}")
            if row["candidate_caption"] != base[key]["caption"]:
                raise RuntimeError(f"Caption text mismatch in {pass_name}: {key}")
            if row["status"] != "ok":
                errors += 1
            for field in RATING_FIELDS:
                score = int(row[field])
                if not 1 <= score <= 5:
                    raise RuntimeError(f"Invalid {field} in {pass_name}: {key}={score}")
            confidence = float(row["confidence"])
            if not 0 <= confidence <= 1:
                invalid_confidence += 1
            mapped[key] = row
        missing = sorted(set(base) - set(mapped))
        if len(rows) != 480 or missing or errors:
            raise RuntimeError(f"Incomplete pass {pass_name}: rows={len(rows)}, missing={len(missing)}, errors={errors}")
        pass_rows[pass_name] = mapped
        integrity["passes"][pass_name] = {
            "rows": len(rows), "errors": errors, "missing": len(missing),
            "invalid_confidence_values": invalid_confidence,
        }

    consensus_rows: list[dict[str, Any]] = []
    for key, item in base.items():
        judgments = [pass_rows[name][key] for name in PASSES]
        row = dict(item)
        for pass_name, judgment in zip(PASSES, judgments):
            for field in RATING_FIELDS:
                row[f"{pass_name}_{field}"] = int(judgment[field])
            row[f"{pass_name}_usable"] = boolean(judgment["usable"])
            confidence = float(judgment["confidence"])
            row[f"{pass_name}_confidence"] = confidence if 0 <= confidence <= 1 else None
        for field in RATING_FIELDS:
            values = [int(judgment[field]) for judgment in judgments]
            row[f"mean_{field}"] = round(mean(values), 4)
            row[f"range_{field}"] = max(values) - min(values)
        row["mean_quality"] = round(mean(row[f"mean_{field}"] for field in RATING_FIELDS), 4)
        usable_votes = sum(boolean(judgment["usable"]) for judgment in judgments)
        row["usable_votes"] = usable_votes
        row["majority_usable"] = usable_votes >= 3
        row["unanimous_usable"] = usable_votes == 4
        row["final_compliant_usable"] = row["majority_usable"] and row["candidate_format_valid"]
        valid_confidence = [float(j["confidence"]) for j in judgments if 0 <= float(j["confidence"]) <= 1]
        row["mean_confidence"] = round(mean(valid_confidence), 4) if valid_confidence else None
        row["invalid_confidence_count"] = 4 - len(valid_confidence)
        consensus_rows.append(row)
    consensus_rows.sort(key=lambda row: (row["dataset_index"], row["caption_model"]))

    rng = random.Random(20261013)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        by_model[row["caption_model"]].append(row)
    ranking_rows: list[dict[str, Any]] = []
    for model, rows in by_model.items():
        by_image = {row["image"]: row["mean_quality"] for row in rows}
        ci_low, ci_high = bootstrap_ci(by_image, rng)
        ranking_rows.append({
            "caption_model": model,
            "caption_model_label": MODEL_LABELS.get(model, model),
            "captions": len(rows),
            "mean_accuracy": round(mean(row["mean_accuracy"] for row in rows), 4),
            "mean_coverage": round(mean(row["mean_coverage"] for row in rows), 4),
            "mean_clarity": round(mean(row["mean_clarity"] for row in rows), 4),
            "mean_quality": round(mean(row["mean_quality"] for row in rows), 4),
            "quality_ci95_low": round(ci_low, 4),
            "quality_ci95_high": round(ci_high, 4),
            "raw_usable_vote_rate": round(mean(row["usable_votes"] / 4 for row in rows), 4),
            "majority_usable_rate": round(mean(float(row["majority_usable"]) for row in rows), 4),
            "format_valid_rate": round(mean(float(row["candidate_format_valid"]) for row in rows), 4),
            "final_compliant_usable_rate": round(mean(float(row["final_compliant_usable"]) for row in rows), 4),
            "mean_judge_confidence": round(mean(row["mean_confidence"] for row in rows if row["mean_confidence"] is not None), 4),
            "invalid_confidence_values": sum(row["invalid_confidence_count"] for row in rows),
        })
    ranking_rows.sort(key=lambda row: row["mean_quality"], reverse=True)
    for index, row in enumerate(ranking_rows, start=1):
        row["quality_rank"] = index

    pairwise_rows: list[dict[str, Any]] = []
    ranked_models = [row["caption_model"] for row in ranking_rows]
    for left_index, left_model in enumerate(ranked_models):
        for right_model in ranked_models[left_index + 1:]:
            left = {row["image"]: row["mean_quality"] for row in by_model[left_model]}
            right = {row["image"]: row["mean_quality"] for row in by_model[right_model]}
            difference, ci_low, ci_high = paired_bootstrap_difference(left, right, rng)
            pairwise_rows.append({
                "higher_ranked_model": left_model,
                "higher_ranked_model_label": MODEL_LABELS.get(left_model, left_model),
                "comparison_model": right_model,
                "comparison_model_label": MODEL_LABELS.get(right_model, right_model),
                "paired_mean_quality_difference": round(difference, 4),
                "difference_ci95_low": round(ci_low, 4),
                "difference_ci95_high": round(ci_high, 4),
                "ci_excludes_zero": ci_low > 0 or ci_high < 0,
                "paired_images": len(left),
            })

    category_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        grouped[(row["caption_model"], row["category"])].append(row)
    for (model, category), rows in sorted(grouped.items()):
        category_rows.append({
            "caption_model": model,
            "caption_model_label": MODEL_LABELS.get(model, model),
            "category": category,
            "captions": len(rows),
            "mean_accuracy": round(mean(row["mean_accuracy"] for row in rows), 4),
            "mean_coverage": round(mean(row["mean_coverage"] for row in rows), 4),
            "mean_clarity": round(mean(row["mean_clarity"] for row in rows), 4),
            "mean_quality": round(mean(row["mean_quality"] for row in rows), 4),
            "final_compliant_usable_rate": round(mean(float(row["final_compliant_usable"]) for row in rows), 4),
        })

    agreement_rows: list[dict[str, Any]] = []
    comparisons = [
        ("Qwen 27B repeatability", "qwen_pass_1", "qwen_pass_2"),
        ("Flash-Next repeatability", "strata_pass_1", "strata_pass_2"),
        ("Cross-judge pass 1", "qwen_pass_1", "strata_pass_1"),
        ("Cross-judge pass 2", "qwen_pass_2", "strata_pass_2"),
    ]
    ordered_keys = sorted(base)
    for comparison, left_name, right_name in comparisons:
        for field in RATING_FIELDS:
            left = [int(pass_rows[left_name][key][field]) for key in ordered_keys]
            right = [int(pass_rows[right_name][key][field]) for key in ordered_keys]
            agreement_rows.append({
                "comparison": comparison,
                "metric": field,
                "n": len(left),
                "exact_agreement": round(mean(float(a == b) for a, b in zip(left, right)), 4),
                "within_one_agreement": round(mean(float(abs(a - b) <= 1) for a, b in zip(left, right)), 4),
                "mean_absolute_difference": round(mean(abs(a - b) for a, b in zip(left, right)), 4),
                "pearson": round(pearson(left, right), 4) if pearson(left, right) is not None else None,
                "spearman": round(pearson(ranks(left), ranks(right)), 4) if pearson(ranks(left), ranks(right)) is not None else None,
                "weighted_kappa": round(quadratic_weighted_kappa(left, right), 4) if quadratic_weighted_kappa(left, right) is not None else None,
            })
        left_bool = [boolean(pass_rows[left_name][key]["usable"]) for key in ordered_keys]
        right_bool = [boolean(pass_rows[right_name][key]["usable"]) for key in ordered_keys]
        agreement_rows.append({
            "comparison": comparison,
            "metric": "usable",
            "n": len(left_bool),
            "exact_agreement": round(mean(float(a == b) for a, b in zip(left_bool, right_bool)), 4),
            "within_one_agreement": 1.0,
            "mean_absolute_difference": round(mean(abs(int(a) - int(b)) for a, b in zip(left_bool, right_bool)), 4),
            "pearson": None,
            "spearman": None,
            "weighted_kappa": round(cohen_kappa(left_bool, right_bool), 4) if cohen_kappa(left_bool, right_bool) is not None else None,
        })

    pass_summary_rows: list[dict[str, Any]] = []
    for pass_name, rows_by_key in pass_rows.items():
        rows = list(rows_by_key.values())
        pass_summary_rows.append({
            "pass": pass_name,
            "judge_family": "Qwen3.8 27B Q4_K_M" if pass_name.startswith("qwen") else "Qwen3.8-Flash-Next IQ3_XXS",
            "seed": int(rows[0]["seed"]),
            "judgments": len(rows),
            "mean_accuracy": round(mean(float(row["accuracy"]) for row in rows), 4),
            "mean_coverage": round(mean(float(row["coverage"]) for row in rows), 4),
            "mean_clarity": round(mean(float(row["clarity"]) for row in rows), 4),
            "usable_rate": round(mean(float(boolean(row["usable"])) for row in rows), 4),
            "mean_wall_time_seconds": round(mean(float(row["wall_time_seconds"]) for row in rows), 4),
            "mean_generation_tokens_per_second": round(mean(float(row["generation_tokens_per_second"]) for row in rows if row["generation_tokens_per_second"]), 4),
            "total_gpu_energy_wh": round(sum(float(row["gpu_energy_wh_estimate"]) for row in rows if row["gpu_energy_wh_estimate"]), 4),
            "peak_gpu_memory_mib": max(float(row["gpu_memory_peak_mib"]) for row in rows if row["gpu_memory_peak_mib"]),
            "peak_gpu_temperature_c": max(float(row["gpu_temperature_peak_c"]) for row in rows if row["gpu_temperature_peak_c"]),
            "peak_system_ram_gib": round(max(float(row["system_ram_used_peak_bytes"]) for row in rows if row["system_ram_used_peak_bytes"]) / 2**30, 4),
        })

    efficiency_rows: list[dict[str, Any]] = []
    generation_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in caption_rows:
        generation_by_model[row["model"]].append(row)
    for model, rows in generation_by_model.items():
        total_seconds = sum(float(row["wall_time_seconds"]) for row in rows)
        energy_wh = sum(float(row["gpu_energy_wh_estimate"]) for row in rows)
        seconds_per_image = total_seconds / len(rows)
        energy_wh_per_image = energy_wh / len(rows)
        efficiency_rows.append({
            "caption_model": model,
            "caption_model_label": MODEL_LABELS.get(model, model),
            "captions": len(rows),
            "mean_wall_time_seconds": round(seconds_per_image, 4),
            "images_per_minute": round(60 / seconds_per_image, 4),
            "mean_generation_tokens_per_second": round(mean(float(row["generation_tokens_per_second"]) for row in rows if row["generation_tokens_per_second"]), 4),
            "peak_gpu_memory_mib": round(max(float(row["gpu_memory_peak_mib"]) for row in rows if row["gpu_memory_peak_mib"]), 2),
            "peak_system_ram_gib": round(max(float(row["system_ram_used_peak_bytes"]) for row in rows if row["system_ram_used_peak_bytes"]) / 2**30, 4),
            "mean_gpu_power_w": round(mean(float(row["gpu_power_average_w"]) for row in rows if row["gpu_power_average_w"]), 4),
            "peak_gpu_power_w": round(max(float(row["gpu_power_peak_w"]) for row in rows if row["gpu_power_peak_w"]), 4),
            "peak_gpu_temperature_c": round(max(float(row["gpu_temperature_peak_c"]) for row in rows if row["gpu_temperature_peak_c"]), 2),
            "measured_gpu_energy_wh": round(energy_wh, 6),
            "gpu_energy_wh_per_image": round(energy_wh_per_image, 6),
            "hours_per_100k_images": round(seconds_per_image * 100000 / 3600, 2),
            "gpu_energy_kwh_per_100k_images": round(energy_wh_per_image * 100000 / 1000, 4),
            "illustrative_electricity_cost_inr_per_100k_at_8_inr_kwh": round(energy_wh_per_image * 100000 / 1000 * 8, 2),
            "format_valid_rate": round(mean(float(boolean(row["caption_format_valid"])) for row in rows), 4),
        })
    efficiency_rows.sort(key=lambda row: row["mean_wall_time_seconds"])

    write_csv(OUTPUT / "caption_level_consensus.csv", consensus_rows)
    write_csv(OUTPUT / "model_ranking.csv", ranking_rows)
    write_csv(OUTPUT / "pairwise_model_differences.csv", pairwise_rows)
    write_csv(OUTPUT / "category_scores.csv", category_rows)
    write_csv(OUTPUT / "judge_agreement.csv", agreement_rows)
    write_csv(OUTPUT / "judge_pass_summary.csv", pass_summary_rows)
    write_csv(OUTPUT / "generation_efficiency.csv", efficiency_rows)

    top = ranking_rows[0]
    summary = {
        "schema_version": 2,
        "integrity": {
            **integrity,
            "unique_caption_keys": len(base),
            "judgments_per_caption": 4,
            "total_judgments": len(base) * 4,
            "complete": len(consensus_rows) == 480,
        },
        "method": {
            "judge_passes": list(PASSES),
            "caption_score": "unweighted mean of accuracy, coverage, and clarity across four blind judge passes",
            "usable_consensus": "at least 3 of 4 judge votes",
            "final_compliant_usable": "majority usable and deterministic 8-unique-comma-phrase format valid",
            "quality_ci": "95% image-level bootstrap interval, 5000 resamples, seed 20261013",
            "electricity_cost": "illustrative INR 8/kWh applied only to measured request-level GPU energy; hardware and idle/system energy excluded",
        },
        "top_model": top,
        "model_ranking": ranking_rows,
        "pairwise_model_differences": pairwise_rows,
        "judge_pass_summary": pass_summary_rows,
        "generation_efficiency": efficiency_rows,
    }
    write_json(OUTPUT / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    aggregate()
