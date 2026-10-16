"""Run a resumable, blinded Strata VLM-as-judge benchmark with telemetry."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from run_caption_benchmark import (
    GpuSampler,
    SystemSampler,
    atomic_json,
    gpu_idle_snapshot,
    gpu_inventory,
    now_utc,
    sha256_file,
    windows_hardware,
)


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("local-data/final-runs/captions-120x4/benchmark.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("local-data/final-120/images"))
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/judge.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=480)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def api_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def judgment_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
        "clarity": {"type": "integer", "minimum": 1, "maximum": 5},
        "usable": {"type": "boolean"},
        "key_error": {"type": "string"},
        "key_omission": {"type": "string"},
        "clarity_issue": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def load_items(input_path: Path, images_dir: Path, seed: int) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dataset_index", "image", "category", "image_sha256", "model", "model_digest",
        "caption", "caption_format_valid",
    }
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError("Caption input is empty or missing required columns.")
    seen: set[str] = set()
    verified_images: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    for row in rows:
        source_key = f"{row['dataset_index']}|{row['image']}|{row['model']}"
        if source_key in seen:
            raise RuntimeError(f"Duplicate caption input: {source_key}")
        seen.add(source_key)
        if not row["caption"].strip():
            raise RuntimeError(f"Blank candidate caption: {source_key}")
        image_path = (images_dir / row["image"]).resolve()
        if not image_path.is_file():
            raise RuntimeError(f"Missing image: {image_path}")
        if row["image"] not in verified_images:
            verified_images[row["image"]] = sha256_file(image_path)
        if verified_images[row["image"]] != row["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch: {image_path}")
        items.append({
            "source_key": source_key,
            "dataset_index": int(row["dataset_index"]),
            "image": row["image"],
            "category": row["category"],
            "image_sha256": row["image_sha256"],
            "caption_model": row["model"],
            "caption_model_digest": row["model_digest"],
            "candidate_caption": row["caption"],
            "candidate_format_valid": row["caption_format_valid"].strip().lower() in {"true", "1", "yes"},
            "image_path": str(image_path),
        })
    random.Random(seed).shuffle(items)
    for index, item in enumerate(items, start=1):
        digest = hashlib.sha256(f"{seed}|{item['source_key']}".encode()).hexdigest()[:10]
        item["order_index"] = index
        item["trial_id"] = f"S{index:04d}-{digest}"
    return items


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def make_payload(args: argparse.Namespace, prompt: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "strata",
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Candidate caption:\n" + item["candidate_caption"]},
                    {"type": "image_url", "image_url": {"url": image_data_url(Path(item["image_path"]))}},
                ],
            },
        ],
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "caption_judgment", "strict": True, "schema": judgment_schema()},
        },
    }


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judgment is not an object")
    required = set(judgment_schema()["required"])
    if set(value) != required:
        raise ValueError(f"judgment fields differ: {sorted(set(value) ^ required)}")
    for field in ("accuracy", "coverage", "clarity"):
        if not isinstance(value[field], int) or not 1 <= value[field] <= 5:
            raise ValueError(f"invalid {field}: {value[field]!r}")
    if not isinstance(value["usable"], bool):
        raise ValueError("usable is not boolean")
    if not isinstance(value["confidence"], (int, float)) or not 0 <= value["confidence"] <= 1:
        raise ValueError("confidence is outside 0..1")
    return value


def request_judgment(
    args: argparse.Namespace, prompt: str, item: dict[str, Any], idle_mib: int | None
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    error: str | None = None
    judgment: dict[str, Any] | None = None
    attempts = 0
    gpu_sampler = GpuSampler()
    system_sampler = SystemSampler()
    gpu_sampler.start()
    system_sampler.start()
    started_at = now_utc()
    started = time.perf_counter()
    for attempts in range(1, args.retries + 2):
        try:
            response = api_json(
                f"{args.base_url}/v1/chat/completions",
                make_payload(args, prompt, item),
                args.timeout,
            )
            content = response["choices"][0]["message"]["content"]
            judgment = validate_judgment(json.loads(content))
            error = None
            break
        except Exception as exc:
            judgment = None
            error = f"{type(exc).__name__}: {exc}"
            if attempts <= args.retries:
                time.sleep(2 * attempts)
    wall_time = round(time.perf_counter() - started, 6)
    gpu = gpu_sampler.stop(idle_mib)
    system = system_sampler.stop()
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    choice = (response.get("choices") or [{}])[0]
    return {
        **{key: value for key, value in item.items() if key != "image_path"},
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "status": "ok" if judgment is not None else "error",
        "attempt_count": attempts,
        "error": error,
        "judgment": judgment,
        "finish_reason": choice.get("finish_reason"),
        "wall_time_seconds": wall_time,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_seconds": round(timings.get("prompt_ms", 0) / 1000, 6) if timings.get("prompt_ms") is not None else None,
        "generation_seconds": round(timings.get("predicted_ms", 0) / 1000, 6) if timings.get("predicted_ms") is not None else None,
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "generation_tokens_per_second": timings.get("predicted_per_second"),
        "draft_tokens": timings.get("draft_n"),
        "draft_tokens_accepted": timings.get("draft_n_accepted"),
        **gpu,
        **system,
        "raw_strata_response": response,
    }


CSV_FIELDS = [
    "pass_id", "trial_id", "order_index", "dataset_index", "image", "category", "image_sha256",
    "caption_model", "caption_model_digest", "candidate_caption", "candidate_format_valid",
    "judge_model", "seed", "temperature", "max_tokens", "context_tokens", "status", "attempt_count",
    "error", "accuracy", "coverage", "clarity", "usable", "confidence", "key_error", "key_omission",
    "clarity_issue", "rationale", "started_at_utc", "finished_at_utc", "finish_reason",
    "wall_time_seconds", "prompt_tokens", "completion_tokens", "total_tokens", "prompt_seconds",
    "generation_seconds", "prompt_tokens_per_second", "generation_tokens_per_second", "draft_tokens",
    "draft_tokens_accepted", "gpu_memory_request_start_mib", "gpu_memory_peak_mib",
    "gpu_memory_peak_minus_system_idle_mib", "gpu_power_average_w", "gpu_power_peak_w",
    "gpu_utilization_average_percent", "gpu_utilization_peak_percent",
    "gpu_memory_utilization_average_percent", "gpu_temperature_average_c", "gpu_temperature_peak_c",
    "gpu_sm_clock_average_mhz", "gpu_sm_clock_peak_mhz", "gpu_memory_clock_average_mhz",
    "gpu_memory_clock_peak_mhz", "gpu_fan_average_percent", "gpu_performance_state_mode",
    "gpu_energy_wh_estimate", "gpu_sample_count", "gpu_sample_interval_seconds",
    "system_cpu_utilization_average_percent", "system_cpu_utilization_peak_percent",
    "system_ram_used_average_bytes", "system_ram_used_peak_bytes", "system_ram_used_average_percent",
    "system_ram_used_peak_percent", "system_sample_count", "system_sample_interval_seconds",
    "cpu_temperature_c", "cpu_temperature_status",
]


def write_csv(document: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in sorted(document["results"], key=lambda row: row["order_index"]):
            judgment = result.get("judgment") or {}
            row = {field: result.get(field) for field in CSV_FIELDS}
            row.update({field: judgment.get(field) for field in judgment if field in CSV_FIELDS})
            row.update({
                "pass_id": document["pass_id"],
                "judge_model": document["judge_model"]["name"],
                "seed": document["settings"]["seed"],
                "temperature": document["settings"]["temperature"],
                "max_tokens": document["settings"]["max_tokens"],
                "context_tokens": document["settings"]["context_tokens"],
            })
            writer.writerow(row)
    temporary.replace(path)


def write_summary(document: dict[str, Any], path: Path) -> None:
    results = document["results"]
    ok = [row for row in results if row.get("status") == "ok"]

    def mean_result(field: str) -> float | None:
        values = [float(row[field]) for row in ok if row.get(field) is not None]
        return round(sum(values) / len(values), 4) if values else None

    judgments = [row["judgment"] for row in ok]
    elapsed = sum(row.get("wall_time_seconds") or 0 for row in results)
    remaining = document["input"]["item_count"] - len(ok)
    mean_wall = elapsed / len(results) if results else None
    summary = {
        "pass_id": document["pass_id"],
        "updated_at_utc": now_utc(),
        "expected_results": document["input"]["item_count"],
        "successful_results": len(ok),
        "error_results": sum(row.get("status") == "error" for row in results),
        "remaining_results": remaining,
        "progress_percent": round(100 * len(ok) / document["input"]["item_count"], 2),
        "mean_accuracy": round(sum(x["accuracy"] for x in judgments) / len(judgments), 4) if judgments else None,
        "mean_coverage": round(sum(x["coverage"] for x in judgments) / len(judgments), 4) if judgments else None,
        "mean_clarity": round(sum(x["clarity"] for x in judgments) / len(judgments), 4) if judgments else None,
        "usable_results": sum(x["usable"] for x in judgments),
        "total_wall_time_seconds": round(elapsed, 6),
        "mean_wall_time_seconds": mean_result("wall_time_seconds"),
        "estimated_remaining_seconds": round(remaining * mean_wall, 1) if mean_wall is not None else None,
        "mean_generation_tokens_per_second": mean_result("generation_tokens_per_second"),
        "total_gpu_energy_wh_estimate": round(sum(row.get("gpu_energy_wh_estimate") or 0 for row in ok), 6),
        "peak_gpu_memory_mib": max((row.get("gpu_memory_peak_mib") or 0 for row in results), default=None),
        "peak_gpu_temperature_c": max((row.get("gpu_temperature_peak_c") or 0 for row in results), default=None),
        "peak_system_ram_used_bytes": max((row.get("system_ram_used_peak_bytes") or 0 for row in results), default=None),
    }
    atomic_json(path, summary)


def validate_resume(document: dict[str, Any], args: argparse.Namespace, items: list[dict[str, Any]], prompt_hash: str) -> None:
    checks = {
        "schema": document.get("schema_version") == SCHEMA_VERSION,
        "pass": document.get("pass_id") == args.pass_id,
        "input": document.get("input", {}).get("sha256") == sha256_file(args.input),
        "prompt": document.get("prompt", {}).get("sha256") == prompt_hash,
        "seed": document.get("settings", {}).get("seed") == args.seed,
        "temperature": document.get("settings", {}).get("temperature") == args.temperature,
        "order": document.get("settings", {}).get("trial_order") == [item["source_key"] for item in items],
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError("Existing run does not match current " + ", ".join(failed))


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.input, args.images_dir, args.seed)
    if len(items) != args.expected_count or len({item["source_key"] for item in items}) != args.expected_count:
        raise SystemExit(f"Preflight failed: expected {args.expected_count} unique caption/image pairs")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_hash = sha256_file(args.prompt_file)
    health = api_json(f"{args.base_url}/health", timeout=10)
    if not health.get("loaded") or not health.get("images"):
        raise RuntimeError(f"Strata is not ready with vision: {health}")
    model_info = api_json(f"{args.base_url}/v1/models", timeout=10)
    preflight = {
        "created_at_utc": now_utc(),
        "ready": True,
        "item_count": len(items),
        "unique_images": len({item["image"] for item in items}),
        "caption_models": dict(Counter(item["caption_model"] for item in items)),
        "input_sha256": sha256_file(args.input),
        "prompt_sha256": prompt_hash,
        "strata_health": health,
        "strata_models": model_info,
        "seed": args.seed,
        "temperature": args.temperature,
    }
    atomic_json(args.output_dir / "preflight.json", preflight)
    print(f"Validated {len(items)} blind pairs - {args.pass_id} - seed {args.seed}", flush=True)
    if args.preflight_only:
        print("Preflight complete; no judgments generated.", flush=True)
        return

    output_json = args.output_dir / "judgments.json"
    output_csv = args.output_dir / "judgments.csv"
    if output_json.exists():
        document = json.loads(output_json.read_text(encoding="utf-8"))
        validate_resume(document, args, items, prompt_hash)
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "test_type": "blind_local_strata_vlm_judge",
            "pass_id": args.pass_id,
            "started_at_utc": now_utc(),
            "completed_at_utc": None,
            "input": {"path": str(args.input), "sha256": preflight["input_sha256"], "item_count": len(items)},
            "images_dir": str(args.images_dir),
            "prompt": {"path": str(args.prompt_file), "sha256": prompt_hash, "text": prompt},
            "judge_model": {
                "name": health.get("model"),
                "family": "Qwen3.8-Flash-Next",
                "quantization": "IQ3_XXS",
                "engine": "Strata",
            },
            "settings": {
                "seed": args.seed,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "context_tokens": health.get("max_context"),
                "reasoning_effort": "none",
                "structured_output": True,
                "blind_to_caption_model": True,
                "trial_order": [item["source_key"] for item in items],
            },
            "system": {"gpu_inventory": gpu_inventory(), **windows_hardware()},
            "strata_health": health,
            "results": [],
        }

    done = {row["source_key"] for row in document["results"] if row.get("status") == "ok"}
    pending = [item for item in items if item["source_key"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]
    idle = gpu_idle_snapshot()
    document["gpu_idle_before_pass"] = idle
    for position, item in enumerate(pending, start=1):
        print(f"[{len(done) + position}/{len(items)}] Judging {item['trial_id']} - {item['image']}", flush=True)
        result = request_judgment(args, prompt, item, idle.get("memory_mib"))
        document["results"] = [row for row in document["results"] if row["source_key"] != item["source_key"]]
        document["results"].append(result)
        document["results"].sort(key=lambda row: row["order_index"])
        atomic_json(output_json, document)
        write_csv(document, output_csv)
        write_summary(document, args.output_dir / "summary.json")
        print(f"  {result['status']} in {result['wall_time_seconds']:.2f}s", flush=True)

    completed = sum(row.get("status") == "ok" for row in document["results"])
    document["completed_at_utc"] = now_utc() if completed == len(items) else None
    atomic_json(output_json, document)
    write_csv(document, output_csv)
    write_summary(document, args.output_dir / "summary.json")
    print(f"Finished: {completed}/{len(items)} successful judgments", flush=True)


if __name__ == "__main__":
    main()
