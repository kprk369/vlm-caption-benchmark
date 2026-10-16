"""Run a resumable, blinded local VLM-as-judge benchmark with telemetry."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_caption_benchmark import (
    GpuSampler,
    SystemSampler,
    api_json,
    atomic_json,
    gpu_idle_snapshot,
    now_utc,
    ns_to_seconds,
    ollama_process_working_set_bytes,
    sha256_file,
    system_information,
    unload_models,
)


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("local-data/final-runs/captions-120x4/benchmark.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("local-data/final-120/images"))
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/judge.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--model", default="qwen3.8:27b-q4_K_M")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


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
    items: list[dict[str, Any]] = []
    verified_images: dict[str, str] = {}
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
        item["trial_id"] = f"J{index:04d}-{digest}"
    return items


def model_inventory(base_url: str, timeout: int) -> dict[str, dict[str, Any]]:
    response = api_json(f"{base_url}/api/tags", timeout=timeout)
    return {entry["name"]: entry for entry in response.get("models", [])}


def make_payload(args: argparse.Namespace, prompt: str, item: dict[str, Any]) -> dict[str, Any]:
    image = base64.b64encode(Path(item["image_path"]).read_bytes()).decode("ascii")
    return {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": f"{prompt}\n\nCandidate caption:\n{item['candidate_caption']}",
            "images": [image],
        }],
        "stream": False,
        "think": False,
        "format": judgment_schema(),
        "keep_alive": "30m",
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judgment is not an object")
    required = set(judgment_schema()["required"])
    if set(value) != required:
        raise ValueError(f"judgment fields differ: {sorted(set(value) ^ required)}")
    for field in ("accuracy", "coverage", "clarity"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or not 1 <= value[field] <= 5:
            raise ValueError(f"invalid {field}: {value[field]!r}")
    if not isinstance(value["usable"], bool):
        raise ValueError("usable is not boolean")
    if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], (int, float)):
        raise ValueError("confidence is not numeric")
    if not 0 <= value["confidence"] <= 1:
        raise ValueError("confidence is outside 0..1")
    for field in ("key_error", "key_omission", "clarity_issue", "rationale"):
        if not isinstance(value[field], str):
            raise ValueError(f"{field} is not text")
    return value


def request_judgment(
    args: argparse.Namespace, prompt: str, item: dict[str, Any], idle_mib: int | None
) -> dict[str, Any]:
    response: dict[str, Any] = {}
    last_error: str | None = None
    attempts = 0
    gpu_sampler = GpuSampler()
    system_sampler = SystemSampler()
    ollama_ram_start = ollama_process_working_set_bytes()
    gpu_sampler.start()
    system_sampler.start()
    started_at = now_utc()
    started = time.perf_counter()
    judgment: dict[str, Any] | None = None
    for attempts in range(1, args.retries + 2):
        try:
            response = api_json(
                f"{args.ollama_url}/api/chat",
                make_payload(args, prompt, item),
                timeout=args.timeout,
            )
            message = response.get("message") or {}
            content = message.get("content")
            if not content:
                raise ValueError(f"empty content; done_reason={response.get('done_reason')}")
            judgment = validate_judgment(json.loads(content))
            last_error = None
            break
        except Exception as exc:
            last_error = f"ResponseValidationError: {type(exc).__name__}: {exc}"
            if attempts <= args.retries:
                time.sleep(2 * attempts)
    wall_time = round(time.perf_counter() - started, 6)
    gpu = gpu_sampler.stop(idle_mib)
    system = system_sampler.stop()
    ollama_ram_end = ollama_process_working_set_bytes()
    message = response.get("message") or {}
    eval_count = response.get("eval_count")
    eval_seconds = ns_to_seconds(response.get("eval_duration"))
    observed_ram = [value for value in (ollama_ram_start, ollama_ram_end) if value is not None]
    return {
        **{key: value for key, value in item.items() if key != "image_path"},
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "status": "ok" if judgment is not None else "error",
        "attempt_count": attempts,
        "error": last_error,
        "judgment": judgment,
        "thinking": message.get("thinking"),
        "done_reason": response.get("done_reason"),
        "wall_time_seconds": wall_time,
        "ollama_total_duration_seconds": ns_to_seconds(response.get("total_duration")),
        "ollama_load_duration_seconds": ns_to_seconds(response.get("load_duration")),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_seconds": ns_to_seconds(response.get("prompt_eval_duration")),
        "eval_count": eval_count,
        "eval_duration_seconds": eval_seconds,
        "generation_tokens_per_second": round(eval_count / eval_seconds, 4) if isinstance(eval_count, int) and eval_seconds else None,
        **gpu,
        **system,
        "ollama_process_working_set_start_bytes": ollama_ram_start,
        "ollama_process_working_set_end_bytes": ollama_ram_end,
        "ollama_process_working_set_max_observed_bytes": max(observed_ram) if observed_ram else None,
        "raw_ollama_response": response,
    }


CSV_FIELDS = [
    "pass_id", "trial_id", "order_index", "dataset_index", "image", "category", "image_sha256",
    "caption_model", "caption_model_digest", "candidate_caption", "candidate_format_valid",
    "judge_model", "judge_model_digest", "seed", "temperature", "num_predict", "num_ctx",
    "status", "attempt_count", "error", "accuracy", "coverage", "clarity", "usable",
    "confidence", "key_error", "key_omission", "clarity_issue", "rationale",
    "started_at_utc", "finished_at_utc", "done_reason", "wall_time_seconds",
    "ollama_total_duration_seconds", "ollama_load_duration_seconds", "prompt_eval_count",
    "prompt_eval_duration_seconds", "eval_count", "eval_duration_seconds", "generation_tokens_per_second",
    "gpu_memory_request_start_mib", "gpu_memory_peak_mib", "gpu_memory_peak_minus_system_idle_mib",
    "gpu_power_average_w", "gpu_power_peak_w", "gpu_utilization_average_percent",
    "gpu_utilization_peak_percent", "gpu_memory_utilization_average_percent",
    "gpu_temperature_average_c", "gpu_temperature_peak_c", "gpu_sm_clock_average_mhz",
    "gpu_sm_clock_peak_mhz", "gpu_memory_clock_average_mhz", "gpu_memory_clock_peak_mhz",
    "gpu_fan_average_percent", "gpu_performance_state_mode", "gpu_energy_wh_estimate",
    "gpu_sample_count", "gpu_sample_interval_seconds", "system_cpu_utilization_average_percent",
    "system_cpu_utilization_peak_percent", "system_ram_used_average_bytes", "system_ram_used_peak_bytes",
    "system_ram_used_average_percent", "system_ram_used_peak_percent", "system_sample_count",
    "system_sample_interval_seconds", "cpu_temperature_c", "cpu_temperature_status",
    "ollama_process_working_set_start_bytes", "ollama_process_working_set_end_bytes",
    "ollama_process_working_set_max_observed_bytes",
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
                "judge_model_digest": document["judge_model"].get("digest"),
                "seed": document["settings"]["seed"],
                "temperature": document["settings"]["temperature"],
                "num_predict": document["settings"]["num_predict"],
                "num_ctx": document["settings"]["num_ctx"],
            })
            writer.writerow(row)
    temporary.replace(path)


def make_document(args: argparse.Namespace, items: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "test_type": "blind_local_vlm_judge",
        "pass_id": args.pass_id,
        "started_at_utc": now_utc(),
        "completed_at_utc": None,
        "input": {"path": str(args.input), "sha256": sha256_file(args.input), "item_count": len(items)},
        "images_dir": str(args.images_dir),
        "prompt": {"path": str(args.prompt_file), "sha256": sha256_file(args.prompt_file), "text": args.prompt_file.read_text(encoding="utf-8")},
        "judge_model": model,
        "settings": {
            "seed": args.seed, "temperature": args.temperature, "num_predict": args.num_predict,
            "num_ctx": args.num_ctx, "think": False, "structured_output": True,
            "blind_to_caption_model": True, "trial_order": [item["source_key"] for item in items],
        },
        "system": system_information(args.ollama_url),
        "results": [],
    }


def validate_resume(document: dict[str, Any], args: argparse.Namespace, items: list[dict[str, Any]], model: dict[str, Any]) -> None:
    checks = {
        "schema": document.get("schema_version") == SCHEMA_VERSION,
        "pass": document.get("pass_id") == args.pass_id,
        "input": document.get("input", {}).get("sha256") == sha256_file(args.input),
        "prompt": document.get("prompt", {}).get("sha256") == sha256_file(args.prompt_file),
        "judge model": document.get("judge_model", {}).get("digest") == model.get("digest"),
        "seed": document.get("settings", {}).get("seed") == args.seed,
        "temperature": document.get("settings", {}).get("temperature") == args.temperature,
        "order": document.get("settings", {}).get("trial_order") == [item["source_key"] for item in items],
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError("Existing run does not match current " + ", ".join(failed))


def write_summary(document: dict[str, Any], path: Path) -> None:
    results = document["results"]
    ok = [row for row in results if row.get("status") == "ok"]
    energy = [row.get("gpu_energy_wh_estimate") for row in ok if row.get("gpu_energy_wh_estimate") is not None]
    summary = {
        "pass_id": document["pass_id"],
        "expected_results": document["input"]["item_count"],
        "successful_results": len(ok),
        "error_results": sum(row.get("status") == "error" for row in results),
        "usable_results": sum(bool((row.get("judgment") or {}).get("usable")) for row in ok),
        "total_wall_time_seconds": round(sum(row.get("wall_time_seconds") or 0 for row in results), 6),
        "total_gpu_energy_wh_estimate": round(sum(energy), 6),
        "peak_gpu_memory_mib": max((row.get("gpu_memory_peak_mib") or 0 for row in results), default=None),
        "peak_gpu_temperature_c": max((row.get("gpu_temperature_peak_c") or 0 for row in results), default=None),
    }
    atomic_json(path, summary)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.input, args.images_dir, args.seed)
    inventory = model_inventory(args.ollama_url, min(args.timeout, 30))
    if args.model not in inventory:
        raise SystemExit(f"Ollama judge model is not installed: {args.model}")
    shown = api_json(f"{args.ollama_url}/api/show", {"model": args.model}, min(args.timeout, 30))
    capabilities = shown.get("capabilities") or []
    if "vision" not in capabilities:
        raise SystemExit(f"Judge model does not report vision capability: {args.model}")
    preflight = {
        "created_at_utc": now_utc(), "ready": len(items) == 480 and len({item['source_key'] for item in items}) == 480,
        "item_count": len(items), "unique_images": len({item['image'] for item in items}),
        "caption_models": dict(Counter(item["caption_model"] for item in items)),
        "judge_model": args.model, "judge_model_digest": inventory[args.model].get("digest"),
        "judge_capabilities": capabilities, "input_sha256": sha256_file(args.input),
        "prompt_sha256": sha256_file(args.prompt_file), "seed": args.seed, "temperature": args.temperature,
    }
    atomic_json(args.output_dir / "preflight.json", preflight)
    if not preflight["ready"]:
        raise SystemExit("Preflight failed: expected 480 unique caption inputs")
    print(f"Validated 480 blind pairs - {args.pass_id} - seed {args.seed}", flush=True)
    if args.preflight_only:
        print("Preflight complete; no judgments generated.", flush=True)
        return

    output_json = args.output_dir / "judgments.json"
    output_csv = args.output_dir / "judgments.csv"
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if output_json.exists():
        document = json.loads(output_json.read_text(encoding="utf-8"))
        validate_resume(document, args, items, inventory[args.model])
    else:
        document = make_document(args, items, inventory[args.model])
    done = {row["source_key"] for row in document["results"] if row.get("status") == "ok"}
    pending = [item for item in items if item["source_key"] not in done]
    if args.limit is not None:
        pending = pending[:args.limit]

    unload_models(args.ollama_url)
    idle = gpu_idle_snapshot()
    document["gpu_idle_before_pass"] = idle
    try:
        for position, item in enumerate(pending, start=1):
            print(f"[{len(done)+position}/480] Judging {item['trial_id']} - {item['image']}", flush=True)
            result = request_judgment(args, prompt, item, idle.get("memory_mib"))
            document["results"] = [row for row in document["results"] if row["source_key"] != item["source_key"]]
            document["results"].append(result)
            document["results"].sort(key=lambda row: row["order_index"])
            atomic_json(output_json, document)
            write_csv(document, output_csv)
            write_summary(document, args.output_dir / "summary.json")
            print(f"  {result['status']} in {result['wall_time_seconds']:.1f}s", flush=True)
    finally:
        unload_models(args.ollama_url)

    completed = sum(row.get("status") == "ok" for row in document["results"])
    document["completed_at_utc"] = now_utc() if completed == len(items) else None
    atomic_json(output_json, document)
    write_csv(document, output_csv)
    write_summary(document, args.output_dir / "summary.json")
    print(f"Finished: {completed}/480 successful judgments", flush=True)


if __name__ == "__main__":
    main()
