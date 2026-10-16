"""Run a resumable, measurement-rich local VLM caption benchmark via Ollama."""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


SCHEMA_VERSION = 2
DEFAULT_MODELS = [
    "qwen3.8:27b-q4_K_M",
    "gemma4:12b-it-q8_0",
    "gemma3:4b-it-q8_0",
    "qwen3-vl:2b-instruct-q8_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("local-data/final-120/manifest.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("local-data/final-120/images"))
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/caption.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", help="Dataset indices to run; default is all rows.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the complete run without loading a model or generating captions.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body or exc.reason}") from exc


def ns_to_seconds(value: Any) -> float | None:
    return round(value / 1_000_000_000, 6) if isinstance(value, int) else None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def unload_models(base_url: str) -> None:
    deadline = time.monotonic() + 120
    while True:
        running = api_json(f"{base_url}/api/ps").get("models", [])
        if not running:
            time.sleep(2)
            return
        for item in running:
            name = item.get("name") or item.get("model")
            if name:
                api_json(f"{base_url}/api/generate", {"model": name, "keep_alive": 0})
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for Ollama to unload its models.")
        time.sleep(1)


class GpuSampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.interval = interval_seconds
        self.samples: list[dict[str, float | int | None]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample_once(self) -> dict[str, float | int | None] | None:
        if not self.executable:
            return None
        try:
            result = subprocess.run(
                [
                    self.executable,
                    "--query-gpu=memory.used,power.draw,utilization.gpu,utilization.memory,"
                    "temperature.gpu,clocks.sm,clocks.mem,fan.speed,pstate",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            def number(value: str) -> float | None:
                try:
                    return float(value)
                except ValueError:
                    return None
            return {
                "t": time.perf_counter(),
                "memory_mib": int(float(values[0])),
                "power_w": number(values[1]),
                "utilization_percent": number(values[2]),
                "memory_utilization_percent": number(values[3]),
                "temperature_c": number(values[4]),
                "sm_clock_mhz": number(values[5]),
                "memory_clock_mhz": number(values[6]),
                "fan_percent": number(values[7]),
                "performance_state": values[8] or None,
            }
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            sample = self.sample_once()
            if sample:
                self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        first = self.sample_once()
        if first:
            self.samples.append(first)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self, system_idle_mib: int | None) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        memories = [int(s["memory_mib"]) for s in self.samples if s["memory_mib"] is not None]
        powers = [float(s["power_w"]) for s in self.samples if s["power_w"] is not None]
        utils = [float(s["utilization_percent"]) for s in self.samples if s["utilization_percent"] is not None]
        memory_utils = [float(s["memory_utilization_percent"]) for s in self.samples if s["memory_utilization_percent"] is not None]
        temperatures = [float(s["temperature_c"]) for s in self.samples if s["temperature_c"] is not None]
        sm_clocks = [float(s["sm_clock_mhz"]) for s in self.samples if s["sm_clock_mhz"] is not None]
        memory_clocks = [float(s["memory_clock_mhz"]) for s in self.samples if s["memory_clock_mhz"] is not None]
        fans = [float(s["fan_percent"]) for s in self.samples if s["fan_percent"] is not None]
        pstates = [str(s["performance_state"]) for s in self.samples if s["performance_state"]]
        peak = max(memories) if memories else None
        duration = 0.0
        if len(self.samples) >= 2:
            duration = float(self.samples[-1]["t"]) - float(self.samples[0]["t"])
        return {
            "gpu_memory_request_start_mib": memories[0] if memories else None,
            "gpu_memory_peak_mib": peak,
            "gpu_memory_peak_minus_system_idle_mib": (
                peak - system_idle_mib if peak is not None and system_idle_mib is not None else None
            ),
            "gpu_power_average_w": round(sum(powers) / len(powers), 3) if powers else None,
            "gpu_power_peak_w": round(max(powers), 3) if powers else None,
            "gpu_utilization_average_percent": round(sum(utils) / len(utils), 3) if utils else None,
            "gpu_utilization_peak_percent": round(max(utils), 3) if utils else None,
            "gpu_memory_utilization_average_percent": round(sum(memory_utils) / len(memory_utils), 3) if memory_utils else None,
            "gpu_temperature_average_c": round(sum(temperatures) / len(temperatures), 3) if temperatures else None,
            "gpu_temperature_peak_c": round(max(temperatures), 3) if temperatures else None,
            "gpu_sm_clock_average_mhz": round(sum(sm_clocks) / len(sm_clocks), 3) if sm_clocks else None,
            "gpu_sm_clock_peak_mhz": round(max(sm_clocks), 3) if sm_clocks else None,
            "gpu_memory_clock_average_mhz": round(sum(memory_clocks) / len(memory_clocks), 3) if memory_clocks else None,
            "gpu_memory_clock_peak_mhz": round(max(memory_clocks), 3) if memory_clocks else None,
            "gpu_fan_average_percent": round(sum(fans) / len(fans), 3) if fans else None,
            "gpu_performance_state_mode": Counter(pstates).most_common(1)[0][0] if pstates else None,
            "gpu_energy_wh_estimate": round((sum(powers) / len(powers)) * duration / 3600, 6) if powers else None,
            "gpu_sample_count": len(self.samples),
            "gpu_sample_interval_seconds": self.interval,
        }


class SystemSampler:
    """Sample total CPU load and physical-memory use without extra packages."""

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval = interval_seconds
        self.samples: list[dict[str, float | int | None]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.previous_cpu: tuple[int, int, int] | None = None

    @staticmethod
    def _filetime_value(value: FileTime) -> int:
        return (int(value.high) << 32) | int(value.low)

    def sample_once(self) -> dict[str, float | int | None] | None:
        if os.name != "nt":
            return None
        idle, kernel, user = self.FileTime(), self.FileTime(), self.FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        current = tuple(self._filetime_value(item) for item in (idle, kernel, user))
        cpu_percent = None
        if self.previous_cpu:
            idle_delta = current[0] - self.previous_cpu[0]
            total_delta = (current[1] - self.previous_cpu[1]) + (current[2] - self.previous_cpu[2])
            if total_delta > 0:
                cpu_percent = max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
        self.previous_cpu = current
        memory = self.MemoryStatus()
        memory.length = ctypes.sizeof(self.MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return None
        used = int(memory.total_physical - memory.available_physical)
        return {
            "t": time.perf_counter(),
            "cpu_percent": cpu_percent,
            "ram_used_bytes": used,
            "ram_used_percent": 100.0 * used / int(memory.total_physical),
        }

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            sample = self.sample_once()
            if sample:
                self.samples.append(sample)
            self.stop_event.wait(self.interval)

    def start(self) -> None:
        first = self.sample_once()
        if first:
            self.samples.append(first)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        cpu = [float(s["cpu_percent"]) for s in self.samples if s["cpu_percent"] is not None]
        ram_bytes = [int(s["ram_used_bytes"]) for s in self.samples if s["ram_used_bytes"] is not None]
        ram_percent = [float(s["ram_used_percent"]) for s in self.samples if s["ram_used_percent"] is not None]
        return {
            "system_cpu_utilization_average_percent": round(sum(cpu) / len(cpu), 3) if cpu else None,
            "system_cpu_utilization_peak_percent": round(max(cpu), 3) if cpu else None,
            "system_ram_used_average_bytes": round(sum(ram_bytes) / len(ram_bytes)) if ram_bytes else None,
            "system_ram_used_peak_bytes": max(ram_bytes) if ram_bytes else None,
            "system_ram_used_average_percent": round(sum(ram_percent) / len(ram_percent), 3) if ram_percent else None,
            "system_ram_used_peak_percent": round(max(ram_percent), 3) if ram_percent else None,
            "system_sample_count": len(self.samples),
            "system_sample_interval_seconds": self.interval,
            "cpu_temperature_c": None,
            "cpu_temperature_status": "unavailable_from_standard_windows_sensor APIs",
        }


def ollama_process_working_set_bytes() -> int | None:
    if os.name != "nt":
        return None
    command = "(Get-Process -Name ollama -ErrorAction SilentlyContinue|Measure-Object WorkingSet64 -Sum).Sum"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=10, check=True,
        )
        value = result.stdout.strip()
        return int(value) if value else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def gpu_idle_snapshot() -> dict[str, Any]:
    sampler = GpuSampler()
    sample = sampler.sample_once()
    return sample or {"memory_mib": None, "power_w": None, "utilization_percent": None}


def gpu_inventory() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    fields = ["name", "memory.total", "driver_version"]
    result = subprocess.run(
        [executable, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    return [dict(zip(fields, (part.strip() for part in line.split(",")))) for line in result.stdout.splitlines()]


def ollama_version(base_url: str) -> str | None:
    try:
        version = api_json(f"{base_url}/api/version").get("version")
        if version:
            return str(version)
    except Exception:
        pass
    candidates = [
        shutil.which("ollama"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"),
    ]
    for candidate in candidates:
        if candidate:
            try:
                if not Path(candidate).is_file():
                    continue
                result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=10)
                return (result.stdout or result.stderr).strip() or None
            except (OSError, PermissionError):
                pass
    return None


def windows_hardware() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    details: dict[str, Any] = {}
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            details["cpu_name"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        pass

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]
    try:
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            details["total_physical_memory_bytes"] = int(memory.total_physical)
    except (AttributeError, OSError):
        pass
    return details


def system_information(base_url: str) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "ollama_version": ollama_version(base_url),
        "gpu_inventory": gpu_inventory(),
        **windows_hardware(),
    }


def load_images(manifest_path: Path, images_dir: Path, indices: list[int] | None) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(indices) if indices else None
    selected = [row for row in rows if requested is None or int(row["dataset_index"]) in requested]
    if requested and {int(row["dataset_index"]) for row in selected} != requested:
        raise RuntimeError("One or more requested dataset indices are absent from the manifest.")
    selected.sort(key=lambda row: int(row["dataset_index"]))
    result: list[dict[str, Any]] = []
    for row in selected:
        path = (images_dir / row["filename"]).resolve()
        with Image.open(path) as image:
            width, height = image.size
        result.append({
            "dataset_index": int(row["dataset_index"]),
            "filename": row["filename"],
            "category": row["category"],
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "width": width,
            "height": height,
        })
    return result


def caption_format_metrics(caption: str | None) -> dict[str, Any]:
    text = (caption or "").strip()
    phrases = [part.strip() for part in text.split(",") if part.strip()]
    normalized = {" ".join(part.lower().split()) for part in phrases}
    return {
        "caption_character_count": len(text),
        "caption_line_count": len(text.splitlines()) if text else 0,
        "caption_phrase_count": len(phrases),
        "caption_unique_phrase_count": len(normalized),
        "caption_format_valid": bool(text) and "\n" not in text and len(phrases) == 8 and len(phrases) == len(normalized),
    }


def request_caption(args: argparse.Namespace, model: str, prompt: str, image: dict[str, Any], idle_mib: int | None) -> dict[str, Any]:
    encoded = base64.b64encode(Path(image["path"]).read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [encoded]}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }
    last_error: str | None = None
    response: dict[str, Any] = {}
    attempts = 0
    sampler = GpuSampler()
    system_sampler = SystemSampler()
    ollama_ram_start = ollama_process_working_set_bytes()
    sampler.start()
    system_sampler.start()
    started_utc = now_utc()
    started = time.perf_counter()
    for attempts in range(1, args.retries + 2):
        try:
            response = api_json(f"{args.ollama_url}/api/chat", payload, timeout=args.timeout)
            last_error = None
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts <= args.retries:
                time.sleep(2 * attempts)
    wall_seconds = round(time.perf_counter() - started, 6)
    gpu = sampler.stop(idle_mib)
    system = system_sampler.stop()
    ollama_ram_end = ollama_process_working_set_bytes()
    message = response.get("message") or {}
    caption = message.get("content")
    eval_count = response.get("eval_count")
    eval_seconds = ns_to_seconds(response.get("eval_duration"))
    return {
        "dataset_index": image["dataset_index"],
        "image": image["filename"],
        "category": image["category"],
        "image_sha256": image["sha256"],
        "model": model,
        "started_at_utc": started_utc,
        "finished_at_utc": now_utc(),
        "status": "error" if last_error else "ok",
        "attempt_count": attempts,
        "error": last_error,
        "caption": caption,
        "thinking": message.get("thinking"),
        "done_reason": response.get("done_reason"),
        "wall_time_seconds": wall_seconds,
        "ollama_total_duration_seconds": ns_to_seconds(response.get("total_duration")),
        "ollama_load_duration_seconds": ns_to_seconds(response.get("load_duration")),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_seconds": ns_to_seconds(response.get("prompt_eval_duration")),
        "eval_count": eval_count,
        "eval_duration_seconds": eval_seconds,
        "generation_tokens_per_second": round(eval_count / eval_seconds, 4) if isinstance(eval_count, int) and eval_seconds else None,
        **caption_format_metrics(caption),
        **gpu,
        **system,
        "ollama_process_working_set_start_bytes": ollama_ram_start,
        "ollama_process_working_set_end_bytes": ollama_ram_end,
        "ollama_process_working_set_max_observed_bytes": max(
            value for value in (ollama_ram_start, ollama_ram_end) if value is not None
        ) if ollama_ram_start is not None or ollama_ram_end is not None else None,
        "raw_ollama_response": response,
    }


CSV_FIELDS = [
    "dataset_index", "image", "category", "image_sha256", "image_size_bytes",
    "image_width_px", "image_height_px", "model", "model_digest", "model_size_bytes",
    "model_parameter_size", "model_quantization_level", "model_family", "model_format",
    "prompt_sha256", "seed", "temperature", "num_predict", "num_ctx", "think", "stream",
    "ollama_version", "gpu_name", "gpu_total_memory_mib", "gpu_driver_version",
    "started_at_utc", "finished_at_utc",
    "status", "attempt_count", "error", "caption", "done_reason", "wall_time_seconds",
    "ollama_total_duration_seconds", "ollama_load_duration_seconds", "prompt_eval_count",
    "prompt_eval_duration_seconds", "eval_count", "eval_duration_seconds",
    "generation_tokens_per_second", "caption_character_count", "caption_line_count",
    "caption_phrase_count", "caption_unique_phrase_count", "caption_format_valid",
    "gpu_memory_request_start_mib", "gpu_memory_peak_mib",
    "gpu_memory_peak_minus_system_idle_mib", "gpu_power_average_w", "gpu_power_peak_w",
    "gpu_utilization_average_percent", "gpu_energy_wh_estimate", "gpu_sample_count",
    "gpu_sample_interval_seconds",
    "gpu_utilization_peak_percent", "gpu_memory_utilization_average_percent",
    "gpu_temperature_average_c", "gpu_temperature_peak_c", "gpu_sm_clock_average_mhz",
    "gpu_sm_clock_peak_mhz", "gpu_memory_clock_average_mhz", "gpu_memory_clock_peak_mhz",
    "gpu_fan_average_percent", "gpu_performance_state_mode",
    "system_cpu_utilization_average_percent", "system_cpu_utilization_peak_percent",
    "system_ram_used_average_bytes", "system_ram_used_peak_bytes",
    "system_ram_used_average_percent", "system_ram_used_peak_percent",
    "system_sample_count", "system_sample_interval_seconds", "cpu_temperature_c", "cpu_temperature_status",
    "ollama_process_working_set_start_bytes", "ollama_process_working_set_end_bytes",
    "ollama_process_working_set_max_observed_bytes",
]


def write_csv(document: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        models = {entry["name"]: entry for entry in document["models"]}
        images = {entry["filename"]: entry for entry in document["images"]}
        gpu = (document.get("system", {}).get("gpu_inventory") or [{}])[0]
        for result in document["results"]:
            row = {field: result.get(field) for field in CSV_FIELDS}
            model = models[result["model"]]
            details = model.get("details") or {}
            image = images[result["image"]]
            settings = document["settings"]
            row.update({
                "image_size_bytes": image.get("size_bytes"),
                "image_width_px": image.get("width"),
                "image_height_px": image.get("height"),
                "model_digest": model.get("digest"),
                "model_size_bytes": model.get("size"),
                "model_parameter_size": details.get("parameter_size"),
                "model_quantization_level": details.get("quantization_level"),
                "model_family": details.get("family"),
                "model_format": details.get("format"),
                "prompt_sha256": document["prompt"]["sha256"],
                "seed": settings["seed"],
                "temperature": settings["temperature"],
                "num_predict": settings["num_predict"],
                "num_ctx": settings["num_ctx"],
                "think": settings["think"],
                "stream": settings["stream"],
                "ollama_version": document.get("system", {}).get("ollama_version"),
                "gpu_name": gpu.get("name"),
                "gpu_total_memory_mib": gpu.get("memory.total"),
                "gpu_driver_version": gpu.get("driver_version"),
            })
            writer.writerow(row)
    temporary.replace(path)


def build_preflight(
    args: argparse.Namespace,
    manifest: Path,
    prompt_path: Path,
    images: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    model_checks = []
    for model in args.models:
        shown = api_json(f"{args.ollama_url}/api/show", {"model": model})
        capabilities = shown.get("capabilities") or []
        model_checks.append({
            "model": model,
            "digest": inventory[model].get("digest"),
            "size_bytes": inventory[model].get("size"),
            "details": inventory[model].get("details"),
            "capabilities": capabilities,
            "vision_capability_present": "vision" in capabilities,
        })
    categories = Counter(item["category"] for item in images)
    hashes = [item["sha256"] for item in images]
    disk = shutil.disk_usage(output_dir)
    checks = {
        "image_count_is_120": len(images) == 120,
        "image_hashes_are_unique": len(set(hashes)) == len(hashes),
        "four_models_requested": len(args.models) == 4,
        "all_models_installed": all(model in inventory for model in args.models),
        "all_models_report_vision": all(item["vision_capability_present"] for item in model_checks),
        "prompt_is_nonempty": bool(prompt_path.read_text(encoding="utf-8").strip()),
        "output_has_at_least_1_gib_free": disk.free >= 1024 ** 3,
    }
    return {
        "preflight_schema_version": 1,
        "created_at_utc": now_utc(),
        "ready": all(checks.values()),
        "checks": checks,
        "planned_observations": len(images) * len(args.models),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "settings": {
            "seed": args.seed, "temperature": args.temperature,
            "num_predict": args.num_predict, "num_ctx": args.num_ctx,
            "think": False, "stream": False, "maximum_retries": args.retries,
        },
        "category_counts": dict(categories),
        "models": model_checks,
        "system": system_information(args.ollama_url),
        "gpu_idle_snapshot": gpu_idle_snapshot(),
        "cpu_temperature_collection": {
            "available": False,
            "value": None,
            "reason": "Standard Windows ACPI temperature interface is access denied; no reliable CPU package sensor is exposed.",
        },
        "output_disk_free_bytes": disk.free,
    }


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    images_dir = args.images_dir.resolve()
    prompt_path = args.prompt_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "benchmark.json"
    output_csv = output_dir / "benchmark.csv"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    images = load_images(manifest, images_dir, args.indices)
    inventory = {entry["name"]: entry for entry in api_json(f"{args.ollama_url}/api/tags").get("models", [])}
    missing = [model for model in args.models if model not in inventory]
    if missing:
        raise SystemExit(f"Models not installed: {', '.join(missing)}")
    if args.preflight_only:
        report = build_preflight(args, manifest, prompt_path, images, inventory, output_dir)
        atomic_json(output_dir / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["ready"] else 1
    config_fingerprint = hashlib.sha256(json.dumps({
        "models": args.models, "images": [item["sha256"] for item in images],
        "prompt": sha256_file(prompt_path), "seed": args.seed, "temperature": args.temperature,
        "num_predict": args.num_predict, "num_ctx": args.num_ctx,
    }, sort_keys=True).encode()).hexdigest()

    if output_json.exists():
        document = json.loads(output_json.read_text(encoding="utf-8"))
        if document.get("config_fingerprint") != config_fingerprint:
            raise SystemExit("Existing output belongs to a different configuration; choose a new output directory.")
        print(f"Resuming {output_json}", flush=True)
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "test_type": "final_dry_run" if len(images) < 120 else "final_benchmark",
            "config_fingerprint": config_fingerprint,
            "started_at_utc": now_utc(), "completed_at_utc": None,
            "system": system_information(args.ollama_url),
            "prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path), "text": prompt},
            "settings": {
                "temperature": args.temperature, "seed": args.seed, "num_predict": args.num_predict,
                "num_ctx": args.num_ctx, "think": False, "stream": False, "keep_alive": "30m",
                "timeout_seconds": args.timeout, "maximum_retries": args.retries,
                "processing_order": "models sequential; images ascending within model",
                "warmup_excluded_from_analysis": True,
            },
            "images": images,
            "models": [inventory[name] for name in args.models],
            "model_runs": [], "results": [],
        }
        atomic_json(output_json, document)

    # Refresh non-experimental environment metadata on resume without altering
    # any completed model/image measurements.
    document["system"] = system_information(args.ollama_url)
    atomic_json(output_json, document)

    completed = {(row["model"], row["image"]) for row in document["results"] if row["status"] == "ok"}
    warmup_image = next((item for item in load_images(manifest, images_dir, None) if item not in images), images[0])
    for model_number, model in enumerate(args.models, start=1):
        pending = [item for item in images if (model, item["filename"]) not in completed]
        if not pending:
            print(f"[{model_number}/{len(args.models)}] {model}: already complete", flush=True)
            continue
        print(f"[{model_number}/{len(args.models)}] {model}: unload and warm up", flush=True)
        unload_models(args.ollama_url)
        idle = gpu_idle_snapshot()
        warmup = request_caption(args, model, prompt, warmup_image, idle.get("memory_mib"))
        if warmup["status"] != "ok":
            raise RuntimeError(f"Warm-up failed for {model}: {warmup['error']}")
        model_run = {
            "model": model, "system_idle_gpu_memory_mib": idle.get("memory_mib"),
            "system_idle_gpu_power_w": idle.get("power_w"),
            "warmup": warmup, "ollama_ps_after_warmup": api_json(f"{args.ollama_url}/api/ps"),
        }
        document["model_runs"] = [entry for entry in document["model_runs"] if entry["model"] != model]
        document["model_runs"].append(model_run)
        atomic_json(output_json, document)
        print(f"  warm-up ok ({warmup['wall_time_seconds']:.2f}s)", flush=True)
        for position, image in enumerate(pending, start=1):
            result = request_caption(args, model, prompt, image, idle.get("memory_mib"))
            document["results"] = [
                row for row in document["results"]
                if not (row["model"] == model and row["image"] == image["filename"])
            ]
            document["results"].append(result)
            document["results"].sort(key=lambda row: (args.models.index(row["model"]), row["dataset_index"]))
            atomic_json(output_json, document)
            write_csv(document, output_csv)
            print(
                f"  [{position}/{len(pending)}] {image['filename']}: {result['status']}, "
                f"{result['wall_time_seconds']:.2f}s, {result['gpu_memory_peak_mib']} MiB, "
                f"format={'ok' if result['caption_format_valid'] else 'check'}",
                flush=True,
            )
        unload_models(args.ollama_url)

    expected = len(images) * len(args.models)
    successful = sum(row["status"] == "ok" for row in document["results"])
    document["completed_at_utc"] = now_utc() if successful == expected else None
    document["summary"] = {
        "expected_results": expected, "successful_results": successful,
        "error_results": expected - successful,
        "format_valid_results": sum(bool(row.get("caption_format_valid")) for row in document["results"]),
        "total_gpu_energy_wh_estimate": round(sum(float(row.get("gpu_energy_wh_estimate") or 0) for row in document["results"]), 6),
    }
    atomic_json(output_json, document)
    write_csv(document, output_csv)
    atomic_json(output_dir / "summary.json", document["summary"])
    print(json.dumps(document["summary"], indent=2), flush=True)
    print(f"Saved {output_json} and {output_csv}", flush=True)
    return 0 if successful == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
