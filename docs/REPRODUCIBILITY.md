# Reproducibility

## Published-analysis reproduction

Python 3.11 or newer is recommended. The aggregate analysis uses only the
standard library; Pillow is required for dataset construction and image-run
validation.

```powershell
python -m pip install -r requirements.txt
python scripts/analysis/aggregate_final_results.py
```

The command reads `results/raw/` and deterministically rewrites the CSV files in
`results/analysis/`. Bootstrap sampling uses fixed seed 20261013.

## Recorded environment

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB reported VRAM
- CPU: Intel Core Ultra 9 275HX
- system memory: approximately 63.4 GiB
- GPU driver: 616.92
- Ollama: 0.40.3
- operating mode: Windows WDDM

Exact model digests and run settings are recorded in
`results/experiment_metadata.json` and the raw CSV files.

## New caption run

Obtain the image data separately, construct `local-data/final-120/`, install the
four Ollama models, and start Ollama. Then run:

```powershell
python scripts/benchmark/run_caption_benchmark.py `
  --output-dir local-data/runs/captions
```

The default model list matches the final experiment. The runner performs a
preflight, verifies image hashes, resumes completed rows, and stores raw local
responses privately.

## New local judge passes

```powershell
python scripts/benchmark/run_local_judge_benchmark.py `
  --input local-data/runs/captions/benchmark.csv `
  --output-dir local-data/runs/judges/qwen38-pass-1 `
  --pass-id qwen38-pass-1 --seed 101

python scripts/benchmark/run_local_judge_benchmark.py `
  --input local-data/runs/captions/benchmark.csv `
  --output-dir local-data/runs/judges/qwen38-pass-2 `
  --pass-id qwen38-pass-2 --seed 202
```

For Flash-Next, start the local Strata vision endpoint and run the equivalent
`run_strata_judge_benchmark.py` commands with seeds 303 and 404. The server must
report both `loaded` and `images` as true at `/health`.

## Publishing a completed private run

The exporter is intentionally tied to the frozen final-run layout. It validates
row counts, model counts, statuses, and key coverage before removing private
image metadata:

```powershell
python scripts/analysis/export_public_results.py
python scripts/analysis/aggregate_final_results.py
```

Run a secret scan and inspect `git status` before every public release. Never
commit `local-data/`, `.env` files, credentials, images, source tables, or raw
service responses.
