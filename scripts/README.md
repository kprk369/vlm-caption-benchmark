# Scripts

Only reusable scripts required for dataset construction, benchmarking,
publication export, and analysis belong here.

- `dataset/build_final_dataset.py`: constructs and validates the private,
  balanced 120-image sample from locally obtained Unsplash Dataset files.
- `benchmark/run_caption_benchmark.py`: resumable Ollama caption generation
  with format validation and systems telemetry.
- `benchmark/run_local_judge_benchmark.py`: resumable blinded Ollama judging
  with strict response validation and telemetry.
- `benchmark/run_strata_judge_benchmark.py`: equivalent blinded judging against
  a local OpenAI-compatible Strata endpoint.
- `analysis/export_public_results.py`: validates the private final experiment
  and writes the sanitized files under `results/raw/`.
- `analysis/aggregate_final_results.py`: deterministically creates the analysis
  tables under `results/analysis/`.

Pilot interfaces, cloud experiments, installation helpers, machine-specific
launchers, and migration scripts are intentionally excluded.
