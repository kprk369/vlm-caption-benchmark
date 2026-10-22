# Local VLM Image-Caption Benchmark

This repository evaluates the quality and systems cost of using local
vision-language models to generate detailed image labels. Each model captioned
the same 120-image sample using one frozen prompt. Four blinded VLM-judge passes
then scored every caption for accuracy, coverage, clarity, and usability.

## Main study

[**Image Captioning on a Laptop: Comparing the Quality and Efficiency of Local AI Models**](paper/main-study.pdf)

The main study includes the methodology, results, prompts, and supplementary figures.
This private PDF backup contains an example photograph; standalone image files remain excluded.

## Final experiment 

- 120 images across six balanced sampling strata
- 4 local caption models and 480 generated captions
- 4 blind judge passes per caption and 1,920 judgments
- request-level latency, tokens, VRAM, RAM, power, temperature, and estimated
  GPU energy recorded on an RTX 5090 Laptop GPU
- paired image-level comparisons with 5,000 bootstrap resamples

| Rank | Caption model | Quality (1–5) | Format-compliant usable | Seconds/image | Peak VRAM (MiB) |
|---:|---|---:|---:|---:|---:|
| 1 | Qwen3.8 27B Q4_K_M | 4.888 | 98.3% | 3.77 | 17,216 |
| 2 | Gemma 4 12B Q8_0 | 4.791 | 97.5% | 1.48 | 13,416 |
| 3 | Qwen3-VL 2B Q8_0 | 4.312 | 0.0% | 0.78 | 4,100 |
| 4 | Gemma 3 4B Q8_0 | 4.269 | 65.8% | 0.99 | 5,556 |

Qwen3.8 ranked first, while Gemma 4 provided the strongest quality-efficiency
tradeoff. The Qwen3-VL score should be read alongside its 0% strict-format
compliance: its captions were often visually useful but did not follow the
required eight-phrase output contract.

## Repository layout

```text
docs/                  Method, data handling, and reproduction instructions
prompts/               Frozen caption-generation and judging prompts
results/raw/            Sanitized caption and judgment records
results/analysis/       Derived tables and the Excel report
scripts/benchmark/      Resumable local benchmark runners
scripts/analysis/       Public export and deterministic aggregation
scripts/dataset/        Private dataset-selection utility
local-data/             Git-ignored licensed images and private run state
```

## Reproduce the analysis

The included sanitized result records are sufficient to reproduce every CSV
analysis table without the licensed images:

```powershell
python -m pip install -r requirements.txt
python scripts/analysis/aggregate_final_results.py
```

See [Reproducibility](docs/REPRODUCIBILITY.md) for a new end-to-end run and
[Methodology](docs/METHODOLOGY.md) for the scoring protocol.

## Data availability

Unsplash images, Dataset rows, source descriptions, URLs, and image hashes are
not included. The repository publishes sanitized experimental outputs with
anonymous benchmark image IDs. See [Data handling](docs/DATA.md).

## Interpretation limits

Both judge models are Qwen3.8 variants. Their agreement is useful evidence, but
it is not independent human ground truth and may favor Qwen-family captions.
No human-cost claim is made because comparable human timing and wage data were
not collected. Results apply to this sample, prompt, quantizations, software,
and hardware—not to all image-captioning workloads.
