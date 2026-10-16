# Methodology

## Research question

How do local vision-language models trade off factual caption quality,
instruction compliance, runtime, VRAM, and projected electricity cost when
labeling a large image collection?

The target is a compact descriptive label rather than a class name. The frozen
caption prompt requires exactly eight unique, factual, comma-separated phrases
on one line. Outputs are evaluated as generated and are never silently fixed.

## Sample

The experiment uses 120 visually reviewed images, balanced across six sampling
strata with 20 images each. Unsplash metadata was used only to form candidate
pools, not as reference truth or judge input. The private set was checked for
duplicate IDs, duplicate files, decoding errors, and stratum imbalance.

## Caption models

Each image was processed once by each model with seed 42, temperature 0, a
4,096-token context, and a 256-token output limit:

- `qwen3.8:27b-q4_K_M`
- `gemma4:12b-it-q8_0`
- `gemma3:4b-it-q8_0`
- `qwen3-vl:2b-instruct-q8_0`

All runs used the same files, order-independent stable IDs, and frozen prompt.
Raw local responses were retained privately. The public caption CSV preserves
parsed outputs and measurement fields.

## Systems measurements

Each request records wall time, token counts, generation speed, GPU memory,
utilization, temperature, clocks, sampled board power, estimated GPU energy,
system CPU utilization, physical-memory use, and observed Ollama working set.
Unavailable sensors remain blank with an explicit status.

GPU energy is the integral of sampled GPU board power during each request. It
does not include laptop idle power, CPU/system energy, display, cooling,
charging loss, or hardware depreciation. Projections to 100,000 images assume
serial processing at the observed request-level mean and are not deployment
quotes.

## Blind judging

Judges receive only the image, anonymous item ID, candidate caption, and frozen
rubric. They do not receive the caption-model identity. Each caption has four
ratings:

- two Qwen3.8 27B Q4_K_M passes, seeds 101 and 202;
- two Qwen3.8-Flash-Next IQ3_XXS passes, seeds 303 and 404.

Each pass scores accuracy, coverage, and clarity from 1 to 5 and records a
binary usable decision. The same-family judge design is a limitation: repeated
passes measure stability, but Qwen3.8 variants are not fully independent judges
and may favor Qwen-family captions.

The Qwen3.8 27B judge violated the requested 0–1 confidence scale in 957 of 960
records, usually returning a 1–5 value. These confidence values are excluded
from confidence averages. Score, usability, and rationale fields remain in the
analysis because they passed their respective validation ranges. The current
runner now rejects and retries future out-of-range confidence responses.

## Outcomes and statistics

For each caption:

- dimension score: mean of four judge ratings;
- quality: unweighted mean of accuracy, coverage, and clarity;
- majority usable: at least three of four usable votes;
- final compliant usable: majority usable and deterministic validation of the
  exact eight-phrase format.

Model means treat images as the unit of analysis. Quality confidence intervals
use 5,000 image-level bootstrap resamples with seed 20261013. Pairwise
differences use paired image resampling because every model captioned the same
images. A 95% paired interval excluding zero is reported as distinguishable in
this sample; it is not a universal superiority claim.

## Human comparison

No matched human-caption timing, wage, or quality baseline was collected.
Accordingly, the study compares local models with one another and does not
claim that they replace human annotators.
