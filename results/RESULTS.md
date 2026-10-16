# Results summary

Qwen3.8 27B Q4_K_M achieved the highest mean quality score, 4.888 out of 5,
with a 95% image-bootstrap interval of 4.858–4.916. Gemma 4 12B Q8_0 followed
at 4.791 (4.747–4.832). Their paired difference was 0.097 points (95% CI
0.049–0.149).

Gemma 4 produced captions about 2.5 times faster than Qwen3.8 and used 13,416
MiB peak VRAM instead of 17,216 MiB. Its measured request-level GPU-energy
projection was 1.64 kWh per 100,000 images, compared with 9.20 kWh for Qwen3.8.
These projections exclude the rest of the laptop and all non-electricity costs.

Qwen3-VL 2B was fastest at 0.78 seconds per image, but none of its outputs
passed the exact eight-phrase comma-separated format validator. Gemma 3 4B was
also fast, but had the lowest mean quality and a 65.8% final compliant-usable
rate.

The ranking is based on AI-judge consensus, not human ground truth. Both judge
models are Qwen3.8 variants, creating a plausible same-family preference for
Qwen3.8 captions. A matched human review is required before drawing conclusions
about replacing human annotation.
