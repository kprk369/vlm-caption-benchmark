# Data handling and availability

## Private inputs

The experiment used 120 images selected from Unsplash Lite Dataset 1.4.1. The
applicable Dataset Terms prohibit publishing or redistributing licensed Dataset
content. Consequently, the following remain under the Git-ignored
`local-data/` directory and are not part of the repository:

- Unsplash Dataset tables and documentation;
- downloaded image files and contact sheets;
- Dataset rows, source descriptions, photo URLs, photographer fields, and image
  hashes;
- API credentials, download logs, pilot experiments, and raw API responses.

Users must obtain the Unsplash Dataset independently and accept its applicable
terms. This repository does not grant rights to any input image or Dataset row.

## Public experimental records

`results/raw/` contains model-generated outputs and benchmark measurements. One
caption row represents one anonymous image-model pair. Judgment files contain
one blinded assessment per caption for a single judge pass.

The public files retain anonymous IDs such as `image_001.jpg`, sampling strata,
model identifiers and digests, prompts/settings hashes, generated captions,
scores, failures, timing, tokens, and systems telemetry. They exclude source
URLs, licensed descriptions, image bytes, hashes, dimensions, and raw service
responses.

## Sampling strata

The sample contains 20 images in each stratum:

- objects and technology;
- people and actions;
- indoor scenes;
- outdoor scenes;
- animals;
- complex or text-heavy scenes.

These strata support balanced sampling. They are not mutually exclusive
ground-truth class labels.

## Data lineage

The canonical chain is:

```text
private images -> caption generation -> blinded judgments
-> sanitized raw CSV files -> deterministic aggregate tables
```

`scripts/analysis/export_public_results.py` validates the private final run and
creates the sanitized public bundle. `scripts/analysis/aggregate_final_results.py`
rebuilds all derived CSV tables from that bundle.
