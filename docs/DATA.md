# Data handling and availability

## Private inputs

The experiment used 120 images selected from Unsplash Lite Dataset 1.4.1.
The [Unsplash Dataset Terms](https://github.com/unsplash/datasets/blob/master/TERMS.md)
restrict publication and redistribution of Licensed Data without Unsplash's
written permission (Section 3). The version accepted when the Dataset was
downloaded governs that use (Section 1.B); the linked current terms are a
reference, not evidence of a separate permission grant.

The following source assets remain under Git-ignored local directories and
are not distributed as standalone repository files:

- Unsplash Dataset tables and documentation;
- standalone downloaded image files and contact sheets;
- Dataset rows, source descriptions, photo URLs, photographer fields, and image
  hashes;
- API credentials, download logs, pilot experiments, and raw API responses.

Users must obtain the Unsplash Dataset independently and accept its applicable
terms. This repository does not grant rights to any input image or Dataset row.

## Current study PDF and caption examples

The current `paper/main-study.pdf` does not reproduce benchmark photographs.
The former example photograph, anonymous ID `image_005.jpg`, has been removed.
The four unedited captions generated for that image remain as text, together
with the analysis charts and tables. Readers cannot independently verify the
captions' visual accuracy from the paper alone because the source image is
withheld. No caption outputs or benchmark scores were changed by this removal.

Earlier PDF versions contained the photograph. This replacement updates the
current branch, not Git history: those historical PDFs may still be retrieved
from earlier commits. No claim is made that all historical copies have been
removed. See [Third-party content and rights](../THIRD_PARTY_NOTICES.md).

## Public experimental records

`results/raw/` contains model-generated outputs and benchmark measurements. One
caption row represents one anonymous image-model pair. Judgment files contain
one blinded assessment per caption for a single judge pass.

The public files retain anonymous IDs such as `image_001.jpg`, sampling strata,
model identifiers and digests, prompts/settings hashes, generated captions,
scores, failures, timing, tokens, and systems telemetry. They exclude source
URLs, licensed descriptions, image bytes, hashes, dimensions, and raw service
responses. The current study PDF also excludes source photographs, as described
above. Historical PDF versions are a separate exception noted above.

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
