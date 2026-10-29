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

## Photograph embedded in the study PDF

`paper/main-study.pdf` reproduces one benchmark photograph, anonymous ID
`image_005.jpg`, in Figure 3 on page 5. Although the standalone source image is
not committed, the photograph is distributed as part of the downloadable PDF.
The repository therefore must not be described as excluding all image content.

The paper's statement that the example image is excluded from the repository
does not describe the embedded reproduction: only the standalone source file is
excluded. This documentation records that distinction; the paper itself has
not been revised.

Separate authorization to reproduce this particular photograph in a distributed
report has not been verified or documented in this repository. Attribution,
inclusion in a research paper, and a repository notice do not establish that
authorization. Its inclusion is not a claim that reproduction permission has
been established. See [Third-party content and rights](../THIRD_PARTY_NOTICES.md).

## Public experimental records

`results/raw/` contains model-generated outputs and benchmark measurements. One
caption row represents one anonymous image-model pair. Judgment files contain
one blinded assessment per caption for a single judge pass.

The public files retain anonymous IDs such as `image_001.jpg`, sampling strata,
model identifiers and digests, prompts/settings hashes, generated captions,
scores, failures, timing, tokens, and systems telemetry. They exclude source
URLs, licensed descriptions, image bytes, hashes, dimensions, and raw service
responses. These exclusions describe the structured experimental records, not
the embedded photograph in the study PDF described above.

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
