# Result files

`raw/captions.csv` contains 480 sanitized caption-generation records. The four
files under `raw/judgments/` contain 480 blinded judgments each.

`analysis/` contains deterministic derived tables:

- `model_ranking.csv`: primary quality and usability results;
- `pairwise_model_differences.csv`: paired bootstrap comparisons;
- `category_scores.csv`: sampling-stratum summaries;
- `judge_agreement.csv`: repeatability and cross-judge agreement;
- `judge_pass_summary.csv`: judge-level score and systems summaries;
- `generation_efficiency.csv`: speed, VRAM, energy, and 100k-image projections;
- `caption_level_consensus.csv`: one consensus record per caption;
- `analysis_summary.json`: machine-readable integrity and headline results;
- `vlm_caption_benchmark_results.xlsx`: convenience report generated from the
  same analysis tables.

The CSV and JSON files are canonical. The Excel file is provided for convenient
review. Input images and Unsplash Dataset records are not redistributed.
