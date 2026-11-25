# Human-validation analysis

Entry point: `python scripts/analyze_human_validation.py` (statistics only; no models, no downloads, no training).

Input (default `outputs/human_validation_v2/completed/`): `Student_Annotations_Final.xlsx`, `Lawyer_Annotations_Final.xlsx`, `PRIVATE_Reference_Key.xlsx`.
Output (default `outputs/human_validation_analysis/`, git-ignored, never overwritten unless `--overwrite`): validation report, correction log,
`individual_annotations.csv`, `pair_consensus.csv`, `metrics/*.csv`, `figures/*.png|pdf`, `human_validation_report.md`, `run_metadata.json`.

Critical validation errors stop all dependent analyses (exit code 2; only the validation report, correction log and metadata are written).

Optional model evaluation (kept separate; opt-in):
* `--predictions file.csv` with `pair_id` (HV2 or source id), `model_name`, `score`, and for held-out claims `checkpoint_id`, `training_splits` (e.g. `train;val`).
* `--run-models lexical_similarity,rule_features_ridge,lexrule_gbm,sentence_transformer,xlmr_crossencoder` - classical baselines are fitted in memory on
  `data/splits/train.csv` only (as the repository's fallback predictor does); Sentence-Transformer weights are loaded offline from the local cache;
  `--xlmr-checkpoint DIR` loads an existing checkpoint (optionally `--xlmr-provenance json` with `{"training_splits": [...], "checkpoint_id": "..."}`).
  Missing models are reported `NOT RUN` with the reason; undeclared training provenance is `UNVERIFIED` (no held-out claim).

Conventions: integer rubric 1-3 / 4-6 / 7-10; continuous consensus mapped with boundaries 3.5 / 6.5; reference categories reported both with the repository
thresholds (4 / 7.5) and harmonised (3.5 / 6.5), never mixed; bias = comparator - standard; group-aware bootstrap over `group_id`, fixed seed.
