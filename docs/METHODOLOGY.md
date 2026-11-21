# Methodology

## Task and relation to JUDGEBERT
Given (original legal sentence, simplified sentence, language in {en, hi}) predict a Legal Meaning Preservation score in [1,10], a 3-way category (Not Preserved < 4 <= Partially Preserved < 7.5 <= Preserved), and show potentially altered legal elements.
The project is **inspired by** the JUDGEBERT line of work on legal meaning preservation. It does **not** reproduce JUDGEBERT's architecture, training data or results; no claim of equivalence or replication is made.

## Data (see DATASET_DOCUMENTATION.md)
552 pairs: 48 author-written (original prototype) + 504 AI-generated (Claude, in-session), labelled by AI-assisted passes. No human annotation, no real-world statute text.

## Splitting and leakage control
Unit = group (all variants of a concept, its English/Hindi counterparts, near-duplicate groups merged by connected components over char-n-gram TF-IDF cosine >= 0.80). Domain-stratified grouped split and **repeated grouped 5-fold CV (2 repeats)**; `assert_no_leakage` checks group, pair_id, concept and identical-original overlap. Test data is never used for threshold, calibration or hyperparameter choice (all calibration is fit on the training portion).

## Models
| id | description | trained? |
|---|---|---|
| tfidf_raw | 1+9*cosine of word TF-IDF | no labels |
| tfidf_cal / tfidf_cal_signfree | linear calibration on train (guarded / sign-free) | yes (train only) |
| lex_ridge | ridge on lexical pair features | yes |
| rule_ridge | ridge on analyzer features | yes (circularity caveat) |
| lexrule_ridge, lexrule_gbm | both feature groups; ridge / gradient boosting | yes |
| **LMPCrossEncoder** (`lmp/models/crossenc.py`) | joint-sequence multilingual encoder; pooled [CLS] + mean + segment-difference; linear regression head on z-scored target + 3-way classification head; Huber + class-weighted CE; differential LR; collapse index; rank-based checkpoint selection | implemented and unit-tested; **trained only as a tiny random-init diagnostic**; pretrained XLM-R **NOT RUN** (weights unreachable) |
| sentence-transformer cosine | existing implementation | **NOT RUN** (Hub unreachable) |

Category for regressors = thresholds on the predicted score (same thresholds for all models).

## Why this objective
Categories are derived from scores, so regression and classification are consistent by construction; the multi-task loss lets the classification head regularise the regression head. Ordinal regression (CORAL) was considered; with ~550 pairs and 3 classes the added parameters were not justified - future work.

## Explainability (`lmp/explain`)
Layer 1 **textual diff** (token alignment with offsets; surface only). Layer 2 **rule findings**: modals/obligation-permission, negation, conditions/exceptions, numbers/dates/deadlines, quantifiers, actors, conjunction/disjunction flips, comparator and bound flips, statute references, party order - English and Devanagari lexicons incl. mixed-language loanwords. Every finding has a heuristic confidence mapped to an uncertainty label and is tagged `layer=rule_based`. A neural-model-derived layer is represented in the data structure (`model_signal`) but **no attribution method is implemented or claimed**; rule findings are not a faithful explanation of any neural score. Nothing is "human-validated".

## Annotation pipeline (see ANNOTATION_GUIDELINES.md)
Providers are pluggable (`generation_intent`, `imported`, `heuristic`, `anthropic` API, `openai_compatible`). The API providers fail loudly without credentials/servers and were **not executed**. Disagreement detection (category differs, |score diff| >= 2, label Jaccard < 0.5) -> adjudication sheet -> consolidation with status ladder that never produces "human" unless a human adjudicator record exists.

## Evaluation protocol
Regression: MAE, RMSE, Pearson, Spearman, bias, collapse index (std(pred)/std(target)). Classification: accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix. 95% CIs by group-cluster bootstrap (300 resamples; indicative). Per-language and per-domain breakdowns, sanity checks (identical / unrelated pairs), ablations on a *fixed* grouped test set shared by all variants.
