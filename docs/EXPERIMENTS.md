# Experiments (all numbers produced by the scripts below; raw outputs in `outputs/`)

> **Read first.** Labels are author-assigned or AI-assisted; no human annotation exists. Results describe behaviour on *this* dataset (552 pairs, 54 groups) and say nothing about real legal text, human judgement or pretrained transformers. Confidence intervals are group-cluster bootstrap and indicative only.

## Status of every planned comparison
| experiment | status |
|---|---|
| E0 collapse diagnosis (tiny random-init encoder) | executed, 3 seeds |
| E1/E2 grouped CV of 7 CPU models, en/hi/domain | executed (5-fold x 2 repeats) |
| E3 dataset ablation (original-48 / expanded / filtered / reviewed) | executed, 5 repeated grouped holdouts |
| E4 augmentation off/on | executed (same runs) |
| E5 sanity (identical / unrelated pairs) | executed |
| E6 rule-explainer vs labels | executed |
| Sentence-Transformer cosine | **NOT RUN** - Hub unreachable; run `python scripts/run_experiments.py` with `--st-path` (`lmp/models/st_baseline.py`; flag `--st-path`) |
| Pretrained XLM-R cross-encoder | **NOT RUN** - weights unreachable; `python scripts/train_crossenc.py --backbone xlm-roberta-base` (or `--local-model-path`). Implemented, unit-tested, smoke-trained only with a tiny random-init encoder |
| Transformer ablations (augmentation, language embedding, freezing) | not run (depend on the line above) |
| Annotation via external API / local LLM | implemented, **not executed** (no credentials/server) |
| Cross-language transfer, domain-holdout | `domain_holdout_split` implemented; not run |

## E0 - collapse diagnosis (tiny RANDOM-INIT encoder, 2 layers, d=64; grouped split; test n~104; mean +- sd over 3 seeds)
| design | collapse index (std pred / std target) | Spearman | macro-F1 | identical pair mean | shuffled pair mean |
|---|---|---|---|---|---|
| legacy design @ lr 2e-5 (as shipped) | **0.05 +- 0.00** | 0.03 +- 0.09 | 0.17 | 4.94 | 4.94 |
| legacy design @ lr 1e-3 | 0.51 +- 0.10 | 0.33 +- 0.09 | 0.38 | 5.16 | 5.81 |
| new `LMPCrossEncoder` | 0.47 +- 0.09 | 0.46 +- 0.09 | 0.39 | 5.36 | 5.83 |
Reading: the shipped optimisation setup reproduces your symptom in every seed (a near-constant output ~ the untrained head); it is an optimisation/selection problem rather than "the model cannot learn". The new design is modestly better than the legacy design at an adequate LR (Spearman difference 0.13 with sd 0.09, n=3 seeds: **not established**). A tiny scratch encoder has no semantic knowledge (identical ~ shuffled), so none of this transfers to XLM-R without running it. `outputs/collapse_diagnosis.json`.

## E1 - grouped CV, out-of-fold, mean of 2 repeats (consolidated dataset, n=552)
| model | Spearman [CI95] | RMSE | MAE | accuracy | macro-F1 | collapse idx |
|---|---|---|---|---|---|---|
| tfidf_raw | -0.45 [-0.53,-0.36] | 4.75 | 4.36 | 0.11 | 0.12 | 0.88 |
| tfidf_cal (guarded) | -0.05 [-0.07,0.04] | 2.64 | 2.29 | 0.35 | 0.17 | 0.02 |
| tfidf_cal_signfree | 0.45 [0.36,0.53] | 2.33 | 1.86 | 0.43 | 0.38 | 0.51 |
| lex_ridge | 0.39 [0.31,0.46] | 2.43 | 1.99 | 0.37 | 0.32 | 0.51 |
| rule_ridge | 0.19 [0.09,0.28] | 2.58 | 2.16 | 0.38 | 0.27 | 0.33 |
| lexrule_ridge | 0.47 [0.38,0.54] | 2.33 | 1.91 | 0.41 | 0.38 | 0.59 |
| **lexrule_gbm** | **0.58 [0.51,0.67]** | **2.13** | **1.49** | **0.66** | **0.63** | 0.68 |
Best model: `lexrule_gbm` (macro-F1 0.63; per-class F1 Not 0.73 / Partial 0.65 / Preserved 0.51; Preserved recall only 0.38). Interpretation limits: (i) rule features share design knowledge with the labels; the lexical-only models (0.39-0.45) show how much a dataset shortcut explains; (ii) tfidf_raw is *negatively* correlated with the labels by dataset construction; (iii) `tfidf_cal` (guarded) is a near-constant predictor.

## E2 - language (repeat 0, OOF)
English vs Hindi Spearman / macro-F1: lexrule_gbm en 0.63 / 0.63, hi 0.56 / 0.63; lexrule_ridge en 0.49 / hi 0.44; tfidf_cal_signfree en 0.48 / hi 0.41. Hindi is consistently a little lower in Spearman; with 276 pairs each, balanced classes, and parallel construction, this difference is small and **its cause (tokenisation, rule coverage, sentence structure) was not isolated**. Per-domain Spearman: `results.json`.

## E3/E4 - dataset and augmentation ablation (fixed grouped test per repeat; mean over 5 repeats; Spearman / macro-F1)
| train set (mean train n) | tfidf_cal_signfree | lex_ridge | lexrule_ridge | lexrule_gbm |
|---|---|---|---|---|
| original 48 (35) | 0.21 / 0.23 | 0.16 / 0.22 | 0.08 / 0.33 | 0.06 / 0.36 |
| expanded (410) | 0.45 / 0.35 | 0.40 / 0.29 | 0.50 / 0.36 | **0.61 / 0.61** |
| expanded + quality filter (409) | 0.45 / 0.35 | 0.40 / 0.29 | 0.50 / 0.36 | 0.61 / 0.61 |
| reviewed subset (160) | 0.45 / 0.32 | 0.35 / 0.23 | 0.42 / 0.43 | 0.57 / 0.55 |
| expanded + augmentation (614) | **-0.45** / 0.18 | -0.33 / 0.21 | 0.01 / 0.30 | 0.49 / 0.48 |
Conclusions supported: more varied data helps every model; the QA filter removed one row so it cannot matter; the smaller reviewed subset is not better than the full set; **augmentation is harmful here** (identity pairs contradict the dataset's "high overlap = changed meaning" regularity) and is therefore off by default.

## E5 - sanity checks (mean predicted score; ideal: identical ~10, unrelated ~1)
tfidf_raw 10.0 / 1.6; tfidf_cal 5.1 / 5.1; tfidf_cal_signfree 3.2 / 7.9; lex_ridge 3.4 / 7.6; rule_ridge 5.6 / 6.1; lexrule_ridge 3.9 / 8.4; **lexrule_gbm 3.0 / 7.3**. The trained lexical models fail this check (identical -> low, unrelated -> high): they learned the dataset artifact, not a notion of meaning. This is why `lmp.inference` sets identical texts to 10 by explicit rule and shows warnings.

## E6 - rule explainer vs dataset labels (n=552; not a validated system)
Changed-element micro-F1 0.42 (conf>=0), 0.43 (>=0.6), 0.40 (>=0.75). Flag rate by reference category: Not 0.81, Partial 0.71, **Preserved 0.82** (at conf>=0.6: 0.76 / 0.66 / 0.71). Per label (precision / recall): altered_numbers 0.61/0.89; changed_obligation 0.49/0.63; contradiction 0.24/0.59; altered_condition 0.23/0.37; changed_permission 0.44/0.26; omission 0.34/0.33. On the 216-pair "reviewed" subset micro-F1 is 0.48. So the rules are a *noisy hint generator* (good for numeric changes, poor for omission/scope/condition) and "a rule fired" says almost nothing about preservation. The prototype's strong-looking explainer results came from 48 author-written pairs and were circular.

## Error analysis
`outputs/error_analysis.md` (OOF predictions of lexrule_gbm; real rows; categories: obligation, permission, contradiction, exception, condition, actor, scope, time limit, number, reference, high-lexical-but-changed, low-lexical-but-preserved, per language). Example failure: `AIG-C001-HI-06` (reference 1, predicted 7.0: "may enter at any time" scored as near-preserved because most tokens overlap). Reference changes have only 2 examples - too few to analyse.

## Reproduce
```
pip install -r requirements.txt
python scripts/prepare_data.py            # build, QA, consolidate, variants, report
python scripts/inspect_dataset.py
python scripts/make_splits.py --seed 42
python scripts/annotate.py validate|compare|agreement|consolidate   # annotation workflow
python scripts/run_experiments.py         # E1-E6 + figures + error analysis (~1 min on CPU)
python scripts/diagnose_collapse.py --seeds 0   # then 1, 2 (each ~3-4 min on 1 CPU)
python scripts/train_crossenc.py --backbone xlm-roberta-base   # needs weights; add --tiny for an offline smoke run
python -m pytest -q
streamlit run app/streamlit_app.py
```
