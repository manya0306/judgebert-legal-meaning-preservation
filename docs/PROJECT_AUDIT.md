# Project audit (initial state -> corrections)

Audited: every file in the uploaded `lmp-judgebert-final-upload.zip` (Python package `lmp/`, `data/`, `scripts/`, `app/`, `tests/`, `configs/`, `outputs/`, docs).
Baseline measured before any change: **28 tests passed** (76 s). Everything below was verified by reading code/outputs or by running experiments in this repo; items I could not verify are marked.

## What already worked (kept)
Config-driven package layout; `preprocess` and `langid`; the rule lexicon and `explain.analyzer` (extended, not replaced); TF-IDF baseline; metric helpers; grouped-by-document splitting idea; the 28 original tests (all still pass; two fixtures/ports are listed below).

## Problems found and what was done

| # | Finding (evidence) | Severity | Correction |
|---|---|---|---|
| 1 | Dataset = **48 author-written pairs** (24 en / 24 hi; 3 domains x 16). A test split of 8 pairs cannot support any metric. | critical | 504 AI-generated pairs added (tier `ai_generated`), 552 total; grouped CV instead of a single tiny test set; limitations documented. Still no human or external data. |
| 2 | **Prediction collapse**: the shipped run (`outputs/results.json`) predicted ~5.2 for identical, unrelated and subtle pairs (the output of an untrained sigmoid head, 1+9*sigmoid(0)=5.5). Config: 32 train pairs (+aug), 10 epochs, lr 2e-5 for head and encoder, patience 3, checkpoint by validation RMSE on 8 pairs. `history.json` was not saved, so the cause is *inferred*, then tested (below). | critical | New `LMPCrossEncoder` + trainer; controlled diagnosis `scripts/diagnose_collapse.py` (EXPERIMENTS.md, E0). |
| 3 | **Why**: (a) head and encoder share one tiny LR -> head barely moves in ~80 steps; (b) RMSE-based checkpoint selection is won by a constant predictor on a small validation set; (c) sigmoid output compresses the target; (d) mean-pooled single sequence. Diagnosis on a tiny random-init encoder: shipped design at shipped LR collapses in **3/3 seeds** (collapse index 0.05); same design at a higher LR does not (0.51). | high | Linear heads on z-scored target, separate head/encoder LR, rank-based checkpoint selection with a collapse penalty, collapse index logged each epoch. Not verified on pretrained XLM-R (blocked). |
| 4 | **Augmentation** added identity pairs (score 10) and unrelated pairs (score 1) for ~100% of train size; trivially separable extremes; rows not structurally traceable. | high | Rewritten as isolated, capped, marked, optional, **off by default**. Ablation E4: it *hurts* (Spearman TF-IDF +0.45 -> -0.45; GBM 0.61 -> 0.49). |
| 5 | **Circular feature baseline**: the feature regressor used the analyzer's features; both were designed around the same examples as the labels. High scores from it are not independent semantic validation. | high | Lexical-only vs rule-only vs combined ablation (E1); rule-explainer evaluated on larger data (E6: micro-F1 0.42; flags 82% of *Preserved* pairs). Documented as non-independent. |
| 6 | `calibrate_linear` falls back to the training mean when the fitted slope is negative -> "calibrated TF-IDF" silently became a constant predictor on the new data (Spearman -0.05, collapse index 0.02). | medium | Added `allow_negative`; both variants reported; sign-free variant exploits dataset construction (see LIMITATIONS). |
| 7 | **Explainer gaps** found by test: no and/or flips, comparator flips (lower/higher), section/statute references, actor swaps; no character offsets or textual diff. | medium | `explain/extra_rules.py`, `explain/spans.py`, `explain/highlight.py`; findings carry spans, `layer`, `uncertainty`; "no finding" explicitly does not mean "preserved". |
| 8 | Splitting by document only; no near-duplicate merging; no CV; no automated leakage assertions. | high | `data/splits.py`: group merge by near-duplicate connected components, domain-stratified grouped split, repeated grouped CV, `assert_no_leakage`. Bug found by a new test during this work: duplicate originals were dropped before comparison, so identical originals in different groups were not merged (fixed). |
| 9 | No provenance, no QA, no annotation infrastructure, no agreement tooling. | high | schema v2 (tiers, label_source, status, qa_flags), `data/qa.py`, `lmp/annotation/*`, weighted kappa and Krippendorff alpha (cross-checked to 6 decimals against the `krippendorff` package). |
| 10 | Old checkpoints/scripts use the legacy `LMPRegressor`. | medium | Kept for the diagnosis experiment; `load_checkpoint` refuses legacy checkpoints with a retraining message; `run_xlmr_smoke.py` marked deprecated. |
| 11 | App: named-example selectbox demo. | medium | Rewritten on `lmp.inference.LMPPredictor`; backend always shown; real dataset rows as examples; `tests/test_app.py` ported to the new UI (same assertions) + new Hindi test. |
| 12 | Old integration test assumed the 48-pair default dataset. | low | Fixture now points at that dataset explicitly (assertions unchanged). |
| 13 | One tokenizer bug in my own tiny test backbone: `token_type_ids` were not returned, silently disabling segment pooling in the first diagnosis run. | self-found | Fixed; diagnosis re-run; only re-run numbers are reported. |

## Environment constraints (measured)
1 CPU, ~3 GB RAM, no GPU; Hugging Face Hub unreachable (HTTP 000); no API key. Consequently no pretrained transformer was trained and no external API was called.
