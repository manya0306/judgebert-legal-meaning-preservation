# Explainable Multilingual Legal Meaning Preservation (PID-08) - JUDGEBERT-inspired research prototype

Given an original legal sentence and its simplification (English or Hindi), the system estimates a **Legal Meaning Preservation (LMP) score (1-10)**, a category (**Preserved / Partially Preserved / Not Preserved**), and lists **potentially altered legal elements** with highlighted evidence spans and uncertainty labels.
Inspired by JUDGEBERT; it does **not** reproduce JUDGEBERT. Not legal advice.

## Honest status in one table
| component | state |
|---|---|
| Dataset: 552 pairs (48 author-written + 504 AI-generated), en/hi balanced, 16 domains, 54 groups | built, QA-checked, **no human annotation, no real-world text** |
| AI-assisted annotation pipeline (providers, blind 2nd pass on 168 pairs, 11 adjudications, agreement tools) | executed with Claude *in-session*; API/local-LLM providers implemented, **not executed** |
| Leakage-safe grouped splits + repeated grouped CV | implemented, tested, used |
| CPU models (TF-IDF, lexical/rule ridge, gradient boosting) | trained and evaluated (best: lexrule_gbm, OOF Spearman 0.58, macro-F1 0.63) |
| Multi-task cross-encoder (`LMPCrossEncoder`) | implemented + tested; **pretrained XLM-R NOT trained here (weights unreachable)** |
| Sentence-Transformer baseline | implemented; **NOT RUN** |
| Explainability: diff + rules + spans + uncertainty | implemented; rule micro-F1 vs labels 0.42 (heuristic, unvalidated) |
| Streamlit app on the real inference pipeline (fallback clearly labelled) | working, tested |
| Tests | 79 passing |
Details: `docs/` (PROJECT_AUDIT, METHODOLOGY, ANNOTATION_GUIDELINES, DATASET_DOCUMENTATION, EXPERIMENTS, LIMITATIONS, IMPLEMENTATION_STATUS).

## Quick start (Windows PowerShell or Linux; run from the project root)
```
pip install -r requirements.txt
python scripts/prepare_data.py                 # rebuild dataset, QA, consolidation
python scripts/run_experiments.py              # E1-E6, figures, error analysis -> outputs/
python -m pytest -q                            # full test-suite
streamlit run app/streamlit_app.py             # demo (uses a transformer checkpoint if present, else the labelled fallback)
```
Transformer (needs Hub access once, ~1.1 GB; `freeze_layers=6` reduces CPU cost; training time on your machine was NOT measured here):
```
python scripts/train_crossenc.py --backbone xlm-roberta-base            # or: --local-model-path D:\models\xlm-roberta-base
python scripts/train_crossenc.py --tiny --epochs 5                      # offline smoke test only (random init)
python scripts/run_experiments.py --st-path paraphrase-multilingual-MiniLM-L12-v2   # adds the Sentence-Transformer baseline
```
The old checkpoints from the first prototype are incompatible (legacy architecture); `load_checkpoint` says so and `train_crossenc.py` retrains.

## Layout
`lmp/data` (schema, build, qa, splits, augment, agreement) - `lmp/annotation` (providers, records, pipeline, evidence) - `lmp/models` (crossenc, train_crossenc, sk_models, tiny, legacy modules) - `lmp/explain` (lexicon, analyzer, extra_rules, spans, highlight) - `lmp/evaluation` (metrics, framework, legacy experiments) - `lmp/inference.py` - `scripts/` - `app/` - `data/{synthetic,generated,expanded,annotation}` - `outputs/`.

## Extending the data
Add `data/generated/concepts_NN.py` (format in `concepts_01.py`) -> `python scripts/prepare_data.py` -> annotate new rows (`scripts/annotate.py`) -> rerun experiments. Real human annotations go into `data/annotation/human/` using the schema in `docs/ANNOTATION_GUIDELINES.md`; no code change is needed to incorporate them.
