# Implementation status

_Last update: end of session. All stages A-H attempted; items marked BLOCKED/NOT DONE need external resources or further work._

## Environment facts (measured)
1 CPU core, ~3 GB RAM, Python 3.12, no GPU. Hugging Face Hub unreachable (HTTP 000). No ANTHROPIC_API_KEY. torch/transformers/streamlit/pytest installed from PyPI.

## Stage status
| stage | state | evidence |
|---|---|---|
| A audit | DONE | docs/PROJECT_AUDIT.md; baseline 28 tests passed |
| B data + methodology | DONE except size target | 552 pairs (target 1,000-2,000 NOT met); QA; grouped splits/CV; isolated augmentation; annotation pipeline; blind pass 2 (168); 11 adjudications; agreement tools |
| C models | DONE (code) / BLOCKED (pretrained runs) | crossenc + trainer + tiny diagnosis; sklearn scorers; ST baseline implemented (NOT RUN) |
| D explainability | DONE (heuristic) | extra_rules, spans, highlight; E6 shows weak precision; no model attribution |
| E experiments | DONE for feasible set | outputs/results.{json,md}, error_analysis.md, figures/, collapse_diagnosis.json |
| F app | DONE | app/streamlit_app.py, AppTest tests |
| G QA | DONE | 79 tests pass (28 original, of which 2 had fixtures/ports adjusted - see PROJECT_AUDIT) |
| H report | DONE | final message + docs |

## Files changed / added
Modified: lmp/data/{schema,io,splits,augment,agreement}.py, lmp/evaluation/metrics.py, lmp/explain/analyzer.py, lmp/models/tiny (new), configs/default.json, requirements.txt, README.md, app/streamlit_app.py, run_xlmr_smoke.py (deprecation note), tests/{test_splits,test_app,test_integration}.py.
New: lmp/data/{build,qa}.py, lmp/annotation/*, lmp/models/{crossenc,train_crossenc,sk_models,tiny}.py, lmp/explain/{extra_rules,spans,highlight}.py, lmp/evaluation/framework.py, lmp/inference.py, scripts/{prepare_data,annotate,inspect_dataset,make_splits,run_experiments,diagnose_collapse,train_crossenc}.py, data/generated/concepts_01-03.py, data/expanded/*, data/annotation/*, docs/*, tests/test_{dataset,annotation,models,explain_inference}_v2.py.

## Known issues
- Best CPU model fails identical/unrelated sanity (E5); app sets identical -> 10 by rule.
- Explainer precision is low (E6).
- Hindi unreviewed by a native speaker.
- Legacy modules (LMPRegressor, old experiments.py) retained deliberately.

## Next actions (in order)
1. On a machine with Hub access: `python scripts/train_crossenc.py --backbone xlm-roberta-base`, then `python scripts/run_experiments.py --st-path paraphrase-multilingual-MiniLM-L12-v2`; record results in docs/EXPERIMENTS.md under new headings (do not overwrite E1-E6).
2. Recruit two independent annotators; annotate >=300 pairs with the guideline schema; run `scripts/annotate.py validate/compare/agreement`.
3. Native-speaker review of the Hindi sentences; fix and re-run `prepare_data.py`.
4. Add more concepts (`concepts_04.py`...) toward 1,000+ pairs, annotate, re-run.
