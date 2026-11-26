# Explainable Multilingual Legal Meaning Preservation (PID-08)

A JUDGEBERT-inspired research prototype for assessing whether English and Hindi legal provisions retain their meaning after simplification. This project extends the research question to multilingual scoring, legal-element explanations and controlled robustness evaluation; it does not reproduce JUDGEBERT.

## Features

- Legal Meaning Preservation (LMP) score from 1 to 10 and independent three-class prediction: Preserved, Partially Preserved, Not Preserved.
- Frozen multilingual MiniLM embeddings combined with lexical, legal-rule and directional features; gradient boosting regression and classification heads.
- Potential legal changes with original and simplified expressions, and optional learned verifier prioritization without suppressing rule findings.
- Conservative deterministic equivalence handling, with raw predictions retained and disagreement warnings.
- English and Hindi Streamlit interface.

## Data and experimental design

The original dataset contains 552 synthetic pairs (48 author-written and 504 AI-generated), balanced across English and Hindi. The fixed grouped split contains 388 training, 52 validation and 112 test pairs. V1 adds 120 targeted contrastive training pairs and a separate 60-pair challenge set. V2 adds another 240 training pairs and a 120-pair challenge set. Matched resampling controls distinguish targeted augmentation from simply increasing training size. Challenge sets remain synthetic and have limited template diversity.

Human evaluation covers 100 pairs with 350 student judgments and 30 expert judgments. The held-out subsets contain 47 student-consensus pairs and 20 expert-rated pairs. Human evaluation does not constitute validation on authentic legal documents. Private annotations are not distributed.

## Selected model and results

The primary V1 targeted hybrid was selected by validation classification macro-F1, then MAE. Regression-derived categories and the independent classification head are distinct outputs.

| Evaluation | Spearman | MAE (1–10) | Classification head macro-F1 |
|---|---:|---:|---:|
| Validation (52) | 0.824 | 1.209 | 0.944 |
| Original test (112) | 0.614 | 1.477 | 0.739 |
| V1 synthetic challenge (60) | 0.890 | 1.141 | 0.944 |
| Held-out student consensus (47) | 0.625 | 1.682 | 0.589 |
| Held-out expert ratings (20) | 0.763 | 2.166 | 0.645 |

Human metrics use raw predictions. The separate 36-case synthetic behavioral audit yielded raw head macro-F1 0.679 and system macro-F1 0.738. These small diagnostic sets are not independent real-world validation. Bootstrap comparisons are exploratory.

### Explanation evaluation

| Method | Validation micro-F1 | Original test micro-F1 | Behavioral micro-F1 | Original test preserved-case false alarm rate |
|---|---:|---:|---:|---:|
| Enhanced rules | 0.408 | 0.441 | 0.894 | 0.759 |
| Learned verifier | 0.608 | 0.483 | 0.500 | 0.655 |
| Evidence filter | 0.528 | 0.449 | 0.732 | 0.517 |

The final interface retains all enhanced rule findings and uses the verifier only for prioritization (validation-selected threshold 0.65). Agreement is not proof of correctness. The prioritization interface is not claimed to improve detection F1. Evidence spans have not been independently human-validated.

## Run locally

Use Python 3.11 from the repository root. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The final model uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Download/cache it once if not already available:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

For an existing local installation with checkpoints, launch directly:

```powershell
python -m streamlit run scripts/final_hybrid_app.py
```

For a fresh checkout, the serialized checkpoints are excluded from Git. With the committed datasets available, rebuild the primary model and verifier:

```powershell
python scripts/train_final_hybrid.py
python scripts/improve_explainer.py
python scripts/finalize_explainer.py
python -m streamlit run scripts/final_hybrid_app.py
```

Expected primary checkpoint: `outputs/final_hybrid/selected.joblib` (targeted, hybrid). Optional verifier: `outputs/explainer_verifier/verifier.joblib`. Only load trusted joblib files. The app uses locally cached encoder weights. A clean-checkout rebuild has not yet been independently verified.

## Experiments and repository layout

| Path | Purpose |
|---|---|
| `data/` | Original synthetic data, grouped splits and augmentation datasets |
| `lmp/models/` | Baselines, XLM-R cross-encoder, hybrid model and equivalence handling |
| `lmp/explain/` | Legal-rule analysis and evidence extraction |
| `lmp/human_validation/` | Human-evaluation analysis |
| `scripts/final_ablation.py` | Baseline ablations and shortcut analysis |
| `scripts/build_contrastive_pilot.py`, `scripts/build_resampling_control.py` | V1 augmentation and matched control |
| `scripts/train_final_hybrid.py` | V1 model comparison and selection |
| `scripts/build_dataset_v2.py`, `scripts/train_hybrid_v2.py` | Extended augmentation experiments |
| `scripts/evaluate_final_human.py` | Private human-alignment evaluation |
| `scripts/evaluate_release.py` | Behavioral audit and paired group bootstrap |
| `scripts/improve_explainer.py`, `scripts/finalize_explainer.py` | Verifier and evidence-filter experiments |
| `scripts/final_hybrid_app.py` | Final demonstration interface |
| `outputs/` | Experimental results; private records and checkpoints excluded as configured |
| `docs/` | Earlier methodology and project documentation; some historical status statements may be outdated |

The repository also contains earlier experimental applications and models. Use the final hybrid application for the demonstration. Do not rebuild the original dataset unnecessarily when reproducing fixed-split results. Human-evaluation scripts require private inputs and cannot be fully reproduced from public aggregate results alone.

## Tests

```powershell
python -m pytest -q
python scripts/test_legal_equivalence.py
```

The last recorded full suite before final integration reported 133 passed and 1 skipped. The final application has executed five manual assessment smoke tests; all five system categories matched expectations, but several raw scores disagreed with the classification head and one faithful paraphrase produced an incorrect obligation finding. These are not an accuracy benchmark.

## Limitations and responsible use

- Synthetic data, provisional AI labels, limited templates and small human-evaluation subsets restrict generalization.
- Original data exhibits lexical-overlap and generation-strategy shortcuts; targeted augmentation improves controlled challenge performance but does not eliminate these biases.
- Hindi performance and complex paraphrase handling remain weaker in several evaluations.
- Raw scores and classification heads can disagree. Model and verifier probabilities are uncalibrated; the 0.60 model review threshold is provisional.
- Deterministic equivalence rules cover narrow patterns, not arbitrary semantic equivalence.
- The explanation engine has substantial false positives and missed changes; verifier agreement does not establish legal correctness or causal model reasoning.
- The prototype assesses sentence pairs, not complete legal documents, jurisdiction-specific interpretation or enforceability.

**Research prototype only. Not legal advice. Human review of complete legal context is required.**
