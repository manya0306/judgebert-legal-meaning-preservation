# Data sources, provenance and licensing

## What is bundled
`data/synthetic/synthetic_pairs.csv` - 48 pairs (24 EN, 24 HI; 3 domains) **written by the project author for software testing**.
They are not extracted from real legal documents and no real-world source is claimed. The `lmp_score` and `error_labels` are the author's
own assignments (`annotation_status = author_assigned_score_not_human_annotation`). Every row has `data_origin = synthetic`.
**This dataset is far too small and too artificial for research-grade training or for conclusions about model quality.**

## Candidate sources for authentic data (NOT yet collected, NOT verified in this session)
Each entry must be checked for current licence/terms before any text is copied. Record the check in the `source_provenance` and `license_note` columns.
| Candidate | Languages | Why it could help | What must be verified first |
|---|---|---|---|
| Indian central/state legislation published by the Government of India (e.g. India Code portal) | EN, HI | Official bilingual statutory text | Government open-data / copyright terms for reuse and redistribution |
| Official Hindi/English model agreements or plain-language guides from government departments or regulators | EN, HI | Existing original-vs-plain-language pairs would be natural simplification data | Per-document reuse terms |
| Consumer-facing explanations published alongside insurance/tenancy regulation | EN, HI | Natural domain coverage (matches the 3 demo domains) | Licence; whether the explanation is an actual simplification of a specific clause |
| Court judgments with official translations | EN, HI | Rich legal language | Terms of the court/publisher; sensitivity of personal data in judgments |
The JUDGEBERT dataset FrJUDGE is French and licensed CC-BY-NC-SA 4.0 per the official repository README; it is not used here.

## How to build the real dataset
1. Choose domain(s) and collect original clauses with documented provenance; 2. obtain or write simplifications (human or system-generated; record which);
3. create annotation sheets (`annotation_tools.py sheets`); 4. annotate independently (>=2 annotators); 5. compute agreement; 6. consolidate;
7. split by `source_doc_id`. Never mix synthetic and authentic rows without the `data_origin` column.
