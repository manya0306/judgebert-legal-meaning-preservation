# Annotation guidelines (for human annotators; the same rubric was applied by the AI passes)

Judge **only** whether the SIMPLIFIED sentence keeps the legal meaning (legal effect) of the ORIGINAL. Do not reward fluency, brevity or style. Do not give legal advice. Annotate independently; do not look at other annotators' records or at model scores.

## Score (1-10; halves allowed) and category
| score | category | meaning |
|---|---|---|
| 9-10 | Preserved | same legal effect, stylistic differences only |
| 7.5-8.5 | Preserved | same legal effect; trivial nuance differs |
| 4-7 | Partially Preserved | core rule survives; a legally relevant element (condition, exception, number, scope, party, formality) is lost or altered |
| 1-3.5 | Not Preserved | legal effect reversed, removed or materially changed |
Thresholds are configurable (`configs/default.json`): Not < 4, Preserved >= 7.5.

## Changed-element labels (any that apply; empty if none)
omission, contradiction, changed_obligation (duty weakened/strengthened: shall<->may/should), changed_permission (permission<->obligation/prohibition), altered_condition (if/unless/and/or logic), missing_exception, scope_change (who/what/how many covered; quantifiers; "any part"), altered_dates (time relation, deadline type), altered_numbers (amounts, periods, percentages), altered_entities (party/actor), altered_reference (section/statute/clause cited). Also record the **evidence**: a verbatim quote from each sentence.

## Worked examples
**English.** O: "The landlord shall give the tenant at least 24 hours' prior notice before entering, except in an emergency."
- S "Except in an emergency, the landlord must tell the tenant at least 24 hours before coming in." -> 9, Preserved, no labels.
- S "... shall give the tenant prior notice before entering, except in an emergency." -> 6, Partial, `altered_numbers` (period dropped).
- S "... at least 48 hours' prior notice ..." -> 3, Not Preserved, `altered_numbers`.
- S "The landlord may enter at any time and inform the tenant afterwards." -> 1, Not Preserved, `changed_obligation|missing_exception|contradiction`.

**Hindi.** O: "किरायेदार मकान मालिक की पूर्व लिखित सहमति के बिना परिसर का कोई भी भाग उप-किराये पर नहीं देगा।"
- S "किरायेदार को परिसर का कोई भी भाग उप-किराये पर देने से पहले मकान मालिक की लिखित सहमति पहले से लेनी होगी।" -> 9, Preserved.
- S "किरायेदार मकान मालिक की सहमति के बिना परिसर का कोई भी भाग उप-किराये पर नहीं देगा।" -> 6, Partial, `omission` ("पूर्व लिखित" dropped).
- S "किरायेदार मकान मालिक को लिखित सूचना देने के बाद ... उप-किराये पर दे सकता है।" -> 2, Not Preserved, `changed_permission|altered_condition`.
Hindi notes: judge negation (नहीं/न/बिना), modals (होगा/सकता है/चाहिए), conditionals (यदि/जब तक/तभी), and loanwords (नोटिस, लाइसेंस) as meaning-bearing; synonyms such as सूचना/नोटिस are acceptable.

## CSV schema for independent human annotation (`data/annotation/human/`, long format)
`pair_id, annotator_id, annotator_type(=human), lmp_score, category, error_labels (| separated), changed_elements (JSON), explanation, evidence_original, evidence_simplified, confidence, notes`.
Adjudication sheet columns: `final_score, final_category, final_error_labels, adjudicator_id, adjudicator_type, rationale`.

## Workflow
1. Export items: `data/expanded/pairs_consolidated.csv` (hide existing labels from annotators). 2. Each annotator fills the schema independently. 3. `python scripts/annotate.py validate --out <file>`; 4. `compare` -> disagreements; 5. adjudicate by discussion or a third annotator; 6. `agreement` -> weighted Cohen's kappa (quadratic on scores, linear on categories), Krippendorff alpha (interval/ordinal). **Report agreement only from genuine independent human records.** The `agreement_report.json` in this repository is AI-vs-AI and is labelled as such.
