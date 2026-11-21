# Experiment 5 - Qualitative error analysis

16 pairs. **All pairs are SYNTHETIC** (author-written); 'Reference' is the author-assigned score, not a human annotation.
Scores are produced by the models that could be run in this environment (see results.json -> meta.model_status).

## SYN-EN-001 (en, tenancy)
- Original: The tenant shall give 30 days' written notice before vacating.
- Simplified: The tenant may tell the landlord before leaving.
- Reference LMP (author): 2 - gold labels: `changed_obligation|altered_numbers|omission`
- Model scores: TF-IDF cosine (raw): 4.31; TF-IDF cosine (calibrated): 3.42; Feature-ridge (offline fallback): 1.00; Sentence-Transformer cosine (raw): 6.41; Sentence-Transformer cosine (calibrated): 3.69; JUDGEBERT-inspired regressor: 5.28
- Explainer labels: `altered_entities|altered_numbers|changed_obligation|omission`

  - **changed_obligation** (conf 0.90, rule `modal-class-mismatch`): detected: Modal expression "shall" (obligation) in the original vs "may" (permission) in the simplified text. Interpretation: A mandatory duty appears to have become optional (permission).
  - **altered_numbers** (conf 0.85, rule `numbers-mismatch`): detected: Number 30 appears only in the original. Interpretation: A specific figure/deadline was dropped; the obligation may now be less precise or unenforceable as written.
  - **omission** (conf 0.65, rule `salient-notion-dropped`): detected: Salient notion "written form" (written) has no counterpart in the simplified text. Interpretation: A formal requirement (written form) may have been dropped.
  - **altered_entities** (conf 0.40, rule `entity-set-differs`): detected: Party/entity terms differ: only in original []; only in simplified ['landlord']. Interpretation: A person, role or thing named in the rule may have been removed or added.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-016 (en, employment)
- Original: The employee shall not work for a competitor for one year after leaving the company.
- Simplified: The employee may work for a competitor after leaving the company.
- Reference LMP (author): 1 - gold labels: `contradiction|changed_obligation|altered_numbers`
- Model scores: TF-IDF cosine (raw): 6.96; TF-IDF cosine (calibrated): 5.41; Feature-ridge (offline fallback): 1.66; Sentence-Transformer cosine (raw): 6.52; Sentence-Transformer cosine (calibrated): 3.80; JUDGEBERT-inspired regressor: 5.30
- Explainer labels: `altered_numbers|contradiction`

  - **contradiction** (conf 0.85, rule `modal-class-mismatch`): detected: Modal expression "shall not" (prohibition) in the original vs "may" (permission) in the simplified text. Interpretation: A prohibition became a permission; the two may have opposite legal effect.
  - **altered_numbers** (conf 0.85, rule `numbers-mismatch`): detected: Number one appears only in the original. Interpretation: A specific figure/deadline was dropped; the obligation may now be less precise or unenforceable as written.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-006 (en, tenancy)
- Original: The tenant shall not make any permanent alterations to the premises.
- Simplified: The tenant may make some changes to the premises.
- Reference LMP (author): 1 - gold labels: `contradiction|changed_permission`
- Model scores: TF-IDF cosine (raw): 7.64; TF-IDF cosine (calibrated): 5.93; Feature-ridge (offline fallback): 1.58; Sentence-Transformer cosine (raw): 5.73; Sentence-Transformer cosine (calibrated): 3.00; JUDGEBERT-inspired regressor: 5.23
- Explainer labels: `contradiction|scope_change`

  - **contradiction** (conf 0.85, rule `modal-class-mismatch`): detected: Modal expression "shall not" (prohibition) in the original vs "may" (permission) in the simplified text. Interpretation: A prohibition became a permission; the two may have opposite legal effect.
  - **scope_change** (conf 0.65, rule `scope-quantifier`): detected: Scope term(s) differ: original ['universal'] vs simplified ['vague']. Interpretation: The set of people/things/situations covered may have been widened or narrowed.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-HI-006 (hi, tenancy)
- Original: किरायेदार मकान में किसी भी प्रकार का स्थायी बदलाव नहीं करेगा।
- Simplified: किरायेदार मकान में कुछ बदलाव कर सकता है।
- Reference LMP (author): 1 - gold labels: `contradiction|changed_permission`
- Model scores: TF-IDF cosine (raw): 5.63; TF-IDF cosine (calibrated): 4.42; Feature-ridge (offline fallback): 1.34; Sentence-Transformer cosine (raw): 5.61; Sentence-Transformer cosine (calibrated): 2.87; JUDGEBERT-inspired regressor: 5.20
- Explainer labels: `contradiction|scope_change`

  - **contradiction** (conf 0.85, rule `modal-class-mismatch`): detected: Modal expression "नहीं करेगा" (prohibition) in the original vs "सकता" (permission) in the simplified text. Interpretation: A prohibition became a permission; the two may have opposite legal effect.
  - **scope_change** (conf 0.65, rule `scope-quantifier`): detected: Scope term(s) differ: original ['universal'] vs simplified ['vague']. Interpretation: The set of people/things/situations covered may have been widened or narrowed.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-005 (en, tenancy)
- Original: If rent is not paid on time, the landlord may terminate the agreement.
- Simplified: The landlord may terminate the agreement.
- Reference LMP (author): 3 - gold labels: `altered_condition|omission`
- Model scores: TF-IDF cosine (raw): 7.44; TF-IDF cosine (calibrated): 5.78; Feature-ridge (offline fallback): 3.47; Sentence-Transformer cosine (raw): 8.88; Sentence-Transformer cosine (calibrated): 6.21; JUDGEBERT-inspired regressor: 5.27
- Explainer labels: `altered_condition`

  - **altered_condition** (conf 0.75, rule `condition-dropped`): detected: Condition marker(s) If appear only in the original. Interpretation: A precondition may have been dropped so the rule now applies unconditionally.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-014 (en, employment)
- Original: The employer may terminate the contract without notice in cases of gross misconduct.
- Simplified: The employer may terminate the contract without notice.
- Reference LMP (author): 3 - gold labels: `altered_condition|omission`
- Model scores: TF-IDF cosine (raw): 7.75; TF-IDF cosine (calibrated): 6.01; Feature-ridge (offline fallback): 3.69; Sentence-Transformer cosine (raw): 9.61; Sentence-Transformer cosine (calibrated): 6.95; JUDGEBERT-inspired regressor: 5.25
- Explainer labels: `altered_condition`

  - **altered_condition** (conf 0.75, rule `condition-dropped`): detected: Condition marker(s) in cases of appear only in the original. Interpretation: A precondition may have been dropped so the rule now applies unconditionally.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-007 (en, tenancy)
- Original: The landlord shall bear the cost of repairs unless the damage was caused by the tenant.
- Simplified: The landlord shall bear the cost of repairs.
- Reference LMP (author): 3 - gold labels: `missing_exception`
- Model scores: TF-IDF cosine (raw): 7.23; TF-IDF cosine (calibrated): 5.62; Feature-ridge (offline fallback): 3.82; Sentence-Transformer cosine (raw): 9.06; Sentence-Transformer cosine (calibrated): 6.40; JUDGEBERT-inspired regressor: 5.28
- Explainer labels: `missing_exception`

  - **missing_exception** (conf 0.80, rule `exception-dropped`): detected: Exception marker(s) unless appear only in the original. Interpretation: A carve-out or exception may have been removed, widening the rule.

- Explainer vs gold labels: exact match

## SYN-EN-024 (en, insurance)
- Original: The policy expires on 31 March 2026 unless renewed.
- Simplified: The policy expires on 31 March 2026.
- Reference LMP (author): 4 - gold labels: `missing_exception`
- Model scores: TF-IDF cosine (raw): 8.67; TF-IDF cosine (calibrated): 6.71; Feature-ridge (offline fallback): 4.24; Sentence-Transformer cosine (raw): 9.73; Sentence-Transformer cosine (calibrated): 7.08; JUDGEBERT-inspired regressor: 5.22
- Explainer labels: `missing_exception`

  - **missing_exception** (conf 0.80, rule `exception-dropped`): detected: Exception marker(s) unless appear only in the original. Interpretation: A carve-out or exception may have been removed, widening the rule.

- Explainer vs gold labels: exact match

## SYN-EN-004 (en, tenancy)
- Original: Rent must be paid on or before the 5th day of each month.
- Simplified: Rent must be paid on or before the 10th day of each month.
- Reference LMP (author): 3 - gold labels: `altered_dates`
- Model scores: TF-IDF cosine (raw): 10.00; TF-IDF cosine (calibrated): 7.71; Feature-ridge (offline fallback): 5.28; Sentence-Transformer cosine (raw): 8.67; Sentence-Transformer cosine (calibrated): 5.99; JUDGEBERT-inspired regressor: 5.33
- Explainer labels: `altered_dates`

  - **altered_dates** (conf 0.90, rule `dates-mismatch`): detected: Original dates: 5th; simplified dates: 10th. Interpretation: A date appears to have been replaced by a different one.

- Explainer vs gold labels: exact match

## SYN-EN-011 (en, employment)
- Original: The employer shall pay salary on or before the 7th of each month.
- Simplified: The employer shall pay salary on or before the 17th of each month.
- Reference LMP (author): 3 - gold labels: `altered_dates`
- Model scores: TF-IDF cosine (raw): 8.42; TF-IDF cosine (calibrated): 6.52; Feature-ridge (offline fallback): 5.19; Sentence-Transformer cosine (raw): 7.98; Sentence-Transformer cosine (calibrated): 5.29; JUDGEBERT-inspired regressor: 5.31
- Explainer labels: `altered_dates`

  - **altered_dates** (conf 0.90, rule `dates-mismatch`): detected: Original dates: 7th; simplified dates: 17th. Interpretation: A date appears to have been replaced by a different one.

- Explainer vs gold labels: exact match

## SYN-EN-008 (en, tenancy)
- Original: The tenant and members of the tenant's family may reside in the premises.
- Simplified: Only the tenant may live in the premises.
- Reference LMP (author): 3 - gold labels: `scope_change|altered_entities`
- Model scores: TF-IDF cosine (raw): 4.04; TF-IDF cosine (calibrated): 3.21; Feature-ridge (offline fallback): 2.27; Sentence-Transformer cosine (raw): 7.79; Sentence-Transformer cosine (calibrated): 5.10; JUDGEBERT-inspired regressor: 5.14
- Explainer labels: `altered_entities|scope_change`

  - **scope_change** (conf 0.65, rule `scope-quantifier`): detected: Scope term(s) differ: original [] vs simplified ['restrictive']. Interpretation: The set of people/things/situations covered may have been widened or narrowed.
  - **altered_entities** (conf 0.55, rule `entity-set-differs`): detected: Party/entity terms differ: only in original ['family']; only in simplified []. Interpretation: A person, role or thing named in the rule may have been removed or added.

- Explainer vs gold labels: exact match

## SYN-EN-012 (en, employment)
- Original: The employee shall not disclose confidential information during or after employment.
- Simplified: The employee shall not disclose confidential information during employment.
- Reference LMP (author): 3 - gold labels: `scope_change|omission`
- Model scores: TF-IDF cosine (raw): 8.09; TF-IDF cosine (calibrated): 6.27; Feature-ridge (offline fallback): 4.91; Sentence-Transformer cosine (raw): 9.87; Sentence-Transformer cosine (calibrated): 7.22; JUDGEBERT-inspired regressor: 5.30
- Explainer labels: `scope_change`

  - **scope_change** (conf 0.55, rule `time-scope-narrowed`): detected: Time relation(s) ['after'] appear only in the original. Interpretation: The period the rule covers may have been narrowed.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-003 (en, tenancy)
- Original: The tenant shall not sublet the premises without the landlord's written consent.
- Simplified: The tenant must get the landlord's written permission before subletting the premises.
- Reference LMP (author): 9 - gold labels: `none`
- Model scores: TF-IDF cosine (raw): 7.43; TF-IDF cosine (calibrated): 5.77; Feature-ridge (offline fallback): 8.64; Sentence-Transformer cosine (raw): 8.60; Sentence-Transformer cosine (calibrated): 5.93; JUDGEBERT-inspired regressor: 5.28
- Explainer labels: `none`


- Explainer vs gold labels: exact match

## SYN-EN-009 (en, employment)
- Original: The employee shall give two months' notice before resigning.
- Simplified: The employee has to tell the employer two months before resigning.
- Reference LMP (author): 9 - gold labels: `none`
- Model scores: TF-IDF cosine (raw): 6.32; TF-IDF cosine (calibrated): 4.93; Feature-ridge (offline fallback): 5.52; Sentence-Transformer cosine (raw): 9.04; Sentence-Transformer cosine (calibrated): 6.37; JUDGEBERT-inspired regressor: 5.31
- Explainer labels: `altered_entities`

  - **altered_entities** (conf 0.40, rule `entity-set-differs`): detected: Party/entity terms differ: only in original []; only in simplified ['employer']. Interpretation: A person, role or thing named in the rule may have been removed or added.

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-015 (en, employment)
- Original: The employee must complete a probation period of six months before confirmation.
- Simplified: New employees will be confirmed after a probation period of six months.
- Reference LMP (author): 9 - gold labels: `none`
- Model scores: TF-IDF cosine (raw): 5.06; TF-IDF cosine (calibrated): 3.98; Feature-ridge (offline fallback): 6.04; Sentence-Transformer cosine (raw): 6.77; Sentence-Transformer cosine (calibrated): 4.05; JUDGEBERT-inspired regressor: 5.31
- Explainer labels: `altered_dates`

  - **altered_dates** (conf 0.60, rule `time-relation-changed`): detected: Time relation changed from ['before'] to ['after']. Interpretation: A deadline or time reference may have shifted (e.g. 'within' vs 'after').

- Explainer vs gold labels: DIFFERS - inspect (false positive / false negative / label ambiguity)

## SYN-EN-017 (en, insurance)
- Original: The insurer shall pay the claim within 30 days of receiving all required documents.
- Simplified: The insurer will pay the claim within 30 days of receiving the required documents.
- Reference LMP (author): 9 - gold labels: `none`
- Model scores: TF-IDF cosine (raw): 7.50; TF-IDF cosine (calibrated): 5.82; Feature-ridge (offline fallback): 8.84; Sentence-Transformer cosine (raw): 9.79; Sentence-Transformer cosine (calibrated): 7.14; JUDGEBERT-inspired regressor: 5.41
- Explainer labels: `none`


- Explainer vs gold labels: exact match
