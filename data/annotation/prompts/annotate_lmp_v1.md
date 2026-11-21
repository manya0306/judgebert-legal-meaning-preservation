# Prompt `annotate_lmp_v1` (used by API / local-model providers; the in-session pass used the same rubric)

You are annotating ONE pair of sentences for LEGAL MEANING PRESERVATION. Judge only whether the SIMPLIFIED sentence keeps the
legal meaning of the ORIGINAL sentence. Do not reward fluency or brevity. Do not give legal advice.

Language: {language}. Domain: {legal_domain}.
ORIGINAL: {original}
SIMPLIFIED: {simplified}

Score 1-10 (integers or .5):
  9-10  same legal effect; wording differs only stylistically
  7.5-8.5 same legal effect; only a trivial nuance differs  -> category "Preserved" (>= 7.5)
  4-7   core rule survives but a legally relevant element (condition, exception, number, scope, party, formality) is lost or altered
        -> category "Partially Preserved" (4 <= score < 7.5)
  1-3.5 legal effect reversed, removed, or materially changed (modal flip, negation, number/date change, actor swap)
        -> category "Not Preserved" (< 4)

Changed-element labels (any that apply, else empty): omission, contradiction, changed_obligation, changed_permission,
altered_condition, missing_exception, scope_change, altered_dates, altered_numbers, altered_entities, altered_reference.

Return ONLY JSON:
{"lmp_score": <number>, "category": "...", "error_labels": ["..."], "changed_elements": [{"element": "...", "original_span": "...", "simplified_span": "..."}],
 "explanation": "<1-3 sentences>", "evidence_original": "<quote from ORIGINAL>", "evidence_simplified": "<quote from SIMPLIFIED or (absent)>",
 "confidence": <0-1>}
Quotes must be verbatim substrings of the inputs.
