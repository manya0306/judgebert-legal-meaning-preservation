# Dataset documentation

## Tiers and provenance (never merged under a "human" label)
| tier | n | origin | labels |
|---|---|---|---|
| original_synthetic | 48 | written by the project author for the prototype | author-assigned scores; not independent annotations |
| ai_generated | 504 | 42 "concepts" (one legal-style rule each, English + Hindi original) x 6 variants x 2 languages, authored by Claude (claude-sonnet-5.5) in-session | generation-pass labels (`ai_generation_intent`); 168 pairs additionally annotated in a blind second pass; 11 disagreements adjudicated by Claude |
| external | 0 | none: no suitable, verifiably licensed paired legal-simplification corpus was reachable (no network access beyond PyPI/GitHub) | - |
| human_annotated | 0 | none | - |
All text is **legal-style**, not quoted from real statutes, contracts or judgments; section numbers are invented. License: project-original generated text, no third-party rights asserted.

## Generation strategy (`generation_strategy`)
faithful_paraphrase (P), heavy_rewrite (H: low lexical overlap, meaning kept), lossy_simplification (L: a legally relevant element lost -> partial), minimal_meaning_edit (M: one-element change -> mostly Not Preserved; high lexical overlap), compound_change (X). Every concept covers: obligation/permission, negation/reversal, conditions, exceptions, actors, scope/quantifiers, deadlines, amounts/penalties, references (few), across 16 domains (tenancy, employment, insurance, consumer, contract, banking, property, family, tax, administrative, criminal procedure, company, privacy, motor vehicles, IP, environment).
Size: **552 pairs, 54 groups** (target was 1,000-2,000; not reached - quality and annotation consistency were prioritised and budget was finite). Extending is mechanical: add `data/generated/concepts_NN.py`, run `scripts/prepare_data.py`.

## Counts (from `data/expanded/dataset_report.json`)
- generated 504 + 48 original; second-pass annotated 168; disagreements 11 (6.5% of 168), all adjudicated; single-pass-only 336.
- Categories: Not 209, Partial 191, Preserved 152; en 276 / hi 276; per language x category within +-3.
- Accepted in the quality-filtered set 551, rejected 1 (`AIG-C039-EN-04`, a lossy variant scored "Preserved" that was flagged by QA and not re-annotated).

## Status ladder (`annotation_status`)
`author_assigned_score_not_independent_annotation` (48) < `ai_single_pass_unreviewed` (336) < `ai_two_pass_agreed` (157) / `ai_adjudicated_not_independent` (11). Nothing is `human*`.

## Quality checks (`lmp/data/qa.py`; flags kept in `qa_flags`, `qa_flags.csv`, never silently dropped)
invalid labels/scores, category-score mismatch, exact duplicates, contradictory annotations, near-duplicates across groups (merged for splitting), concept/translation group integrity, repeated openings/edit signatures, Hindi script ratio/orthography (double virama, stray matra, fused Latin, repeated word), EN<->HI glossary consistency, label-vs-text plausibility, class/language imbalance, lexical-cue reliance. Final flags: 30 info (`numeric_label_without_numeric_difference`: vague-quantity changes), 8 info (repeated edit signature "-shall +may"), 2 warn (`identical_tokens_low_score`: genuine actor-swap pairs, kept), 1 warn lossy-scored-preserved (rejected), 1 warn terminology (benign). These checks catch mechanical problems only; they cannot certify Hindi fluency or legal correctness. **No native-speaker or legal-expert review has taken place.**

## Known construction artifacts (important)
- Minimal-edit meaning changes (M) overlap the original *more* than faithful heavy paraphrases (H). Jaccard vs score: Spearman **-0.47** (en -0.50, hi -0.42). A sign-free TF-IDF calibration therefore reaches Spearman 0.45 without any semantics. Treat ~0.45 as the "artifact floor" for this dataset.
- English and Hindi are parallel by construction (translation counterparts), so cross-language agreement is inflated; grouping prevents leakage but not this similarity.
- Identical pairs never occur in the data; lexical models score them as *not* preserved (E5). The app special-cases identical text by rule.
- Same-model generation and annotation: errors are correlated; kappa 0.99 is not evidence of correct labels.
