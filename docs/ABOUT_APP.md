### What this tool does
Given an original legal sentence and a simplified version (English or Hindi), it (1) scores similarity with baselines, (2) predicts a
Legal Meaning Preservation (LMP) score from 1-10 with a model that learns from LMP annotations, and (3) lists potentially altered legal
elements (obligation/permission, negation, conditions, exceptions, numbers, dates, scope, parties) using transparent rules.

### Important limitations
- **Demo data are synthetic.** The 48 bundled pairs were written by the project author; their scores are author-assigned, not independent human annotations.
- Models trained on them are for pipeline demonstration only. A fine-tuned transformer needs a real annotated dataset (target ≥300 pairs, ≥2 annotators).
- The rule-based analyser is a heuristic aid with known false positives/negatives (e.g. "before"/"after" paraphrases, implicit changes).
- Similarity and legal meaning preservation are different quantities and are displayed separately.
- **Not legal advice.**
- Inspired by JUDGEBERT (Beauchemin et al., EMNLP 2025); this is not an exact reproduction.
