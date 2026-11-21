# Limitations (read before citing any number)

1. **No human annotation and no real-world legal text.** All 552 pairs are author-written or AI-generated legal-*style* sentences. Labels come from the generating model (+ a same-family blind second pass on 168 pairs). The reported AI-vs-AI agreement (weighted kappa 0.99, Krippendorff alpha 0.99/0.97) is *not* human inter-annotator agreement and mostly shows one model applying one rubric consistently; shared mistakes are invisible to it.
2. **Hindi quality is unreviewed.** Automatic checks only catch mechanical orthography problems. No native-speaker or legal-expert review occurred; legal terminology choices may be unidiomatic.
3. **Small and parallel.** 54 groups; English and Hindi are parallel translations; 6 variants per concept share a source sentence. Grouped CV prevents leakage but effective sample size is ~54, so CIs are wide and domain-level numbers (12-48 pairs) are anecdotal.
4. **Construction artifacts.** Meaning-changing minimal edits overlap the original more than faithful paraphrases (Spearman Jaccard/score -0.47); a sign-free TF-IDF reaches 0.45 Spearman without semantics. Trained lexical models fail the identical/unrelated sanity check (E5). Reported correlations are therefore partly shortcut-driven; the best model (0.58) is only moderately above this floor and fails on subtle reversals (e.g. "may enter at any time").
5. **No pretrained transformer was trained or evaluated.** Hugging Face was unreachable. The cross-encoder is implemented and tested, the collapse diagnosis used a tiny random-init encoder, and the Sentence-Transformer baseline was not run. Nothing here shows that XLM-R would or would not outperform the CPU models, or that the redesigned training avoids collapse on pretrained weights.
6. **Explanations are heuristics.** Rule micro-F1 on the dataset is 0.42; the rules flag ~80% of Preserved pairs. They are not faithful explanations of any neural model, not validated by lawyers, and "no finding" never implies preservation. No model-attribution method (LIME/SHAP/attention) is implemented in this version.
7. **Thresholds and scale.** The 1-10 score and 4/7.5 thresholds are conventions of this project, not an established legal standard. Scores near boundaries are unreliable.
8. **Single-sentence scope.** No cross-sentence context, definitions, or jurisdiction knowledge; invented section numbers; no statute lookup.
9. **Ablations** (E3/E4) use five repeated grouped holdouts from one dataset; differences of ~0.05 Spearman should not be interpreted.
10. **Size target missed.** 552 pairs vs the 1,000-2,000 target.
11. **Research prototype, not legal advice.**

## What evidence would make stronger claims possible
Independent annotation of >=300-500 pairs by two lawyers/linguists (workflow and tooling provided); real simplified legal text with verified licences; native-speaker Hindi review; a pretrained-encoder run with grouped CV and several seeds; an independent test of the explainer against human-marked changed elements; error bars from more groups.
