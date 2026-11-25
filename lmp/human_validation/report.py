"""Markdown research report generated ONLY from computed tables (no hard-coded results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import CATEGORIES, EXPERT_CONFIDENCE_NOTE


def f(x, d=2, sign=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if x != x else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def pick(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def val(df, **kw):
    r = pick(df, **kw)
    return float(r.value.iloc[0]) if len(r) else np.nan


def with_ci(df, **kw):
    r = pick(df, **kw)
    if r.empty:
        return "n/a"
    r = r.iloc[0]
    s = f(r.value, 3)
    return s + (f" [{f(r.ci_low, 3)}, {f(r.ci_high, 3)}]" if r.ci_low == r.ci_low else "")


def md(df, cols=None, rename=None):
    d = df[cols] if cols else df
    h = [(rename or {}).get(c, c) for c in d.columns]
    L = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
    for r in d.itertuples(index=False):
        L.append("| " + " | ".join(f(v, 3) if isinstance(v, float) else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(L)


def build_report(cons, ind, tables, validation, fig_status, meta, spec) -> str:
    ag, ex, ref, sg, sem, mod = (tables[k] for k in ("student_agreement", "expert", "reference", "subgroup", "semantic", "model"))
    n, n_en, n_hi = len(cons), int((cons.language == "en").sum()), int((cons.language == "hi").sum())
    n_test, n_exp = int(cons.is_heldout_test.sum()), int(cons.expert_reviewed.sum())
    n_exp_test = int((cons.expert_reviewed & cons.is_heldout_test).sum())
    L = ["# Human validation analysis - JUDGEBERT legal meaning preservation", "",
         f"_Generated from the completed annotation workbooks (inputs hashed in run_metadata.json). Seed {meta['seed']}, {meta['n_boot']} group-bootstrap resamples. PRIVATE: contains raw annotations and explanations._", ""]
    L += ["## 1. Annotation methodology", "",
          f"* {n} unique synthetic legal-style sentence pairs ({n_en} English, {n_hi} Hindi). Annotators A1-A3 judged all {n}; A4 judged the {n_en} English pairs "
          f"(Hindi missing by design). Student judgments: {int((ind.annotator_role == 'student').sum())}. One legal professional judged {n_exp} of the pairs "
          f"({int(((cons.language == 'en') & cons.expert_reviewed).sum())} EN, {int(((cons.language == 'hi') & cons.expert_reviewed).sum())} HI); total human judgments: {len(ind)}.",
          "* Rubric (integers 1-10): 1-3 Not Preserved, 4-6 Partially Preserved, 7-10 Preserved. Annotators also gave confidence, a semantic change type and a written explanation.",
          "* Student consensus: mean score (primary), median (sensitivity); majority-vote category with explicit tie flag; continuous mean/median mapped to categories with boundaries at 3.5 and 6.5 "
          "(kept separate from majority vote). Scores are never rounded. Student consensus and the expert judgment are kept separate; neither is treated as definitive ground truth.",
          "* Agreement: pairwise quadratic-weighted and unweighted Cohen's kappa, Fleiss' kappa, Krippendorff's alpha (ordinal/nominal; interval as sensitivity), Spearman/Pearson, MAE, exact-category and within-1/2-point agreement. "
          "CIs resample whole provision/concept groups (`group_id`), because variants of one provision are not independent.",
          "* Sign convention: bias = comparator - standard (reference - human; student - expert). Positive reference bias = the reference label is more favourable than the human judgment.",
          f"* Expert corrections (manually incorporated before analysis; verified, not re-applied): " + "; ".join(f"{p} {o}->{s} ({c})" for p, o, s, c in spec.corrections) + ".",
          f"* Expert confidence: {EXPERT_CONFIDENCE_NOTE}", ""]
    L += ["## 2. Dataset composition", "", "**Language x dataset split**", "",
          md(pd.crosstab(cons.language, cons.dataset_split).reset_index().rename(columns={"language": "language"})), "",
          "**Complexity x language**", "", md(pd.crosstab(cons.complexity, cons.language).reset_index()), "",
          "**Reference label provenance**", "", md(cons.reference_label_type.value_counts().rename_axis("reference_label_type").reset_index(name="pairs")), "",
          f"Fixed test split: {n_test} pairs (expert-reviewed: {n_exp_test}); supplemental train/validation: {n - n_test} pairs (expert-reviewed: {n_exp - n_exp_test}). "
          "The fixed test split was already used in exploratory model evaluation and the supplemental pairs were used for model development; none of this is an untouched final test set.", ""]
    L += ["## 3. Data validation", ""]
    d = validation.to_dict()
    L += [f"Overall: **{d['overall']}**. Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(d["counts"].items())) + ". Full list: `validation_report.md`.", ""]
    for c in validation.checks:
        if c.status in ("WARN", "UNVERIFIED"):
            L.append(f"* {c.status}: {c.message}")
    L += ["", "## 4. Student inter-annotator agreement", ""]
    heads = ["fleiss_kappa_category", "krippendorff_alpha_ordinal_score", "krippendorff_alpha_nominal_category", "mean_pairwise_qwk_score", "mean_pairwise_mae_score", "mean_pairwise_exact_category_agreement", "mean_pairwise_within_1_point", "mean_pairwise_within_2_points"]
    rows = []
    for sub in ("english", "english_A1-A3", "hindi", "combined_A1-A3"):
        r = pick(ag, scope="student_agreement", subset=sub)
        if r.empty:
            continue
        rows.append({"subset": sub, "raters": r.rater_set.iloc[0], "pairs": int(cons.language.eq("en").sum()) if sub.startswith("english") else n_hi if sub == "hindi" else n,
                     **{h.replace("krippendorff_alpha", "alpha").replace("mean_pairwise_", "pw_"): with_ci(ag, scope="student_agreement", subset=sub, metric=h) for h in heads}})
    L += [md(pd.DataFrame(rows)), "", "Brackets = 95% group-bootstrap CI. English with A1-A3 only is shown for like-for-like comparison with Hindi; A4 is English only.", "",
          "**Pairwise annotator statistics** (n = pairs both annotators judged)", ""]
    pw = pick(ag, scope="student_agreement_pairwise")
    wide = pw.pivot_table(index=["subset", "rater_set", "n"], columns="metric", values="value").reset_index()
    L += [md(wide[[c for c in ["subset", "rater_set", "n", "qwk_score", "kappa_category", "spearman_score", "pearson_score", "mae_score", "exact_category_agreement", "within_1_point", "within_2_points"] if c in wide]]), ""]
    dsp = pick(ag, scope="per_example_dispersion")
    L += ["**Per-example dispersion**", "", md(dsp.pivot_table(index="subset", columns="metric", values="value").reset_index()), ""]
    L += ["## 5. Student consensus vs expert", ""]
    if ex.empty:
        L += ["No expert-reviewed pairs.", ""]
    else:
        e = pick(ex, comparator="student_mean")
        w = e.pivot_table(index="subset", columns="metric", values="value").reset_index()
        L += [f"Single legal professional, {n_exp} overlapping pairs. Student mean vs expert (bias = student - expert):", "",
              md(w[[c for c in ["subset", "spearman", "pearson", "mae", "rmse", "bias", "within_1_point", "category_accuracy", "category_kappa", "category_macro_f1"] if c in w]]), "",
              f"All expert pairs: MAE {with_ci(ex, subset='expert_all', comparator='student_mean', metric='mae')}, bias {with_ci(ex, subset='expert_all', comparator='student_mean', metric='bias')}, "
              f"Spearman {with_ci(ex, subset='expert_all', comparator='student_mean', metric='spearman')}. Correlation does not establish absolute agreement: see MAE and bias.", ""]
        q = tables["qualitative"]
        for tname in ("expert_vs_student_largest_score_gaps", "expert_vs_student_category_disagreement"):
            t = q[q.table == tname]
            L += [f"**{tname}** ({len(t)} pairs; full texts and explanations in `metrics/qualitative_disagreements.csv`)", ""]
            if len(t):
                L += [md(t[["pair_id", "language", "dataset_split", "student_scores", "student_mean", "expert_score", "expert_category", "expert_semantic_change"]].head(10)), ""]
    L += ["## 6. Validation of the AI-generated / author-assigned reference labels", "",
          "AI-human label agreement below is **not** trained-model performance. Reference categories come in two clearly separate variants: `original_repo_thresholds_4_7.5` "
          "(Preserved >= 7.5, Not Preserved < 4) and `harmonized_human_rubric_3.5_6.5`.", ""]
    s = pick(ref, scope="reference_vs_human", comparator="student_mean", reference_variant="score")
    s = s.pivot_table(index="subset", columns="metric", values="value").reset_index()
    L += [md(s[[c for c in ["subset", "spearman", "pearson", "mae", "rmse", "bias", "within_1_point", "share_comparator_higher", "share_comparator_lower", "wilcoxon_signed_rank_p"] if c in s]]), ""]
    b = pick(ref, scope="reference_vs_human", comparator="student_mean", reference_variant="score", subset="all", metric="bias")
    if len(b):
        r0 = b.iloc[0]
        direction = "higher (more favourable to preservation)" if r0.value > 0 else "lower (less favourable)"
        sig = "the CI excludes 0" if (r0.ci_low == r0.ci_low and (r0.ci_low > 0 or r0.ci_high < 0)) else "the CI includes 0 or is undefined"
        L += [f"Overall, reference scores are on average {f(abs(r0.value))} points {direction} than the student mean (bias {with_ci(ref, scope='reference_vs_human', comparator='student_mean', reference_variant='score', subset='all', metric='bias')}; {sig}).", ""]
    for comp in ("student_mean", "student_median", "student_majority_vote", "student_individual", "expert"):
        t = ref[(ref.scope == "reference_vs_human") & (ref.comparator == comp) & (ref.subset == "all") & ref.metric.isin(["category_accuracy", "category_macro_f1", "category_kappa"])]
        if len(t):
            w = t.pivot_table(index="reference_variant", columns="metric", values="value").reset_index()
            L += [f"**Category agreement, reference vs {comp} (all annotated pairs)**", "", md(w), ""]
    L += ["**Subgroups** (language, complexity, domain, provenance, semantic label, split) are in `metrics/subgroup_analysis.csv`; correlations need n >= 10, MAE/bias n >= 5.", ""]
    L += ["## 7. Language and complexity", ""]
    for dim in ("language", "complexity"):
        t = sg[(sg.dimension == dim) & sg.metric.isin(["n_pairs", "student_sd_mean", "share_student_range_ge_3", "krippendorff_alpha_ordinal", "reference_vs_student_mean_mae", "reference_vs_student_mean_bias"])]
        L += [md(t.pivot_table(index="group", columns="metric", values="value").reset_index().rename(columns={"group": dim})), ""]
    L += ["## 8. Semantic change-type analysis", "",
          "Human labels (13 types) and the dataset's error taxonomy (11 labels) differ; the mapping used is in `semantic_error_analysis.csv` (section `taxonomy_mapping`). "
          "No evidence-span accuracy is claimed: the human annotations contain no independently annotated spans.", ""]
    t = sem[sem.section.isin(["student_label_agreement", "student_expert_label_agreement"])]
    L += [md(t[["section", "subset", "metric", "n", "value", "note"]]), ""]
    L += ["## 9. Model evaluation", ""]
    ev = mod[mod.status == "EVALUATED"] if len(mod) else mod
    if len(ev):
        t = ev[(ev.target == "student_mean") & (ev.subset.isin(["all", "fixed_test"]))]
        L += [md(t[["model_name", "subset", "n", "mae", "rmse", "spearman", "accuracy", "f1_macro", "heldout_status", "training_provenance"]]), "",
              "Held-out statements apply only where `heldout_status` says so; the fixed test split was previously evaluated and is not an untouched final test set."]
    nr = mod[mod.status == "NOT RUN"] if len(mod) else mod
    for r in nr.itertuples():
        L.append(f"* {r.model_name}: {r.reason}")
    L += ["", "## 10. Figures", ""] + [f"* {k}: {v}" for k, v in fig_status.items()] + [""]
    L += ["## 11. Limitations and supported conclusions", "",
          "* The text is synthetic legal-style (AI-generated or author-written), not authentic statutes, and the benchmark is sentence-level; paragraph- and document-level legal reasoning is not established.",
          f"* The human sample is small ({n} pairs, {n_en}/{n_hi} by language; {n_exp} expert pairs) and only one legal professional participated; the expert is not definitive legal ground truth.",
          "* Some annotated pairs were previously used for model development, and the fixed test set was already evaluated; neither is an untouched final test.",
          "* AI-generated reference labels are not independent human ground truth. Correlation does not establish absolute score agreement (see MAE, bias, within-k).",
          f"* Expert confidence responses were excluded: {EXPERT_CONFIDENCE_NOTE}",
          "* Group-bootstrap CIs rely on few independent groups; treat them as rough uncertainty ranges. Subgroup cells with small n are not interpretable.",
          "* No legal reliability is claimed. Supported: descriptive agreement among the annotators, and how far the existing labels and (if run) models deviate from this human sample."]
    return "\n".join(L) + "\n"
