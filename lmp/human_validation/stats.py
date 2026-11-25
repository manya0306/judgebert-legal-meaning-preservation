"""Agreement, expert-comparison, reference-label validation, subgroup and semantic-label analyses.

All functions take the consolidated datasets from ``ingest`` and return long-format DataFrames. Group-aware bootstrap
CIs resample whole provision/concept groups (``group_id``) with a fixed seed, because variants of one provision (and its
EN/HI translation) are not independent observations. Undefined statistics are NaN with an explanatory note."""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score

from lmp.evaluation.metrics import regression_metrics

from .ingest import CATEGORIES, CHANGE_TYPES, OTHER_FREE_TEXT, STUDENT_SHEETS

NAN = float("nan")
MIN_N_BIAS, MIN_N_CORR = 5, 10          # smallest subgroup sizes for MAE/bias and for correlations / alpha
COLS = ["scope", "subset", "comparator", "reference_variant", "metric", "rater_set", "n", "value", "ci_low", "ci_high", "note"]
CAT_CODE = {c: i for i, c in enumerate(CATEGORIES)}
LABELS = list(CHANGE_TYPES) + [OTHER_FREE_TEXT]
# Human semantic-change labels -> dataset error taxonomy (lmp.data.schema.ERROR_LABELS). No equivalent -> empty set.
HUMAN_TO_DATASET = {
    "None (preserved)": set(), "Duty-Permission-Prohibition": {"changed_obligation", "changed_permission"}, "Condition": {"altered_condition"},
    "Exception-Proviso": {"missing_exception"}, "Party": {"altered_entities"}, "Scope": {"scope_change"}, "Right-Remedy-Procedure": set(),
    "Deadline-Time": {"altered_dates"}, "Number-Threshold": {"altered_numbers"}, "Reversal": {"contradiction"}, "Other omission": {"omission"},
    "Added content": set(), "Other": set(), OTHER_FREE_TEXT: set()}
NO_DATASET_EQUIVALENT = {k for k, v in HUMAN_TO_DATASET.items() if not v and k != "None (preserved)"}


# ----------------------------------------------------------------------------- primitives
def cohen_kappa(a, b, k, weights=None) -> float:
    """Cohen's kappa on integer codes 0..k-1 (weights None = unweighted, 'quadratic'); NaN if undefined."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n < 2:
        return NAN
    O = np.zeros((k, k))
    np.add.at(O, (a, b), 1)
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = (np.subtract.outer(np.arange(k), np.arange(k)) / max(k - 1, 1)) ** 2 if weights == "quadratic" else 1 - np.eye(k)
    den = (W * E).sum()
    return NAN if den == 0 else float(1 - (W * O).sum() / den)


def safe_corr(x, y, kind="spearman") -> float:
    from scipy.stats import pearsonr, spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return NAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float((spearmanr if kind == "spearman" else pearsonr)(x, y)[0])


def fleiss_kappa(codes, k) -> tuple[float, int]:
    """Fleiss' kappa for integer codes (units x raters, -1 = missing). Only units rated by every rater are used."""
    codes = np.asarray(codes, int)
    if codes.size == 0:
        return NAN, 0
    codes = codes[(codes >= 0).all(axis=1)]
    u, r = codes.shape if codes.ndim == 2 else (0, 0)
    if u < 2 or r < 2:
        return NAN, int(u)
    counts = np.zeros((u, k))
    for j in range(r):
        np.add.at(counts, (np.arange(u), codes[:, j]), 1)
    Pi = ((counts ** 2).sum(1) - r) / (r * (r - 1))
    pj = counts.sum(0) / (u * r)
    Pe = (pj ** 2).sum()
    return (NAN if Pe >= 1 else float((Pi.mean() - Pe) / (1 - Pe))), int(u)


def krippendorff_alpha(M, level="ordinal") -> float:
    """Krippendorff's alpha (units x raters, NaN = missing; units with <2 values are ignored)."""
    M = np.asarray(M, float)
    if M.size == 0 or np.isnan(M).all():
        return NAN
    vals = np.unique(M[~np.isnan(M)])
    k = len(vals)
    if k < 2:
        return NAN
    I = np.full(M.shape, -1)
    ok = ~np.isnan(M)
    I[ok] = np.searchsorted(vals, M[ok])
    counts = np.zeros((M.shape[0], k))
    for r in range(M.shape[1]):
        m = I[:, r] >= 0
        counts[np.flatnonzero(m), I[m, r]] += 1
    mu = counts.sum(1)
    use = mu >= 2
    if not use.any():
        return NAN
    c, mu = counts[use], mu[use]
    w = c / (mu - 1)[:, None]
    o = c.T @ w - np.diag(w.sum(0))
    nc = o.sum(1)
    N = nc.sum()
    if N <= 1 or (nc > 0).sum() < 2:
        return NAN
    if level == "nominal":
        d = 1.0 - np.eye(k)
    elif level == "interval":
        d = (vals[:, None] - vals[None, :]) ** 2
    else:
        cs = np.cumsum(nc)
        S = cs[None, :] - cs[:, None] + nc[:, None]
        d = (S - (nc[:, None] + nc[None, :]) / 2) ** 2
        d = np.where(np.arange(k)[:, None] <= np.arange(k)[None, :], d, d.T)
        np.fill_diagonal(d, 0)
    De = (np.outer(nc, nc) * d).sum() / (N * (N - 1))
    return NAN if De == 0 else float(1 - (o * d).sum() / N / De)


def group_bootstrap(groups, stat_fn, n_boot=1000, seed=42, alpha=0.05):
    """Percentile CI from resampling whole groups with replacement. ``stat_fn(idx)`` receives resampled row indices."""
    groups = np.asarray(groups)
    if n_boot <= 0 or len(groups) < 4:
        return NAN, NAN
    ug, inv = np.unique(groups.astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    if len(ug) < 3:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        v = stat_fn(np.concatenate([members[i] for i in pick]))
        if v is not None and np.isfinite(v):
            out.append(v)
    if len(out) < max(20, n_boot // 2):
        return NAN, NAN
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _r(scope, subset, metric, n, value, lo=NAN, hi=NAN, note="", comparator="", variant="", rater_set=""):
    return {"scope": scope, "subset": subset, "comparator": comparator, "reference_variant": variant, "metric": metric, "rater_set": rater_set,
            "n": int(n), "value": value, "ci_low": lo, "ci_high": hi, "note": note}


def subset_masks(df, only=None) -> dict:
    en, test = df["language"] == "en", df["is_heldout_test"].astype(bool)
    ai = df["reference_label_type"].astype(str).str.startswith("AI-GENERATED")
    m = {"all": pd.Series(True, index=df.index), "fixed_test": test, "supplemental_train_val": ~test, "english": en, "hindi": ~en,
         "english_fixed_test": en & test, "hindi_fixed_test": ~en & test, "english_supplemental": en & ~test, "hindi_supplemental": ~en & ~test,
         "ai_generated_labels": ai, "author_assigned_labels": ~ai}
    return {k: v for k, v in m.items() if only is None or k in only}


# ----------------------------------------------------------------------------- student agreement
def _codes(df, cols):
    return np.array([[CAT_CODE.get(v, -1) if isinstance(v, str) else -1 for v in df[c]] for c in cols]).T if len(df) else np.zeros((0, len(cols)), int)


def _pair_metrics(sx, sy, cx, cy) -> dict:
    n = len(sx)
    d = np.abs(sx - sy)
    return {"n": n, "qwk_score": cohen_kappa(sx.astype(int) - 1, sy.astype(int) - 1, 10, "quadratic") if n else NAN,
            "kappa_category": cohen_kappa(cx, cy, 3) if n else NAN, "spearman_score": safe_corr(sx, sy), "pearson_score": safe_corr(sx, sy, "pearson"),
            "mae_score": float(d.mean()) if n else NAN, "exact_category_agreement": float((cx == cy).mean()) if n else NAN,
            "within_1_point": float((d <= 1).mean()) if n else NAN, "within_2_points": float((d <= 2).mean()) if n else NAN}


def student_agreement(cons: pd.DataFrame, annotators=STUDENT_SHEETS, n_boot=1000, seed=42) -> pd.DataFrame:
    rows = []
    en, hi = cons[cons.language == "en"], cons[cons.language == "hi"]
    ann = [a for a in annotators if f"score_{a}" in cons.columns]
    a3 = [a for a in ann if a != "A4"]
    scopes = [("english", en, ann), ("english_A1-A3", en, a3), ("hindi", hi, a3), ("combined_A1-A3", cons, a3)]
    for name, sub, raters in scopes:
        sub = sub.reset_index(drop=True)
        if len(sub) < 2 or len(raters) < 2:
            continue
        S = sub[[f"score_{a}" for a in raters]].to_numpy(float)
        C = _codes(sub, [f"cat_{a}" for a in raters])
        g = sub["group_id"].to_numpy()
        rs = "+".join(raters)
        pairs = list(combinations(range(len(raters)), 2))

        def pw(idx, i, j, key):
            s, c = S[idx], C[idx]
            m = ~np.isnan(s[:, i]) & ~np.isnan(s[:, j])
            if m.sum() < 2:
                return NAN
            return _pair_metrics(s[m, i], s[m, j], c[m, i], c[m, j])[key]

        def mean_pw(idx, key):
            v = [pw(idx, i, j, key) for i, j in pairs]
            return float(np.nanmean(v)) if not np.isnan(v).all() else NAN
        head = {
            "fleiss_kappa_category": lambda idx: fleiss_kappa(C[idx], 3)[0],
            "krippendorff_alpha_ordinal_score": lambda idx: krippendorff_alpha(S[idx], "ordinal"),
            "krippendorff_alpha_interval_score": lambda idx: krippendorff_alpha(S[idx], "interval"),
            "krippendorff_alpha_nominal_category": lambda idx: krippendorff_alpha(np.where(C[idx] >= 0, C[idx], np.nan), "nominal"),
            "mean_pairwise_qwk_score": lambda idx: mean_pw(idx, "qwk_score"),
            "mean_pairwise_kappa_category": lambda idx: mean_pw(idx, "kappa_category"),
            "mean_pairwise_mae_score": lambda idx: mean_pw(idx, "mae_score"),
            "mean_pairwise_exact_category_agreement": lambda idx: mean_pw(idx, "exact_category_agreement"),
            "mean_pairwise_within_1_point": lambda idx: mean_pw(idx, "within_1_point"),
            "mean_pairwise_within_2_points": lambda idx: mean_pw(idx, "within_2_points")}
        full = np.arange(len(sub))
        for metric, fn in head.items():
            v = fn(full)
            n_used = fleiss_kappa(C, 3)[1] if metric == "fleiss_kappa_category" else int((~np.isnan(S)).any(axis=1).sum())
            lo, hi_ = group_bootstrap(g, fn, n_boot, seed) if metric != "krippendorff_alpha_interval_score" else (NAN, NAN)
            note = "" if v == v else "undefined (insufficient variation or data)"
            rows.append(_r("student_agreement", name, metric, n_used, v, lo, hi_, note, rater_set=rs))
        for i, j in pairs:
            m = ~np.isnan(S[:, i]) & ~np.isnan(S[:, j])
            pm = _pair_metrics(S[m, i], S[m, j], C[m, i], C[m, j])
            for key, v in pm.items():
                if key == "n":
                    continue
                lo = hi_ = NAN
                if key in ("qwk_score", "mae_score"):
                    lo, hi_ = group_bootstrap(g, lambda idx, i=i, j=j, key=key: pw(idx, i, j, key), n_boot, seed)
                rows.append(_r("student_agreement_pairwise", name, key, pm["n"], v, lo, hi_, "" if v == v else "undefined", rater_set=f"{raters[i]}-{raters[j]}"))
        # per-example dispersion summary
        sd = np.nanstd(S, axis=1, ddof=1) if S.shape[1] > 1 else np.full(len(sub), NAN)
        rng_ = np.nanmax(S, axis=1) - np.nanmin(S, axis=1)
        rows += [_r("per_example_dispersion", name, "mean_score_sd", len(sub), float(np.nanmean(sd)), rater_set=rs),
                 _r("per_example_dispersion", name, "mean_score_range", len(sub), float(np.nanmean(rng_)), rater_set=rs),
                 _r("per_example_dispersion", name, "share_range_ge_3", len(sub), float((rng_ >= 3).mean()), rater_set=rs),
                 _r("per_example_dispersion", name, "share_category_disagreement", len(sub), float(sub["student_category_disagreement"].mean()), rater_set=rs)]
    return pd.DataFrame(rows, columns=COLS)


def pairwise_matrix(cons, language, annotators, key="qwk_score") -> pd.DataFrame:
    """Annotator x annotator matrix of a pairwise statistic (for the heatmap)."""
    sub = cons[cons.language == language]
    M = pd.DataFrame(NAN, index=annotators, columns=annotators)
    for a in annotators:
        M.loc[a, a] = 1.0 if key in ("qwk_score", "kappa_category", "spearman_score") else 0.0
    for a, b in combinations(annotators, 2):
        sa, sb = sub[f"score_{a}"].to_numpy(float), sub[f"score_{b}"].to_numpy(float)
        m = ~np.isnan(sa) & ~np.isnan(sb)
        if m.sum() >= 2:
            ca, cb = _codes(sub[m], [f"cat_{a}", f"cat_{b}"]).T
            M.loc[a, b] = M.loc[b, a] = _pair_metrics(sa[m], sb[m], ca, cb)[key]
    return M


# ----------------------------------------------------------------------------- score/category comparison block
def compare_block(t_score, p_score, t_cat, p_cat, groups, ci=False, n_boot=1000, seed=42, bias_label="comparator - standard"):
    """Metrics of a comparator (p) against a standard (t). Returns list of (metric, value, lo, hi, n, note)."""
    out = []
    ts, ps = (np.asarray(t_score, float), np.asarray(p_score, float)) if t_score is not None else (None, None)
    groups = np.asarray(groups)
    if ts is not None and len(ts):
        m = ~np.isnan(ts) & ~np.isnan(ps)
        ts, ps, g = ts[m], ps[m], groups[m]
        n = len(ts)
        if n:
            rm = regression_metrics(ts, ps)
            vals = {"mae": rm["mae"], "rmse": rm["rmse"], "bias": float(np.mean(ps - ts)), "spearman": rm["spearman"] if n >= MIN_N_CORR else NAN,
                    "pearson": rm["pearson"] if n >= MIN_N_CORR else NAN, "within_1_point": float(np.mean(np.abs(ps - ts) <= 1)),
                    "share_comparator_higher": float(np.mean(ps > ts)), "share_comparator_lower": float(np.mean(ps < ts)), "share_within_0.5": float(np.mean(np.abs(ps - ts) <= 0.5))}
            fns = {"mae": lambda i: np.mean(np.abs(ps[i] - ts[i])), "bias": lambda i: np.mean(ps[i] - ts[i]), "spearman": lambda i: safe_corr(ts[i], ps[i])}
            for k, v in vals.items():
                lo = hi = NAN
                if ci and k in fns and n >= MIN_N_BIAS:
                    lo, hi = group_bootstrap(g, fns[k], n_boot, seed)
                note = f"bias = {bias_label}" if k == "bias" else ("n<%d: correlation not reported" % MIN_N_CORR if k in ("spearman", "pearson") and n < MIN_N_CORR else "")
                out.append((k, v, lo, hi, n, note))
            if n >= 6 and np.any(ps != ts):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out.append(("wilcoxon_signed_rank_p", float(wilcoxon(ps, ts).pvalue), NAN, NAN, n, "two-sided, paired; asks whether the median difference is zero"))
    if t_cat is not None:
        tc, pc = np.asarray(t_cat, object), np.asarray(p_cat, object)
        m = np.array([isinstance(a, str) and isinstance(b, str) for a, b in zip(tc, pc)], bool)
        tc, pc, g = tc[m], pc[m], groups[m]
        n = len(tc)
        if n:
            acc = lambda i: float(np.mean(tc[i] == pc[i]))  # noqa: E731
            lo, hi = group_bootstrap(g, acc, n_boot, seed) if ci and n >= MIN_N_BIAS else (NAN, NAN)
            kap = cohen_kappa([CAT_CODE[x] for x in tc], [CAT_CODE[x] for x in pc], 3)
            out += [("category_accuracy", acc(np.arange(n)), lo, hi, n, ""),
                    ("category_macro_f1", float(f1_score(tc, pc, labels=list(CATEGORIES), average="macro", zero_division=0)), NAN, NAN, n, "standard = truth; labels with no support count as 0"),
                    ("category_weighted_f1", float(f1_score(tc, pc, labels=list(CATEGORIES), average="weighted", zero_division=0)), NAN, NAN, n, ""),
                    ("category_kappa", kap, NAN, NAN, n, "" if kap == kap else "undefined (single category)")]
    return out


def confusion_rows(truth, pred, comparison, subset):
    t = [a for a, b in zip(truth, pred) if isinstance(a, str) and isinstance(b, str)]
    p = [b for a, b in zip(truth, pred) if isinstance(a, str) and isinstance(b, str)]
    if not t:
        return []
    cm = confusion_matrix(t, p, labels=list(CATEGORIES))
    return [{"comparison": comparison, "subset": subset, "truth": CATEGORIES[i], "predicted": CATEGORIES[j], "count": int(cm[i, j])} for i in range(3) for j in range(3)]


def _emit(rows, scope, subset, comparator, variant, block, rater_set=""):
    for metric, v, lo, hi, n, note in block:
        rows.append(_r(scope, subset, metric, n, v, lo, hi, note, comparator, variant, rater_set))


def expert_comparison(cons, n_boot=1000, seed=42):
    e = cons[cons.expert_reviewed].reset_index(drop=True)
    rows, conf = [], []
    if e.empty:
        return pd.DataFrame(columns=COLS), pd.DataFrame()
    for sname, mask in subset_masks(e, ("all", "fixed_test", "supplemental_train_val", "english", "hindi")).items():
        sub = e[mask]
        if sub.empty:
            continue
        label = "expert_" + sname
        for comp, sc_col, cat_col in (("student_mean", "student_mean", "student_mean_category"), ("student_median", "student_median", "student_median_category"),
                                      ("student_majority_vote", None, "student_majority_category")):
            blk = compare_block(sub.expert_score if sc_col else None, sub[sc_col] if sc_col else None, sub.expert_category, sub[cat_col], sub.group_id,
                                ci=True, n_boot=n_boot, seed=seed, bias_label="student - expert")
            _emit(rows, "student_vs_expert", label, comp, "", blk)
            conf += confusion_rows(sub.expert_category, sub[cat_col], f"expert(truth) vs {comp}(pred)", label)
        if sname in ("all", "english", "hindi"):
            rows.append(_r("student_vs_expert", label, "n_category_ties_excluded", len(sub), int(sub.student_majority_tie.sum()), note="pairs with a tied student majority vote are excluded from majority-vote category metrics", comparator="student_majority_vote"))
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(conf)


def reference_comparison(cons, ind, n_boot=1000, seed=42):
    """AI/author reference labels vs student mean / median / majority / individual annotations / expert."""
    rows, conf = [], []
    stu = ind[ind.annotator_role == "student"].reset_index(drop=True)
    exp = ind[ind.annotator_role == "expert"].reset_index(drop=True)
    ci_sets = {"all", "english", "hindi", "fixed_test", "supplemental_train_val"}
    for sname in subset_masks(cons):
        c = cons[subset_masks(cons)[sname]].reset_index(drop=True)
        if c.empty:
            continue
        for comp in ("student_mean", "student_median", "student_majority_vote", "student_individual", "expert"):
            if comp == "student_individual":
                d = stu[subset_masks(stu)[sname]]
                ts, cat, grp = d.human_score, d.human_category, d.group_id
            elif comp == "expert":
                d = exp[subset_masks(exp)[sname]]
                ts, cat, grp = d.human_score, d.human_category, d.group_id
            else:
                d = c
                ts = {"student_mean": d.student_mean, "student_median": d.student_median}.get(comp)
                cat = {"student_mean": d.student_mean_category, "student_median": d.student_median_category, "student_majority_vote": d.student_majority_category}[comp]
                grp = d.group_id
            if len(d) == 0:
                continue
            ref_s, ref_o = d.ref_score, d.ref_category_original
            ref_h = d.ref_category_harmonized if "ref_category_harmonized" in d else d.ref_score.map(lambda s: None if s != s else ("Not Preserved" if s < 3.5 else "Preserved" if s >= 6.5 else "Partially Preserved"))
            ci = sname in ci_sets and comp in ("student_mean", "student_individual", "expert")
            # score metrics (reference is the comparator, human the standard): bias = reference - human
            if ts is not None:
                _emit(rows, "reference_vs_human", sname, comp, "score", compare_block(ts, ref_s, None, None, grp, ci, n_boot, seed, "reference - human"))
            for variant, rc in (("original_repo_thresholds_4_7.5", ref_o), ("harmonized_human_rubric_3.5_6.5", ref_h)):
                _emit(rows, "reference_vs_human", sname, comp, variant, compare_block(None, None, cat, rc, grp, ci, n_boot, seed))
                conf += confusion_rows(list(cat), list(rc), f"{comp}(truth) vs reference[{variant}](pred)", sname)
            if comp in ("student_mean", "student_individual", "expert") and ts is not None:
                for nm, s in (("reference", ref_s), ("human", ts)):
                    s = pd.Series(s).astype(float)
                    for k, v in (("mean", s.mean()), ("sd", s.std(ddof=1)), ("min", s.min()), ("median", s.median()), ("max", s.max())):
                        rows.append(_r("score_distribution", sname, f"{nm}_{k}", len(s), float(v), comparator=comp))
    return pd.DataFrame(rows, columns=COLS), pd.DataFrame(conf)


# ----------------------------------------------------------------------------- subgroup analysis
def subgroup_analysis(cons) -> pd.DataFrame:
    rows = []
    c = cons.copy()
    c["student_modal_change"] = c["student_modal_change"].fillna("(no unique modal label)")
    for dim in ("language", "complexity", "legal_domain", "reference_label_type", "dataset_split", "student_modal_change", "ref_category_original"):
        for val, sub in c.groupby(dim):
            n = len(sub)
            base = dict(dimension=dim, group=val)

            def add(metric, v, nn=n, note=""):
                rows.append({**base, "metric": metric, "n": nn, "value": v, "note": note})
            add("n_pairs", n)
            add("student_sd_mean", float(sub.student_sd.mean()))
            add("student_range_mean", float(sub.student_range.mean()))
            add("share_student_range_ge_3", float(sub.student_high_disagreement.mean()))
            add("share_student_category_disagreement", float(sub.student_category_disagreement.mean()))
            S = sub[[f"score_{a}" for a in STUDENT_SHEETS if f"score_{a}" in sub]].to_numpy(float)
            add("krippendorff_alpha_ordinal", krippendorff_alpha(S, "ordinal") if n >= MIN_N_CORR else NAN, note="" if n >= MIN_N_CORR else f"n<{MIN_N_CORR}")
            if n >= MIN_N_BIAS:
                rm = regression_metrics(sub.student_mean, sub.ref_score)
                add("reference_vs_student_mean_mae", rm["mae"])
                add("reference_vs_student_mean_bias", float((sub.ref_score - sub.student_mean).mean()), note="reference - student mean")
                add("reference_vs_student_mean_spearman", rm["spearman"] if n >= MIN_N_CORR else NAN, note="" if n >= MIN_N_CORR else f"n<{MIN_N_CORR}")
                add("reference_harmonized_category_accuracy", float((sub.ref_category_harmonized == sub.student_mean_category).mean()))
            else:
                add("reference_vs_student_mean_mae", NAN, note=f"n<{MIN_N_BIAS}")
            e = sub[sub.expert_reviewed]
            if len(e) >= MIN_N_BIAS:
                add("student_mean_vs_expert_mae", float((e.student_mean - e.expert_score).abs().mean()), len(e))
                add("student_mean_vs_expert_bias", float((e.student_mean - e.expert_score).mean()), len(e), "student - expert")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- semantic label analysis
def _lab_code(x):
    return LABELS.index(x) if x in LABELS else -1


def semantic_analysis(cons, ind):
    rows, extra = [], []
    add = lambda section, subset, label, metric, n, v, note="": rows.append({"section": section, "subset": subset, "label": label, "metric": metric, "n": int(n), "value": v, "note": note})  # noqa: E731
    stu, exp = ind[ind.annotator_role == "student"], ind[ind.annotator_role == "expert"]
    for role, d in (("student", stu), ("expert", exp)):
        for sname, mask in subset_masks(d, ("all", "english", "hindi")).items():
            sub = d[mask]
            vc = sub.semantic_change_type.fillna("(blank)").value_counts()
            for lab, k in vc.items():
                add("label_frequency", f"{role}_{sname}", lab, "count", len(sub), int(k))
                add("label_frequency", f"{role}_{sname}", lab, "share", len(sub), float(k / len(sub)))
    ann = [a for a in STUDENT_SHEETS if f"change_{a}" in cons]
    for sname, sub in (("all", cons), ("english", cons[cons.language == "en"]), ("hindi", cons[cons.language == "hi"]), ("changed_pairs_student_mean_lt_6.5", cons[cons.student_mean < 6.5])):
        if len(sub) < 2:
            continue
        cols = [f"change_{a}" for a in ann if not (sname == "hindi" and a == "A4")]
        C = np.array([[_lab_code(v) for v in sub[c]] for c in cols]).T
        M = np.where(C >= 0, C, np.nan).astype(float)
        pe = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                m = (C[:, i] >= 0) & (C[:, j] >= 0)
                if m.sum():
                    pe.append(float((C[m, i] == C[m, j]).mean()))
        f, nf = fleiss_kappa(C, len(LABELS))
        add("student_label_agreement", sname, "", "mean_pairwise_exact_agreement", len(sub), float(np.mean(pe)) if pe else NAN)
        add("student_label_agreement", sname, "", "fleiss_kappa_nominal", nf, f, "pairs where every rater gave a label")
        add("student_label_agreement", sname, "", "krippendorff_alpha_nominal", len(sub), krippendorff_alpha(M, "nominal"))
    e = cons[cons.expert_reviewed & cons.student_modal_change.notna() & cons.expert_semantic_change.notna()]
    for sname, mask in subset_masks(e, ("all", "english", "hindi", "fixed_test")).items():
        sub = e[mask]
        if len(sub) == 0:
            continue
        add("student_expert_label_agreement", sname, "", "exact_agreement_modal_student_vs_expert", len(sub), float((sub.student_modal_change == sub.expert_semantic_change).mean()), "pairs with a unique student modal label")
        add("student_expert_label_agreement", sname, "", "cohen_kappa_nominal", len(sub), cohen_kappa([_lab_code(x) for x in sub.student_modal_change], [_lab_code(x) for x in sub.expert_semantic_change], len(LABELS)) if len(sub) >= MIN_N_BIAS else NAN, "" if len(sub) >= MIN_N_BIAS else f"n<{MIN_N_BIAS}")
    add("student_expert_label_agreement", "all", "", "n_expert_pairs_without_unique_student_modal_label", int(cons.expert_reviewed.sum()), int((cons.expert_reviewed & cons.student_modal_change.isna()).sum()))
    # per human label: disagreement and reference mismatch
    for lab, sub in cons.dropna(subset=["student_modal_change"]).groupby("student_modal_change"):
        add("by_modal_student_label", "all", lab, "n_pairs", len(sub), len(sub))
        add("by_modal_student_label", "all", lab, "mean_student_sd", len(sub), float(sub.student_sd.mean()))
        add("by_modal_student_label", "all", lab, "mean_reference_minus_student_mean", len(sub), float((sub.ref_score - sub.student_mean).mean()), "reference - student mean")
    # human taxonomy vs dataset (reference) taxonomy
    for lab, ds in HUMAN_TO_DATASET.items():
        add("taxonomy_mapping", "", lab, "dataset_labels", 0, ";".join(sorted(ds)) if ds else "(no dataset equivalent)")
    mod = cons.dropna(subset=["student_modal_change"])
    has_ref = mod[mod.ref_error_labels.astype(str).str.len() > 0]
    mapped = has_ref[~has_ref.student_modal_change.isin(NO_DATASET_EQUIVALENT | {"None (preserved)"})]
    hit = [bool(HUMAN_TO_DATASET[m] & set(r.split("|"))) for m, r in zip(mapped.student_modal_change, mapped.ref_error_labels)]
    add("reference_vs_student_label", "all", "", "share_student_modal_label_in_reference_error_labels", len(mapped), float(np.mean(hit)) if hit else NAN,
        "pairs with reference error labels and a student modal label that has a dataset equivalent")
    add("reference_vs_student_label", "all", "", "n_unmappable_student_modal_label", len(has_ref) - len(mapped), len(has_ref) - len(mapped), "student label without dataset equivalent or 'None (preserved)'")
    noref = mod[mod.ref_error_labels.astype(str).str.len() == 0]
    add("reference_vs_student_label", "all", "", "share_student_none_when_reference_has_no_error", len(noref), float((noref.student_modal_change == "None (preserved)").mean()) if len(noref) else NAN)
    for lab in sorted({x for r in has_ref.ref_error_labels for x in r.split("|") if x}):
        sub = has_ref[has_ref.ref_error_labels.map(lambda r: lab in r.split("|"))]
        h = [lab in HUMAN_TO_DATASET.get(m, set()) for m in sub.student_modal_change]
        add("reference_vs_student_label", "all", lab, "share_student_modal_matches_this_reference_label", len(sub), float(np.mean(h)), "reference labels are AI-generated; multi-label")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- qualitative tables
def qualitative_tables(cons, ind, top=10) -> pd.DataFrame:
    stu, exp = ind[ind.annotator_role == "student"], ind[ind.annotator_role == "expert"].set_index("pair_id")

    def expl(pid):
        d = stu[stu.pair_id == pid].sort_values("annotator_id")
        return " || ".join(f"{r.annotator_id}: {r.human_score:g} {r.semantic_change_type or '-'} | {r.explanation or ''}" for r in d.itertuples())
    out = []

    def add(table, df):
        for r in df.itertuples():
            e = exp.loc[r.pair_id] if r.pair_id in exp.index else None
            out.append({"table": table, "pair_id": r.pair_id, "source_pair_id": r.source_pair_id, "language": r.language, "dataset_split": r.dataset_split, "complexity": r.complexity,
                        "legal_domain": r.legal_domain, "original": r.original, "simplified": r.simplified, "student_scores": ",".join(f"{getattr(r, f'score_{a}'):g}" for a in STUDENT_SHEETS if getattr(r, f"score_{a}") == getattr(r, f"score_{a}")),
                        "student_mean": r.student_mean, "student_sd": r.student_sd, "student_explanations": expl(r.pair_id), "reference_score": r.ref_score,
                        "reference_category_original": r.ref_category_original, "reference_error_labels": r.ref_error_labels,
                        "expert_score": r.expert_score, "expert_category": r.expert_category, "expert_semantic_change": r.expert_semantic_change,
                        "expert_reasoning": e.explanation if e is not None else None, "expert_ambiguity_or_concern": e.notes if e is not None else None,
                        "abs_student_expert_diff": abs(r.student_mean - r.expert_score) if r.expert_score == r.expert_score else np.nan,
                        "abs_reference_student_diff": abs(r.ref_score - r.student_mean)})
    ec = cons[cons.expert_reviewed].copy()
    ec["_d"] = (ec.student_mean - ec.expert_score).abs()
    add("expert_vs_student_largest_score_gaps", ec.sort_values("_d", ascending=False).head(top))
    add("expert_vs_student_category_disagreement", ec[ec.student_mean_category != ec.expert_category])
    add("expert_vs_student_semantic_label_disagreement", ec[ec.student_modal_change.notna() & (ec.student_modal_change != ec.expert_semantic_change)])
    cc = cons.copy()
    add("student_high_disagreement", cc.sort_values(["student_range", "student_sd"], ascending=False).head(top))
    cc["_r"] = (cc.ref_score - cc.student_mean).abs()
    add("reference_vs_student_largest_gaps", cc.sort_values("_r", ascending=False).head(top))
    return pd.DataFrame(out)
