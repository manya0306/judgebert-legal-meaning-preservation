"""Inter-annotator agreement for multi-annotator CSVs (pair_id, annotator_id, lmp_score, error_labels)."""
from __future__ import annotations
import itertools, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from .schema import ERROR_LABELS
from ..config import score_to_category

def pairwise_score_agreement(ann: pd.DataFrame) -> pd.DataFrame:
    wide = ann.pivot(index="pair_id", columns="annotator_id", values="lmp_score")
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) < 3 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
            continue
        rows.append({"annotators": f"{a}~{b}", "n": len(sub),
                     "pearson": pearsonr(sub[a], sub[b])[0], "spearman": spearmanr(sub[a], sub[b])[0]})
    return pd.DataFrame(rows)

def category_kappa(ann: pd.DataFrame) -> pd.DataFrame:
    ann = ann.copy()
    ann["cat"] = ann["lmp_score"].map(score_to_category)
    wide = ann.pivot(index="pair_id", columns="annotator_id", values="cat")
    rows = []
    for a, b in itertools.combinations(wide.columns, 2):
        sub = wide[[a, b]].dropna()
        if len(sub):
            rows.append({"annotators": f"{a}~{b}", "n": len(sub), "kappa": cohen_kappa_score(sub[a], sub[b])})
    return pd.DataFrame(rows)

def error_label_kappa(ann: pd.DataFrame) -> pd.DataFrame:
    """Per-label Cohen's kappa between each annotator pair (binary present/absent)."""
    rows = []
    for a, b in itertools.combinations(sorted(ann["annotator_id"].unique()), 2):
        sa = ann[ann.annotator_id == a].set_index("pair_id")["error_labels"].fillna("")
        sb = ann[ann.annotator_id == b].set_index("pair_id")["error_labels"].fillna("")
        common = sa.index.intersection(sb.index)
        for lab in ERROR_LABELS:
            ya = [int(lab in str(sa[i]).split("|")) for i in common]
            yb = [int(lab in str(sb[i]).split("|")) for i in common]
            if len(set(ya) | set(yb)) < 2:
                continue
            rows.append({"annotators": f"{a}~{b}", "label": lab, "n": len(common), "kappa": cohen_kappa_score(ya, yb)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ v2: weighted kappa and Krippendorff's alpha
def weighted_kappa(a, b, weights: str = "quadratic", labels=None) -> float:
    """Cohen's kappa with linear/quadratic weights for two raters on ordinal data (use integer-rounded scores or ordered categories)."""
    return float(cohen_kappa_score(a, b, weights=weights, labels=labels))


def krippendorff_alpha(matrix, level: str = "interval") -> float:
    """Krippendorff's alpha. `matrix`: raters x units array with np.nan for missing. level: nominal | ordinal | interval.
    Standard coincidence-matrix implementation (Krippendorff 2011); units with <2 ratings are ignored."""
    M = np.asarray(matrix, float)
    units = [M[:, u][~np.isnan(M[:, u])] for u in range(M.shape[1])]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    vals = np.array(sorted({v for u in units for v in u}))
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    O = np.zeros((k, k))
    for u in units:
        m = len(u)
        for i in range(m):
            for j in range(m):
                if i != j:
                    O[idx[u[i]], idx[u[j]]] += 1.0 / (m - 1)
    n_c = O.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    if level == "nominal":
        delta = (vals[:, None] != vals[None, :]).astype(float)
    elif level == "interval":
        delta = (vals[:, None] - vals[None, :]) ** 2
    elif level == "ordinal":
        cum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for c in range(k):
            for d in range(k):
                if c != d:
                    lo, hi = min(c, d), max(c, d)
                    delta[c, d] = (cum[hi] - (n_c[hi] + n_c[lo]) / 2 - (cum[lo - 1] if lo > 0 else 0)) ** 2
    else:
        raise ValueError("level must be nominal, ordinal or interval")
    Do = (O * delta).sum() / n
    De = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else float("nan")


def agreement_report(ann: pd.DataFrame, annotator_ids: list[str] | None = None, title: str = "") -> dict:
    """Agreement between the chosen annotators (default: all). Reports weighted kappa for every annotator pair and alpha overall.
    The caller must pass only genuinely separate annotation records; the report states annotator types so AI-vs-AI is never presented
    as human agreement."""
    ann = ann.copy()
    if annotator_ids:
        ann = ann[ann.annotator_id.isin(annotator_ids)]
    types = ann.groupby("annotator_id")["annotator_type"].first().to_dict() if "annotator_type" in ann.columns else {}
    cat_order = {"Not Preserved": 0, "Partially Preserved": 1, "Preserved": 2}
    ann["cat_i"] = ann["lmp_score"].map(score_to_category).map(cat_order)
    wide_s = ann.pivot_table(index="pair_id", columns="annotator_id", values="lmp_score", aggfunc="first")
    wide_c = ann.pivot_table(index="pair_id", columns="annotator_id", values="cat_i", aggfunc="first")
    rep = {"title": title, "annotator_types": types, "n_units_with_2plus_ratings": int((wide_s.notna().sum(axis=1) >= 2).sum()), "n_units_total": int(len(wide_s)), "pairs": []}
    kinds = set(types.values())
    rep["warning"] = ("AI-vs-AI agreement: NOT human inter-annotator agreement; model errors are correlated." if "human" not in kinds
                      else "Contains human annotators.")
    for a, b in itertools.combinations(wide_s.columns, 2):
        sub = wide_s[[a, b]].dropna()
        if len(sub) < 5:
            continue
        sc_a, sc_b = sub[a].round().astype(int), sub[b].round().astype(int)
        cs = wide_c.loc[sub.index, [a, b]]
        rep["pairs"].append({"annotators": f"{a}~{b}", "n": len(sub),
                             "weighted_kappa_score_quadratic": weighted_kappa(sc_a, sc_b, "quadratic", labels=list(range(1, 11))) if sc_a.nunique() > 1 or sc_b.nunique() > 1 else float("nan"),
                             "weighted_kappa_category_linear": weighted_kappa(cs[a].astype(int), cs[b].astype(int), "linear", labels=[0, 1, 2]),
                             "category_accuracy_agreement": float((cs[a] == cs[b]).mean()),
                             "mean_abs_score_diff": float((sub[a] - sub[b]).abs().mean())})
    if wide_s.shape[1] >= 2:
        rep["krippendorff_alpha_score_interval"] = krippendorff_alpha(wide_s.T.values, "interval")
        rep["krippendorff_alpha_category_ordinal"] = krippendorff_alpha(wide_c.T.values, "ordinal")
    return rep
