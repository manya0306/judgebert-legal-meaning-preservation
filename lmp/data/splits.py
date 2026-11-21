"""Leakage-aware splitting.

Unit of splitting = GROUP, never row. A group contains everything that must not be separated:
  * all simplifications of one original sentence (variants),
  * its English/Hindi translation counterparts (same `concept_id` / `group_id`),
  * groups that are NEAR-DUPLICATES of each other (merged by connected components over text similarity),
  * augmented rows (they inherit their parent's group and exist only in the training portion).
Backward compatible: `document_split`, `domain_holdout_split`, `_group_ids` keep their old behaviour for frames
that have no `group_id` column.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


# ------------------------------------------------------------------ grouping
def _base_groups(df: pd.DataFrame) -> pd.Series:
    if "group_id" in df.columns and df["group_id"].astype("string").str.strip().replace("", pd.NA).notna().all():
        return df["group_id"].astype(str)
    if "template_group_id" in df.columns:
        explicit = df["template_group_id"].astype("string").str.strip()
        valid = explicit.notna() & explicit.ne("")
    else:
        valid = pd.Series(False, index=df.index)
        explicit = pd.Series(pd.NA, index=df.index, dtype="string")
    normalized = df["source_doc_id"].astype(str).str.replace(r"^SYN-(?:EN|HI)-", "SYN-", regex=True)
    groups = "document:" + normalized
    groups.loc[valid] = "template:" + explicit.loc[valid]
    return groups


def _group_ids(df: pd.DataFrame) -> pd.Series:
    """Backward-compatible name: shared grouping IDs, bilingual counterparts together."""
    return _base_groups(df)


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def merge_near_duplicate_groups(df: pd.DataFrame, threshold: float = 0.80) -> pd.Series:
    """Return a 'split_group' Series: base groups merged when any original sentence in one group is a near-duplicate (char n-gram
    TF-IDF cosine >= threshold, same language) of an original sentence in another group."""
    base = _base_groups(df)
    uf = _UF(sorted(base.unique()))
    for lang, sub in df.assign(_g=base).groupby("language"):
        uniq = sub.drop_duplicates(["original", "_g"])
        if len(uniq) < 2:
            continue
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(uniq.original)
        S = (X @ X.T).toarray()
        np.fill_diagonal(S, 0)
        g = uniq["_g"].tolist()
        for i, j in zip(*np.where(np.triu(S) >= threshold)):
            uf.union(g[i], g[j])
    return base.map(uf.find).rename("split_group")


# ------------------------------------------------------------------ splitting
def grouped_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.2, seed=42, near_dup_threshold: float = 0.80):
    """Domain-stratified, group-aware train/val/test split. Returns (train, val, test) with a `split_group` column."""
    df = df.reset_index(drop=True).copy()
    df["split_group"] = merge_near_duplicate_groups(df, near_dup_threshold).values
    gdom = df.groupby("split_group")["legal_domain"].agg(lambda s: s.value_counts().index[0])
    rng = np.random.RandomState(seed)
    test_g, val_g = set(), set()
    for dom, gs in gdom.groupby(gdom):
        names = sorted(gs.index)
        rng.shuffle(names)
        n = len(names)
        n_te = int(round(n * test_frac)) if n >= 3 else 0
        n_va = int(round(n * val_frac)) if n >= 4 else 0
        test_g |= set(names[:n_te])
        val_g |= set(names[n_te:n_te + n_va])
    if not test_g or not val_g:       # tiny datasets: fall back to global group shuffle
        names = sorted(gdom.index)
        rng.shuffle(names)
        n = len(names)
        n_te, n_va = max(1, int(round(n * test_frac))), max(1, int(round(n * val_frac)))
        if n_te + n_va >= n:
            raise ValueError(f"Not enough independent groups ({n}) for the requested split.")
        test_g, val_g = set(names[:n_te]), set(names[n_te:n_te + n_va])
    sg = df["split_group"]
    tr, va, te = (df[~sg.isin(test_g | val_g)].reset_index(drop=True), df[sg.isin(val_g)].reset_index(drop=True),
                  df[sg.isin(test_g)].reset_index(drop=True))
    assert_no_leakage(tr, va, te)
    return tr, va, te


def document_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.2, seed=42):
    """Original-API split (kept for backward compatibility); now delegates to the domain-stratified grouped split."""
    return grouped_split(df, val_frac, test_frac, seed)


def grouped_kfold(df: pd.DataFrame, k: int = 5, seed: int = 42, near_dup_threshold: float = 0.80):
    """Yield (fold, train_idx, test_idx) over a reset-index copy. Stratified by legal domain at group level, no group in two folds."""
    df = df.reset_index(drop=True)
    sg = merge_near_duplicate_groups(df, near_dup_threshold)
    dom = df.groupby(sg)["legal_domain"].agg(lambda s: s.value_counts().index[0])
    y = sg.map(dom).values
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (a, b) in enumerate(sgkf.split(df, y, groups=sg.values)):
        yield f, a, b


def repeated_grouped_kfold(df: pd.DataFrame, k: int = 5, repeats: int = 3, seed: int = 42):
    for r in range(repeats):
        for f, a, b in grouped_kfold(df, k, seed + r):
            yield r, f, a, b


def domain_holdout_split(df: pd.DataFrame, held_out_domain: str, val_frac=0.15, seed=42):
    """Train on all other domains and test on the unseen domain. Validation = a group-level sample of the remaining domains."""
    df = df.reset_index(drop=True)
    test = df[df["legal_domain"] == held_out_domain].reset_index(drop=True)
    if test.empty:
        raise ValueError(f"Unknown or empty held-out domain: {held_out_domain}")
    rest = df[df["legal_domain"] != held_out_domain].reset_index(drop=True)
    rest["split_group"] = merge_near_duplicate_groups(rest).values
    test["split_group"] = merge_near_duplicate_groups(test).values
    names = sorted(rest["split_group"].unique())
    np.random.RandomState(seed).shuffle(names)
    n_val = max(1, int(round(len(names) * val_frac)))
    val_g = set(names[:n_val])
    va = rest[rest.split_group.isin(val_g)].reset_index(drop=True)
    tr = rest[~rest.split_group.isin(val_g)].reset_index(drop=True)
    return tr, va, test


# ------------------------------------------------------------------ leakage checks and reporting
def assert_no_leakage(*splits: pd.DataFrame) -> None:
    """Raise AssertionError on any group, pair_id, concept or exact-original-text overlap between splits."""
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            if len(a) == 0 or len(b) == 0:
                continue
            col = "split_group" if "split_group" in a.columns else None
            ga = set(a[col]) if col else set(_base_groups(a))
            gb = set(b[col]) if col else set(_base_groups(b))
            assert not (ga & gb), f"group leakage between splits {i} and {j}: {sorted(ga & gb)[:3]}"
            assert not (set(a.pair_id) & set(b.pair_id)), "pair_id leakage"
            if "concept_id" in a.columns:
                ca, cb = set(a.concept_id.fillna("")) - {""}, set(b.concept_id.fillna("")) - {""}
                assert not (ca & cb), f"concept (translation/paraphrase) leakage: {sorted(ca & cb)[:3]}"
            oa = set(zip(a.language, a.original.str.lower().str.strip()))
            ob = set(zip(b.language, b.original.str.lower().str.strip()))
            assert not (oa & ob), "identical original sentence appears in two splits"


def split_report(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    rep = {}
    for name, d in (("train", train), ("val", val), ("test", test)):
        rep[name] = {"n_pairs": len(d), "n_groups": int(d["split_group"].nunique()) if "split_group" in d else int(_base_groups(d).nunique()),
                     "language": d.language.value_counts().to_dict(), "category": d.category.value_counts().to_dict(),
                     "domain": d.legal_domain.value_counts().to_dict()}
    try:
        assert_no_leakage(train, val, test)
        rep["leakage_check"] = "passed (no shared group, pair_id, concept or identical original sentence)"
    except AssertionError as e:
        rep["leakage_check"] = f"FAILED: {e}"
    n_groups = sum(rep[k]["n_groups"] for k in ("train", "val", "test"))
    rep["statistical_power_note"] = (f"{n_groups} independent groups; a test set of {rep['test']['n_pairs']} pairs from {rep['test']['n_groups']} groups "
                                     "supports only coarse conclusions - prefer grouped cross-validation.")
    return rep
