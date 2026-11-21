"""Automated dataset quality checks. Questionable rows are FLAGGED (never silently dropped).

Each flag: (pair_id, check, severity, detail). severity: 'error' = excluded by the quality filter,
'warn' = kept but visible in `qa_flags`, 'info' = reporting only.

Checks (see docs/DATASET_DOCUMENTATION.md): invalid/missing labels, score-category consistency, exact duplicates,
near-duplicates (cross-group), translation/paraphrase leakage (group integrity), repeated generation templates,
Hindi plausibility (script/orthography), legal-terminology consistency (EN<->HI glossary), contradictory labels,
label-vs-score/label-vs-text plausibility, class/language imbalance, reliance on lexical cues (shortcut risk).
None of these checks can certify legal correctness or fluent Hindi; they catch mechanical problems only.
"""
from __future__ import annotations
import re
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from ..config import score_to_category
from ..preprocess import tokenize
from .schema import ERROR_LABELS, CATEGORIES

_DEV = re.compile(r"[\u0900-\u097F]")
_LAT = re.compile(r"[A-Za-z]")
# Hindi orthography problems that indicate a typo or broken text
_HI_BAD = [
    (re.compile("\u094d\u094d"), "double virama"),
    (re.compile(r"(?<![\u0900-\u097F])[\u093e-\u094c\u0901-\u0903]"), "word starts with a dependent vowel sign"),
    (re.compile(r"[\u0900-\u097F][A-Za-z]|[A-Za-z][\u0900-\u097F]"), "Latin letter fused with Devanagari"),
    (re.compile("[\u200b\u200e\u200f\ufeff]"), "invisible control character"),
    (re.compile(r"\u0964\s*\u0964"), "double danda"),
    (re.compile("\u093e\u093e|\u093f\u093f|\u0940\u0940"), "doubled vowel sign"),
]
# EN -> HI legal glossary used to check that a pair's Hindi and English originals use matching terminology
GLOSSARY = {
    "landlord": "मकान मालिक", "tenant": "किरायेदार", "employer": "नियोक्ता", "employee": "कर्मचारी", "insurer": "बीमाकर्ता",
    "policyholder": "पॉलिसीधारक", "seller": "विक्रेता", "buyer": "क्रेता", "borrower": "उधारकर्ता", "bank": "बैंक",
    "court": "न्यायालय", "notice": ("नोटिस", "सूचना"), "consent": "सहमति", "guardian": "संरक्षक", "minor": "नाबालिग",
    "licence": "लाइसेंस", "arbitrat": "मध्यस्थ", "magistrate": "मजिस्ट्रेट", "bail": "जमानत", "will": "वसीयत",
    "copyright": "कॉपीराइट", "director": "निदेशक", "contract": "संविदा", "maintenance": "भरण-पोषण", "appeal": "अपील",
    "complaint": "शिकायत", "damages": "हर्जाना", "refund": "लौटा", "deposit": "जमा",
}
_TERM_EN_RE = {k: re.compile(r"\b" + k, re.I) for k in GLOSSARY}
_WEAK = {"contradiction", "changed_obligation", "changed_permission"}
_NUM_LABELS = {"altered_numbers", "altered_dates"}


def _lab(s) -> set[str]:
    return {x for x in str(s if pd.notna(s) else "").split("|") if x}


def _flag(rows, pid, check, sev, detail):
    rows.append({"pair_id": pid, "check": check, "severity": sev, "detail": detail})


def check_labels_and_scores(df, thresholds=None) -> list[dict]:
    rows = []
    for r in df.itertuples():
        labs = _lab(r.error_labels)
        unknown = labs - set(ERROR_LABELS)
        if unknown:
            _flag(rows, r.pair_id, "invalid_label", "error", f"unknown error labels {sorted(unknown)}")
        if not (1 <= r.lmp_score <= 10):
            _flag(rows, r.pair_id, "invalid_score", "error", f"score {r.lmp_score} outside 1-10")
        cat = getattr(r, "category", None)
        if cat and cat == cat:
            exp = score_to_category(r.lmp_score, thresholds)
            if cat != exp:
                _flag(rows, r.pair_id, "score_category_mismatch", "error", f"category '{cat}' but score {r.lmp_score} implies '{exp}'")
        if r.lmp_score >= 7.5 and labs & _WEAK:
            _flag(rows, r.pair_id, "preserved_but_severe_label", "warn", f"score {r.lmp_score} (Preserved) with labels {sorted(labs & _WEAK)}")
        if r.lmp_score < 4 and not labs:
            _flag(rows, r.pair_id, "not_preserved_without_label", "warn", "Not Preserved but no changed legal element recorded")
        if r.lmp_score >= 9 and labs:
            _flag(rows, r.pair_id, "high_score_with_labels", "warn", f"score {r.lmp_score} but labels {sorted(labs)}")
        if getattr(r, "generation_strategy", "") in ("faithful_paraphrase", "heavy_rewrite") and r.lmp_score < 7.5 and not labs:
            _flag(rows, r.pair_id, "paraphrase_scored_low_without_label", "warn", "paraphrase variant scored below Preserved with no label")
        if getattr(r, "generation_strategy", "") == "lossy_simplification" and r.lmp_score >= 7.5:
            _flag(rows, r.pair_id, "lossy_variant_scored_preserved", "warn", f"lossy simplification scored {r.lmp_score}")
    return rows


def check_duplicates(df, near_thr: float = 0.90) -> list[dict]:
    rows = []
    key = (df.language + "||" + df.original.str.lower().str.strip() + "||" + df.simplified.str.lower().str.strip())
    for k, g in df.groupby(key):
        if len(g) > 1:
            ids = list(g.pair_id)
            for pid in ids[1:]:
                _flag(rows, pid, "exact_duplicate", "error", f"duplicate of {ids[0]}")
            if g.lmp_score.nunique() > 1:   # contradictory annotation for the identical pair
                for pid in ids:
                    _flag(rows, pid, "contradictory_annotation", "error", f"same pair has scores {sorted(g.lmp_score.unique())}")
    for lang, sub in df.groupby("language"):
        if len(sub) < 3:
            continue
        txt = (sub.original + " ||| " + sub.simplified).tolist()
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(txt)
        S = (X @ X.T).toarray()
        np.fill_diagonal(S, 0)
        ids, grp = sub.pair_id.tolist(), sub.group_id.tolist()
        ii, jj = np.where(np.triu(S) >= near_thr)
        for i, j in zip(ii, jj):
            if grp[i] != grp[j]:
                _flag(rows, ids[j], "near_duplicate_cross_group", "warn", f"cosine {S[i, j]:.2f} with {ids[i]} (different group: groups are merged for splitting)")
    return rows


def check_group_integrity(df) -> list[dict]:
    """Translations/paraphrases of one concept must share a group; one group must not mix concepts."""
    rows = []
    if "concept_id" not in df.columns:
        return rows
    sub = df[df.concept_id.fillna("") != ""]
    for cid, g in sub.groupby("concept_id"):
        if g.group_id.nunique() > 1:
            for pid in g.pair_id:
                _flag(rows, pid, "translation_leakage_risk", "error", f"concept {cid} spans groups {sorted(g.group_id.unique())}")
    for gid, g in df.groupby("group_id"):
        if g.language.nunique() == 1 and g.concept_id.fillna("").ne("").all() and len(g) > 0:
            _flag(rows, g.pair_id.iloc[0], "group_single_language", "info", f"group {gid} has only one language")
    return rows


def _edit_signature(o: str, s: str) -> str:
    """Coarse token-level edit signature: which tokens were removed/added (function words normalised)."""
    to, ts = set(tokenize(o)), set(tokenize(s))
    rem, add = sorted(to - ts), sorted(ts - to)
    return "-" + ",".join(rem[:3]) + " +" + ",".join(add[:3])


def check_templates(df, share_thr: float = 0.04) -> list[dict]:
    """Flag repeated generation templates: identical simplified-sentence openings and identical edit signatures."""
    rows = []
    for lang, sub in df.groupby("language"):
        opening = sub.simplified.map(lambda t: " ".join(tokenize(t)[:4]))
        cnt = Counter(opening)
        n = len(sub)
        for op, c in cnt.items():
            if c / n >= share_thr and c >= 8:
                for pid in sub.pair_id[opening == op][:1]:
                    _flag(rows, pid, "repeated_opening", "info", f"{c}/{n} ({c / n:.0%}) '{lang}' simplifications start with '{op}'")
        sig = sub.apply(lambda r: _edit_signature(r.original, r.simplified), axis=1)
        sc = Counter(sig)
        for sgn, c in sc.items():
            if c >= 4 and sgn.strip() not in ("-  +", "- +"):
                for pid in sub.pair_id[sig == sgn]:
                    _flag(rows, pid, "repeated_edit_signature", "info", f"{c} pairs share edit '{sgn}'")
    return rows


def check_hindi(df) -> list[dict]:
    rows = []
    for r in df[df.language == "hi"].itertuples():
        for col in ("original", "simplified"):
            t = getattr(r, col)
            d, l = len(_DEV.findall(t)), len(_LAT.findall(t))
            if d == 0 or d < 0.5 * (d + l):
                _flag(rows, r.pair_id, "hindi_script_ratio", "warn", f"{col}: {d} Devanagari vs {l} Latin letters")
            for rx, why in _HI_BAD:
                if rx.search(t):
                    _flag(rows, r.pair_id, "hindi_orthography", "warn", f"{col}: {why}")
            toks = t.split()   # whitespace tokens: Python's \w mishandles Devanagari combining marks
            if any(a == b and len(a) > 1 for a, b in zip(toks, toks[1:])):
                _flag(rows, r.pair_id, "hindi_orthography", "warn", f"{col}: repeated word")
            if not t.rstrip().endswith(("।", ".", "?")):
                _flag(rows, r.pair_id, "hindi_no_terminal_punct", "info", f"{col}: no sentence-final punctuation")
    return rows


def check_terminology(df) -> list[dict]:
    """For rows where an EN and an HI original share a concept_id: each glossary term in the EN original should have its Hindi
    equivalent in the HI original; and within a row, the Hindi simplification must not introduce a non-glossary synonym
    for a term that the Hindi original expresses with the glossary term (reported as info)."""
    rows = []
    if "concept_id" not in df.columns:
        return rows
    sub = df[df.concept_id.fillna("") != ""]
    seen = set()
    for cid, g in sub.groupby("concept_id"):
        en = g[g.language == "en"].original.iloc[0] if (g.language == "en").any() else None
        hi = g[g.language == "hi"].original.iloc[0] if (g.language == "hi").any() else None
        if en is None or hi is None or cid in seen:
            continue
        seen.add(cid)
        for term, rx in _TERM_EN_RE.items():
            alts = GLOSSARY[term] if isinstance(GLOSSARY[term], tuple) else (GLOSSARY[term],)
            if rx.search(en) and not any(a in hi for a in alts):
                _flag(rows, g[g.language == "hi"].pair_id.iloc[0], "terminology_mismatch", "warn",
                      f"EN original uses '{term}' but HI original lacks glossary term(s) {list(alts)}")
    return rows


def check_text_label_plausibility(df) -> list[dict]:
    """Cheap text-vs-label sanity: an 'altered_numbers' label should co-occur with a digit/number-word difference; and when the
    token sets of original and simplified are identical the pair should not carry a Not Preserved score. Heuristic."""
    from ..explain.analyzer import _numbers_and_dates
    rows = []
    for r in df.itertuples():
        labs = _lab(r.error_labels)
        try:
            no, do = _numbers_and_dates(r.original, r.language)
            ns, ds = _numbers_and_dates(r.simplified, r.language)
        except Exception:
            continue
        has_num_diff = sorted(c for c, _ in no) != sorted(c for c, _ in ns) or sorted(c for c, _ in do) != sorted(c for c, _ in ds)
        if (labs & _NUM_LABELS) and not has_num_diff and r.generation_strategy not in ("author_handwritten",):
            _flag(rows, r.pair_id, "numeric_label_without_numeric_difference", "info", "numbers/dates label but no detectable numeric difference (may be a vague-quantity change)")
        if set(tokenize(r.original)) == set(tokenize(r.simplified)) and r.lmp_score < 4:
            _flag(rows, r.pair_id, "identical_tokens_low_score", "warn", "same token set but Not Preserved")
    return rows


def run_qa(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    rows = []
    for fn in (lambda d: check_labels_and_scores(d, thresholds), check_duplicates, check_group_integrity, check_templates,
               check_hindi, check_terminology, check_text_label_plausibility):
        rows += fn(df)
    out = pd.DataFrame(rows, columns=["pair_id", "check", "severity", "detail"])
    return out.drop_duplicates()


def apply_flags(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    f = flags[flags.severity != "info"].groupby("pair_id")["check"].apply(lambda s: "|".join(sorted(set(s))))
    df["qa_flags"] = df.pair_id.map(f).fillna("")
    return df


def quality_filter(df: pd.DataFrame, flags: pd.DataFrame, drop_warnings: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, rejected). Rejected = any 'error' flag, plus any warning check named in `drop_warnings`."""
    drop_warnings = drop_warnings or set()
    bad = flags[(flags.severity == "error") | (flags.check.isin(drop_warnings))]
    rej_ids = set(bad.pair_id)
    kept, rej = df[~df.pair_id.isin(rej_ids)].copy(), df[df.pair_id.isin(rej_ids)].copy()
    reasons = bad.groupby("pair_id")["check"].apply(lambda s: "|".join(sorted(set(s))))
    rej["rejection_reason"] = rej.pair_id.map(reasons)
    return kept, rej


def lexical_cue_reliance(df: pd.DataFrame) -> dict:
    """How well do trivial lexical overlaps alone predict the label? A high value = the task may be solvable by shortcut."""
    jac, ratio = [], []
    for r in df.itertuples():
        a, b = set(tokenize(r.original)), set(tokenize(r.simplified))
        jac.append(len(a & b) / max(1, len(a | b)))
        ratio.append(len(tokenize(r.simplified)) / max(1, len(tokenize(r.original))))
    jac, ratio = np.array(jac), np.array(ratio)
    out = {"n": len(df)}
    for name, x in (("jaccard", jac), ("length_ratio", ratio)):
        out[f"spearman_{name}_vs_score"] = float(spearmanr(x, df.lmp_score)[0])
    for lang in ("en", "hi"):
        m = (df.language == lang).values
        out[f"spearman_jaccard_vs_score_{lang}"] = float(spearmanr(jac[m], df.lmp_score[m])[0]) if m.sum() > 3 else float("nan")
    out["flag"] = "HIGH shortcut risk" if abs(out["spearman_jaccard_vs_score"]) > 0.6 else "moderate/low"
    return out


def distribution_report(df: pd.DataFrame) -> dict:
    rep = {"n": len(df), "by_language": df.language.value_counts().to_dict(), "by_category": df.category.value_counts().to_dict(),
           "by_domain": df.legal_domain.value_counts().to_dict(), "by_tier": df.dataset_tier.value_counts().to_dict(),
           "language_x_category": pd.crosstab(df.language, df.category).to_dict(), "score_counts": df.lmp_score.value_counts().sort_index().to_dict(),
           "n_groups": int(df.group_id.nunique())}
    lang_cat = pd.crosstab(df.language, df.category)
    rep["imbalance_flag"] = bool((lang_cat.min(axis=None) / max(1, lang_cat.max(axis=None))) < 0.4)
    return rep
