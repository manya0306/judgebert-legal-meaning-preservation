"""Ingestion + validation of the completed human-validation workbooks and the two consolidated datasets.

Never writes to the input workbooks. Every check is recorded in a ``ValidationReport``; critical failures stop the
dependent analyses (the caller checks ``Loaded.ok``)."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

STUDENT_FILE, LAWYER_FILE, KEY_FILE = "Student_Annotations_Final.xlsx", "Lawyer_Annotations_Final.xlsx", "PRIVATE_Reference_Key.xlsx"
STUDENT_SHEETS = ("A1", "A2", "A3", "A4")
EXPERT_ID = "EXPERT"
CATEGORIES = ("Not Preserved", "Partially Preserved", "Preserved")
CONFIDENCES = ("High", "Medium", "Low")
CHANGE_TYPES = ("None (preserved)", "Duty-Permission-Prohibition", "Condition", "Exception-Proviso", "Party", "Scope",
                "Right-Remedy-Procedure", "Deadline-Time", "Number-Threshold", "Reversal", "Other omission", "Added content", "Other")
OTHER_FREE_TEXT = "Other (free text)"
STAT_COLS = ["orig_words", "simp_words", "orig_chars", "simp_chars", "orig_sentences", "simp_sentences", "orig_paragraphs",
             "simp_paragraphs", "word_reduction_pct", "sentence_count_diff", "paragraph_count_diff"]
HARMONIZED = {"not_preserved_below": 3.5, "preserved_from": 6.5}     # human rubric 1-3 / 4-6 / 7-10 as boundaries for continuous scores
LANG_CODE = {"english": "en", "hindi": "hi", "en": "en", "hi": "hi"}
SPLIT_FILES = (("train", "train"), ("val", "validation"), ("test", "test"))
S_COL = {"pair_id": "Pair ID", "language": "Language", "domain": "Legal domain", "complexity": "Complexity", "original": "Original legal text",
         "simplified": "Simplified text", "score": "Preservation score (1-10)", "category": "Preservation category", "confidence": "Confidence",
         "change": "Main semantic change type", "explanation": "Brief explanation of the decision", "notes": "Optional notes / ambiguity"}
L_COL = {**S_COL, "change": "Main legal meaning change, if any", "explanation": "Brief legal reasoning", "notes": "Ambiguity or concern",
         "comments": "Optional additional comments"}
KEY_REQUIRED = ["pair_id", "source_pair_id", "language", "legal_domain", "complexity_level", "dataset_split", "is_heldout_test", "selection_source",
                "original", "simplified", "AI_REFERENCE_score", "AI_REFERENCE_category", "AI_REFERENCE_error_labels", "reference_label_type",
                "label_source", "annotation_status", "group_id", "concept_id", "source_doc_id", "data_origin", "in_lawyer_packet", *STAT_COLS]
ASSIGN_REQUIRED = ["annotator", "position", "pair_id", "source_pair_id", "language"]
EXPERT_CONFIDENCE_NOTE = ("The lawyer interpreted 'Low' confidence as 'very little legal meaning preserved' rather than uncertainty about his own judgment; "
                          "the whole expert confidence field is therefore excluded from confidence-based analysis (raw responses preserved).")


@dataclass(frozen=True)
class DesignSpec:
    """The annotation design the workbooks must satisfy (defaults = the real study; tests pass a smaller one)."""
    annotators: tuple = (("A1", ("en", "hi")), ("A2", ("en", "hi")), ("A3", ("en", "hi")), ("A4", ("en",)))
    n_en: int = 50
    n_hi: int = 50
    lawyer_en: int = 15
    lawyer_hi: int = 15
    expected_student_split: tuple | None = (("test", 47), ("supplemental", 53))
    expected_expert_split: tuple | None = (("test", 20), ("supplemental", 10))
    corrections: tuple = (("HV2-HI-001", 8, 6, "Partially Preserved"), ("HV2-EN-018", 7, 6, "Partially Preserved"))

    @property
    def ann(self): return dict(self.annotators)

    @property
    def n_judgments(self): return sum(self.n_en if "en" in l else 0 for _, l in self.annotators) + sum(self.n_hi if "hi" in l else 0 for _, l in self.annotators)


# ----------------------------------------------------------------------------- small helpers
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    return s or None


def norm_text(x) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(x or "")).replace("\u00a0", " ")).strip()


def to_bool(x) -> bool:
    return x is True or str(x).strip().lower() in ("true", "1", "yes", "y")


def rubric_category(score) -> str:
    return "Not Preserved" if score <= 3 else "Partially Preserved" if score <= 6 else "Preserved"


def cat_from_score(score, thr=HARMONIZED):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    return "Not Preserved" if score < thr["not_preserved_below"] else "Preserved" if score >= thr["preserved_from"] else "Partially Preserved"


def normalize_change(label):
    """Raw semantic-change label -> canonical label (case-insensitive); unknown non-empty text -> 'Other (free text)'."""
    s = clean(label)
    if s is None:
        return None
    for c in CHANGE_TYPES:
        if s.lower() == c.lower():
            return c
    return OTHER_FREE_TEXT


def source_split_tag(x) -> str:
    return "test" if x == "test" else "supplemental"


# ----------------------------------------------------------------------------- validation report
@dataclass
class Check:
    id: str
    severity: str          # critical | warning | info
    status: str            # PASS | FAIL | WARN | INFO | UNVERIFIED
    message: str
    details: dict = field(default_factory=dict)


class ValidationReport:
    def __init__(self):
        self.checks: list[Check] = []

    def add(self, id, ok, message, severity="critical", details=None):
        status = "PASS" if ok else ("FAIL" if severity == "critical" else "WARN")
        self.checks.append(Check(id, severity, status, message, details or {}))
        return ok

    def info(self, id, message, details=None, status="INFO"):
        self.checks.append(Check(id, "info", status, message, details or {}))

    @property
    def critical_failures(self): return [c for c in self.checks if c.severity == "critical" and c.status == "FAIL"]

    @property
    def ok(self): return not self.critical_failures

    def to_dict(self):
        n = Counter(c.status for c in self.checks)
        return {"overall": "PASS" if self.ok else "FAIL (critical errors - dependent analyses were not run)", "counts": dict(n),
                "checks": [c.__dict__ for c in self.checks]}

    def to_markdown(self):
        d = self.to_dict()
        L = ["# Validation report", "", f"**Overall: {d['overall']}**  |  " + ", ".join(f"{k}: {v}" for k, v in sorted(d["counts"].items())), "",
             "| status | severity | check | message |", "|---|---|---|---|"]
        for c in sorted(self.checks, key=lambda c: ({"FAIL": 0, "WARN": 1, "UNVERIFIED": 2, "INFO": 3, "PASS": 4}[c.status], c.id)):
            L.append(f"| {c.status} | {c.severity} | `{c.id}` | {c.message.replace('|', '/')} |")
        return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- reading
def sheet_table(wb, name, anchor, rep, check_id, max_scan=6):
    """Table of one worksheet: the header row is the first of the top rows containing ``anchor``."""
    if name not in wb.sheetnames:
        rep.add(f"{check_id}.sheet", False, f"required worksheet '{name}' is missing (found {wb.sheetnames})")
        return None
    rows = list(wb[name].iter_rows(values_only=True))
    for i, row in enumerate(rows[:max_scan]):
        if any(clean(c) == anchor for c in row):
            hdr = [clean(c) or f"_col{j}" for j, c in enumerate(row)]
            df = pd.DataFrame(rows[i + 1:], columns=hdr)
            df["_excel_row"] = np.arange(i + 2, i + 2 + len(df))
            df = df[df.drop(columns="_excel_row").notna().any(axis=1)].reset_index(drop=True)
            return df
    rep.add(f"{check_id}.header", False, f"worksheet '{name}': header row containing '{anchor}' not found in the first {max_scan} rows")
    return None


def _require(df, cols, rep, check_id, label):
    missing = [c for c in cols if c not in df.columns]
    rep.add(f"{check_id}.columns", not missing, f"{label}: required columns present" if not missing else f"{label}: missing columns {missing}")
    return not missing


def load_source_dataset(data_root: Path, rep) -> pd.DataFrame | None:
    frames = []
    for fname, split in SPLIT_FILES:
        p = Path(data_root) / "data" / "splits" / f"{fname}.csv"
        if p.exists():
            d = pd.read_csv(p, keep_default_na=False)
            d["dataset_split_source"] = split
            frames.append(d)
    if len(frames) < 3:
        rep.info("source.dataset", f"source dataset splits not found under {Path(data_root) / 'data' / 'splits'}: text/provenance/split checks against the dataset are UNVERIFIED", status="UNVERIFIED")
        return None
    df = pd.concat(frames, ignore_index=True)
    rep.add("source.unique_ids", df["pair_id"].is_unique, f"source dataset: {len(df)} pairs with unique pair_id across train/val/test")
    return df


# ----------------------------------------------------------------------------- checks
def check_key(key, assign, src, spec, rep):
    if not (_require(key, KEY_REQUIRED, rep, "key", "Master_Key") and _require(assign, ASSIGN_REQUIRED, rep, "assign", "Assignment_Map")):
        return
    key["language"] = key["language"].map(lambda x: LANG_CODE.get(str(x).strip().lower(), str(x)))
    for c in ("is_heldout_test", "in_lawyer_packet"):
        key[c] = key[c].map(to_bool)
    n = spec.n_en + spec.n_hi
    rep.add("key.rows", len(key) == n and key["pair_id"].is_unique and key["source_pair_id"].is_unique, f"Master_Key has {len(key)} rows with unique neutral and source IDs (expected {n})")
    lc = key["language"].value_counts().to_dict()
    rep.add("key.languages", lc.get("en", 0) == spec.n_en and lc.get("hi", 0) == spec.n_hi, f"Master_Key language counts {lc} (expected en={spec.n_en}, hi={spec.n_hi})")
    rep.add("key.lawyer_flags", int(key.in_lawyer_packet.sum()) == spec.lawyer_en + spec.lawyer_hi, f"{int(key.in_lawyer_packet.sum())} pairs flagged for the lawyer packet (expected {spec.lawyer_en + spec.lawyer_hi})")
    rep.add("key.split_flag", bool((key.is_heldout_test == (key.dataset_split == "test")).all()), "is_heldout_test is true exactly for dataset_split == 'test' (no training pair labelled as held-out test)")
    rep.add("key.reference_typed", bool(key.reference_label_type.astype(str).str.contains("AI-GENERATED|AUTHOR-ASSIGNED").all()), "every reference label is typed AI-generated or author-assigned (no human label in the key)")
    sc = Counter(source_split_tag(s) for s in key.dataset_split)
    if spec.expected_student_split:
        exp = dict(spec.expected_student_split)
        rep.add("key.student_split_counts", dict(sc) == exp, f"student-annotated pairs by split {dict(sc)} (prior selection report: {exp})", severity="warning")
    if spec.expected_expert_split:
        ec = Counter(source_split_tag(s) for s in key.loc[key.in_lawyer_packet, "dataset_split"]); exp = dict(spec.expected_expert_split)
        rep.add("key.expert_split_counts", dict(ec) == exp, f"expert-reviewed pairs by split {dict(ec)} (prior selection report: {exp})", severity="warning")
    # assignment map
    rep.add("assign.rows", len(assign) == spec.n_judgments and not assign.duplicated(["annotator", "pair_id"]).any(), f"Assignment_Map has {len(assign)} unique (annotator, pair) rows (expected {spec.n_judgments})")
    kk = key.set_index("pair_id")
    bad = [r.pair_id for r in assign.itertuples() if r.pair_id not in kk.index or kk.at[r.pair_id, "source_pair_id"] != r.source_pair_id or LANG_CODE.get(str(r.language).lower()) != kk.at[r.pair_id, "language"]]
    rep.add("assign.consistent_with_key", not bad, "Assignment_Map pair/source/language agree with Master_Key" if not bad else f"inconsistent rows for {bad[:5]}")
    for a, langs in spec.annotators:
        sub = assign[assign.annotator == a]
        got = Counter(kk.loc[sub.pair_id, "language"]) if len(sub) and set(sub.pair_id) <= set(kk.index) else Counter()
        exp = {l: (spec.n_en if l == "en" else spec.n_hi) for l in langs}
        rep.add(f"assign.{a}", dict(got) == exp, f"{a} assigned {dict(got)} (expected {exp})")
    sets = {l: [set(assign[(assign.annotator == a)].pair_id) & set(key[key.language == l].pair_id) for a, ls in spec.annotators if l in ls] for l in ("en", "hi")}
    rep.add("assign.shared_sets", all(all(s == v[0] for s in v) for v in sets.values()), "all annotators assigned the same shared English set; all Hindi annotators the same Hindi set")
    if src is not None:
        check_key_vs_source(key, src, rep)


def check_key_vs_source(key, src, rep):
    s = src.set_index("pair_id")
    missing = [p for p in key.source_pair_id if p not in s.index]
    rep.add("key.source_ids_exist", not missing, "every source_pair_id exists in the dataset" if not missing else f"{len(missing)} source IDs not in dataset: {missing[:5]}")
    if missing:
        return
    j = key.join(s, on="source_pair_id", rsuffix="_src")
    rep.add("key.split_membership", bool((j.dataset_split == j.dataset_split_source).all()), "dataset_split in the key equals the real split file of every pair")
    rep.add("key.text_vs_dataset", bool(all(norm_text(a) == norm_text(b) and norm_text(c) == norm_text(d) for a, b, c, d in zip(j.original, j.original_src, j.simplified, j.simplified_src))), "original/simplified text in the key equals the source dataset (whitespace-normalised)")
    rep.add("key.language_domain_vs_dataset", bool(((j.language == j.language_src) & (j.legal_domain == j.legal_domain_src)).all()), "language and legal domain equal the source dataset")
    prov = [c for c in ("data_origin", "label_source", "annotation_status", "group_id", "concept_id", "source_doc_id") if c + "_src" in j.columns]
    bad = []

    for c in prov:
        left = j[c].astype("string").fillna("")
        right = j[c + "_src"].astype("string").fillna("")

        equal = left.eq(right)

        if not equal.all():
            bad.append(c)
    rep.add("key.provenance_vs_dataset", not bad, "provenance/group fields equal the source dataset" if not bad else f"mismatching provenance fields: {bad}")
    ok = bool((np.abs(pd.to_numeric(j.AI_REFERENCE_score) - pd.to_numeric(j.lmp_score)) < 1e-9).all() and (j.AI_REFERENCE_category == j.category).all())
    rep.add("key.reference_vs_dataset", ok, "reference score and category equal the source dataset")


def parse_student_long(tables: dict, spec, rep) -> pd.DataFrame:
    rows = []
    for a in STUDENT_SHEETS:
        df = tables.get(a)
        if df is None or not _require(df, list(S_COL.values()), rep, f"student.{a}", f"sheet {a}"):
            continue
        for _, r in df.iterrows():
            rows.append({"annotator_id": a, "excel_row": int(r["_excel_row"]), **{k: r[v] for k, v in S_COL.items()}})
    out = pd.DataFrame(rows)
    return out


def validate_responses(long: pd.DataFrame, rep, role="student", valid_ids=None, scale_label="") -> pd.DataFrame:
    """Row-level validation. Adds score (float), category, score_issue, category_issue, inconsistent columns."""
    long = long.copy()
    long["pair_id"] = long["pair_id"].map(clean)
    sc = pd.to_numeric(long["score"].map(lambda x: clean(x)), errors="coerce")
    long["score_num"] = sc
    long["category"] = long["category"].map(clean)
    long["score_blank"] = long["score"].map(clean).isna()
    nonnum = long[(~long.score_blank) & sc.isna()]
    out_rng = long[sc.notna() & ((sc < 1) | (sc > 10))]
    non_int = long[sc.notna() & (sc % 1 != 0)]
    bad_cat = long[long.category.notna() & ~long.category.isin(CATEGORIES)]
    rep.add(f"{role}.score_numeric", nonnum.empty, f"{role}: all non-blank scores numeric" if nonnum.empty else f"{role}: {len(nonnum)} non-numeric scores, e.g. {nonnum[['annotator_id','pair_id']].head(3).values.tolist()}")
    rep.add(f"{role}.score_range", out_rng.empty, f"{role}: all scores within 1-10" if out_rng.empty else f"{role}: {len(out_rng)} scores outside 1-10: {out_rng[['annotator_id','pair_id','score_num']].head(3).values.tolist()}")
    rep.add(f"{role}.score_integer", non_int.empty, f"{role}: all scores are whole numbers (integer rubric)" if non_int.empty else f"{role}: {len(non_int)} non-integer scores: {non_int[['annotator_id','pair_id','score_num']].head(3).values.tolist()}")
    rep.add(f"{role}.category_valid", bad_cat.empty, f"{role}: all categories valid" if bad_cat.empty else f"{role}: {len(bad_cat)} invalid categories: {bad_cat[['annotator_id','pair_id','category']].head(3).values.tolist()}")
    ok = sc.notna() & long.category.isin(CATEGORIES) & (sc >= 1) & (sc <= 10) & (sc % 1 == 0)
    exp = sc.where(ok).map(lambda s: rubric_category(s) if s == s else None)
    inc = long[ok & (exp != long.category)]
    long["inconsistent"] = ok & (exp != long.category)
    rep.add(f"{role}.score_category_consistent", inc.empty, f"{role}: category matches score under the rubric (1-3 / 4-6 / 7-10) in every response" if inc.empty else f"{role}: {len(inc)} score-category inconsistencies: {inc[['annotator_id','pair_id','score_num','category']].head(5).values.tolist()}")
    return long


def check_student(tables, key, assign, src_tables_text, spec, rep):
    """Student sheets vs assignment: duplicates, missing (genuine) vs missing-by-design, extras, text consistency."""
    long = parse_student_long(tables, spec, rep)
    if long.empty:
        rep.add("student.parsed", False, "no student responses could be parsed")
        return long
    long = validate_responses(long, rep, "student")
    blank_id = long[long.pair_id.isna()]
    rep.add("student.pair_id_present", blank_id.empty, "every non-empty student row has a Pair ID" if blank_id.empty else f"{len(blank_id)} rows without Pair ID")
    long = long[long.pair_id.notna()]
    dup = long[long.duplicated(["annotator_id", "pair_id"], keep=False)]
    rep.add("student.no_duplicates", dup.empty, "no duplicate (annotator, pair) responses" if dup.empty else f"{len(dup)} duplicated rows: {dup[['annotator_id','pair_id']].drop_duplicates().head(5).values.tolist()}")
    long = long.drop_duplicates(["annotator_id", "pair_id"], keep="first")
    rep.add("student.counts", True, "student rows per sheet: " + str(long.groupby("annotator_id").size().to_dict()), severity="info")
    exp_pairs = {a: set(assign[assign.annotator == a].pair_id) for a, _ in spec.annotators}
    allp = set(key.pair_id)
    for a, _ in spec.annotators:
        got = set(long[long.annotator_id == a].pair_id)
        rep.add(f"student.{a}.no_unassigned", got <= exp_pairs[a], f"{a}: no responses to pairs outside its assignment" if got <= exp_pairs[a] else f"{a}: unassigned pairs answered: {sorted(got - exp_pairs[a])[:5]}")
        done = set(long[(long.annotator_id == a) & (long.score_num.notna()) & long.category.notna()].pair_id)
        missing = sorted(exp_pairs[a] - done)
        rep.add(f"student.{a}.complete", not missing, f"{a}: all {len(exp_pairs[a])} assigned pairs have a score and category" if not missing else f"{a}: {len(missing)} genuinely missing responses (score or category blank/absent): {missing[:5]}")
        rep.info(f"student.{a}.missing_by_design", f"{a}: {len(allp - exp_pairs[a])} pairs intentionally not assigned (missing by design, not errors)", status="INFO")
    conf = long.confidence.map(clean)
    rep.add("student.confidence_complete", conf.notna().all(), "student confidence present in every response" if conf.notna().all() else f"{int(conf.isna().sum())} student responses without confidence", severity="warning")
    badc = long[conf.notna() & ~conf.isin(CONFIDENCES)]
    rep.add("student.confidence_valid", badc.empty, "student confidence values valid (High/Medium/Low)" if badc.empty else f"{len(badc)} invalid confidence values", severity="warning")
    # text/language/domain/complexity consistency within student sheets and vs key
    kk = key.set_index("pair_id")
    mism = []
    for r in long.itertuples():
        if r.pair_id not in kk.index:
            mism.append((r.annotator_id, r.pair_id, "unknown pair"))
            continue
        k = kk.loc[r.pair_id]
        if (norm_text(r.original) != norm_text(k.original) or norm_text(r.simplified) != norm_text(k.simplified) or LANG_CODE.get(str(r.language).strip().lower()) != k.language
                or norm_text(r.complexity).lower() != str(k.complexity_level).lower() or norm_text(r.domain).lower().replace("_", " ") != str(k.legal_domain).lower().replace("_", " ")):
            mism.append((r.annotator_id, r.pair_id, "text/language/domain/complexity differs from key"))
    rep.add("student.text_consistency", not mism, "student sheets: text, language, domain and complexity identical across annotators and to the key" if not mism else f"{len(mism)} inconsistent rows: {mism[:5]}")
    return long


def check_lawyer(df, key, student_long, spec, rep):
    if df is None or not _require(df, list(L_COL.values()), rep, "lawyer", "Lawyer_Review"):
        return None
    long = df.rename(columns={v: k for k, v in L_COL.items()}).copy()
    long["annotator_id"] = EXPERT_ID
    long["excel_row"] = long["_excel_row"]
    long = validate_responses(long, rep, "expert")
    long = long[long.pair_id.notna()]
    rep.add("expert.no_duplicates", not long.pair_id.duplicated().any(), "no duplicate expert responses")
    kk = key.set_index("pair_id")
    ids = set(long.pair_id)
    lang = Counter(kk.loc[[p for p in long.pair_id if p in kk.index], "language"])
    rep.add("expert.counts", len(long) == spec.lawyer_en + spec.lawyer_hi and lang.get("en", 0) == spec.lawyer_en and lang.get("hi", 0) == spec.lawyer_hi, f"expert reviewed {len(long)} pairs {dict(lang)} (expected en={spec.lawyer_en}, hi={spec.lawyer_hi})")
    rep.add("expert.subset_of_student", ids <= set(key.pair_id) and ids <= set(student_long.pair_id), "every expert pair is one of the student-annotated pairs")
    rep.add("expert.matches_key_flags", ids == set(key[key.in_lawyer_packet].pair_id), "expert pairs equal the in_lawyer_packet flags of the private key")
    missing = long[long.score_num.isna() | long.category.isna()]
    rep.add("expert.complete", missing.empty, "every expert row has a score and category" if missing.empty else f"{len(missing)} expert rows missing score/category")
    mism = [r.pair_id for r in long.itertuples() if r.pair_id in kk.index and (norm_text(r.original) != norm_text(kk.at[r.pair_id, "original"]) or norm_text(r.simplified) != norm_text(kk.at[r.pair_id, "simplified"]) or LANG_CODE.get(str(r.language).strip().lower()) != kk.at[r.pair_id, "language"])]
    rep.add("expert.text_consistency", not mism, "expert sheet text/language equal the key" if not mism else f"expert text mismatches: {mism[:5]}")
    rep.info("expert.confidence_excluded", EXPERT_CONFIDENCE_NOTE + " Raw values retained in the individual dataset.", status="INFO")
    return long


def verify_corrections(expert, spec, rep) -> pd.DataFrame:
    """Verify (never apply) the manually incorporated expert corrections; return the correction log."""
    rows = []
    obs = expert.set_index("pair_id") if expert is not None else pd.DataFrame()
    for pid, orig, final, cat in spec.corrections:
        found = pid in obs.index
        s_obs = float(obs.at[pid, "score_num"]) if found else np.nan
        c_obs = obs.at[pid, "category"] if found else None
        ok = found and s_obs == final and c_obs == cat
        rep.add(f"expert.correction.{pid}", ok, f"{pid}: expert score {s_obs:g} / {c_obs} (expected corrected {final} / {cat}; original {orig}); correction NOT re-applied" if found else f"{pid}: not found in expert sheet")
        rows.append({"pair_id": pid, "item": "expert score/category correction", "original_score": orig, "expected_final_score": final, "observed_final_score": s_obs,
                     "expected_final_category": cat, "observed_final_category": c_obs, "verified": bool(ok),
                     "status": "manually incorporated into Lawyer_Annotations_Final.xlsx; expert-confirmed correction; verified only, NOT re-applied by this pipeline"})
    rows.append({"pair_id": "ALL_EXPERT", "item": "expert confidence field", "original_score": None, "expected_final_score": None, "observed_final_score": None,
                 "expected_final_category": None, "observed_final_category": None, "verified": True,
                 "status": "raw responses preserved; field excluded from confidence-based analysis. " + EXPERT_CONFIDENCE_NOTE})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- consolidated datasets
def build_individual(student: pd.DataFrame, expert: pd.DataFrame | None, key: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    kk = key.set_index("pair_id")
    corrected = set(corrections.loc[corrections.pair_id != "ALL_EXPERT", "pair_id"])
    recs = []
    frames = [(student, "student")] + ([(expert, "expert")] if expert is not None else [])
    for df, role in frames:
        for r in df.itertuples():
            if pd.isna(r.score_num) or r.category is None:
                continue
            k = kk.loc[r.pair_id]
            conf = clean(r.confidence)
            if role == "expert":
                cstat, cvalid = "excluded_expert_rubric_misinterpretation", False
            else:
                cstat = "missing" if conf is None else "valid" if conf in CONFIDENCES else "invalid"
                cvalid = cstat == "valid"
            recs.append({
                "pair_id": r.pair_id, "source_pair_id": k.source_pair_id, "annotator_id": r.annotator_id, "annotator_role": role, "language": k.language,
                "legal_domain": k.legal_domain, "complexity": k.complexity_level, "group_id": k.group_id, "concept_id": k.concept_id, "source_doc_id": k.source_doc_id,
                "original": k.original, "simplified": k.simplified, **{c: k[c] for c in STAT_COLS},
                "dataset_split": k.dataset_split, "is_heldout_test": bool(k.is_heldout_test), "selection_source": k.selection_source,
                "reference_label_type": k.reference_label_type, "label_source": k.label_source, "data_origin": k.data_origin,
                "ref_score": float(k.AI_REFERENCE_score), "ref_category_original": k.AI_REFERENCE_category, "ref_error_labels": k.AI_REFERENCE_error_labels or "",
                "human_score": float(r.score_num), "human_category": r.category, "semantic_change_raw": clean(r.change), "semantic_change_type": normalize_change(r.change),
                "explanation": clean(r.explanation), "notes": clean(r.notes) if role == "student" else clean(getattr(r, "notes", None)),
                "expert_additional_comments": clean(getattr(r, "comments", None)) if role == "expert" else None,
                "confidence_raw": conf, "confidence_valid_for_analysis": cvalid, "confidence_status": cstat,
                "expert_correction_manually_incorporated": bool(role == "expert" and r.pair_id in corrected), "response_excel_row": int(r.excel_row)})
    return pd.DataFrame(recs)


def _majority(cats):
    c = Counter(x for x in cats if x)
    if not c:
        return None, False, ""
    top = max(c.values())
    winners = sorted(k for k, v in c.items() if v == top)
    return (winners[0] if len(winners) == 1 else None), len(winners) > 1, ";".join(f"{k}:{v}" for k, v in sorted(c.items()))


def build_consensus(ind: pd.DataFrame, key: pd.DataFrame, annotators=STUDENT_SHEETS, high_disagreement_range: float = 3.0) -> pd.DataFrame:
    stu = ind[ind.annotator_role == "student"]
    exp = ind[ind.annotator_role == "expert"].set_index("pair_id")
    rows = []
    for k in key.itertuples():
        s = stu[stu.pair_id == k.pair_id].set_index("annotator_id")
        sc = s.human_score.astype(float)
        maj, tie, votes = _majority(list(s.human_category))
        lab = Counter(x for x in s.semantic_change_type if x)
        top = max(lab.values()) if lab else 0
        winners = sorted(l for l, v in lab.items() if v == top)
        mean = float(sc.mean()) if len(sc) else np.nan
        med = float(sc.median()) if len(sc) else np.nan
        r = {"pair_id": k.pair_id, "source_pair_id": k.source_pair_id, "language": k.language, "legal_domain": k.legal_domain, "complexity": k.complexity_level,
             "group_id": k.group_id, "concept_id": k.concept_id, "dataset_split": k.dataset_split, "is_heldout_test": bool(k.is_heldout_test),
             "selection_source": k.selection_source, "reference_label_type": k.reference_label_type, "label_source": k.label_source, "data_origin": k.data_origin,
             "original": k.original, "simplified": k.simplified, **{c: getattr(k, c) for c in STAT_COLS}}
        for a in annotators:
            r[f"score_{a}"] = float(sc[a]) if a in sc.index else np.nan
            r[f"cat_{a}"] = s.human_category.get(a) if a in s.index else None
            r[f"change_{a}"] = s.semantic_change_type.get(a) if a in s.index else None
        r.update({"n_student_raters": int(len(sc)), "student_mean": mean, "student_median": med,
                  "student_sd": float(sc.std(ddof=1)) if len(sc) > 1 else np.nan, "student_range": float(sc.max() - sc.min()) if len(sc) else np.nan,
                  "student_high_disagreement": bool(len(sc) and sc.max() - sc.min() >= high_disagreement_range),
                  "student_category_disagreement": bool(len(set(s.human_category)) > 1), "student_majority_category": maj, "student_majority_tie": tie, "student_category_votes": votes,
                  "student_mean_category": cat_from_score(mean), "student_median_category": cat_from_score(med),
                  "student_modal_change": winners[0] if len(winners) == 1 else None, "student_modal_change_tie": len(winners) > 1,
                  "student_change_agreement_share": top / sum(lab.values()) if lab else np.nan,
                  "ref_score": float(k.AI_REFERENCE_score), "ref_category_original": k.AI_REFERENCE_category, "ref_category_harmonized": cat_from_score(float(k.AI_REFERENCE_score)),
                  "ref_error_labels": k.AI_REFERENCE_error_labels or "", "expert_reviewed": k.pair_id in exp.index,
                  "expert_score": float(exp.at[k.pair_id, "human_score"]) if k.pair_id in exp.index else np.nan,
                  "expert_category": exp.at[k.pair_id, "human_category"] if k.pair_id in exp.index else None,
                  "expert_semantic_change": exp.at[k.pair_id, "semantic_change_type"] if k.pair_id in exp.index else None,
                  "expert_correction_manually_incorporated": bool(exp.at[k.pair_id, "expert_correction_manually_incorporated"]) if k.pair_id in exp.index else False})
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- orchestration
@dataclass
class Loaded:
    report: ValidationReport
    hashes: dict
    key: pd.DataFrame | None = None
    assign: pd.DataFrame | None = None
    source: pd.DataFrame | None = None
    student: pd.DataFrame | None = None
    expert: pd.DataFrame | None = None
    corrections: pd.DataFrame | None = None
    individual: pd.DataFrame | None = None
    consensus: pd.DataFrame | None = None

    @property
    def ok(self): return self.report.ok and self.consensus is not None


def load_and_validate(input_dir, spec: DesignSpec = DesignSpec(), data_root=None, names=(STUDENT_FILE, LAWYER_FILE, KEY_FILE)) -> Loaded:
    rep = ValidationReport()
    d = Path(input_dir)
    paths = dict(zip(("student", "lawyer", "key"), (d / n for n in names)))
    hashes = {}
    for k, p in paths.items():
        ok = p.exists()
        rep.add(f"file.{k}", ok, f"input workbook {p.name} found" if ok else f"input workbook not found: {p}")
        if ok:
            hashes[p.name] = sha256(p)
    out = Loaded(rep, hashes)
    if not rep.ok:
        return out
    try:
        wbs = {k: load_workbook(p, data_only=False) for k, p in paths.items()}   # opened in memory only; never saved
    except Exception as e:  # noqa: BLE001
        rep.add("file.readable", False, f"could not read workbooks: {type(e).__name__}: {e}")
        return out
    stu_tabs = {a: sheet_table(wbs["student"], a, "Pair ID", rep, f"student.{a}") for a in STUDENT_SHEETS}
    lawyer_df = sheet_table(wbs["lawyer"], "Lawyer_Review", "Pair ID", rep, "lawyer")
    key = sheet_table(wbs["key"], "Master_Key", "pair_id", rep, "key")
    assign = sheet_table(wbs["key"], "Assignment_Map", "annotator", rep, "assign")
    if not rep.ok or key is None or assign is None:
        return out
    src = load_source_dataset(data_root or Path.cwd(), rep)
    check_key(key, assign, src, spec, rep)
    if not rep.ok:
        out.key = key
        return out
    student = check_student(stu_tabs, key, assign, None, spec, rep)
    expert = check_lawyer(lawyer_df, key, student, spec, rep)
    corr = verify_corrections(expert, spec, rep) if expert is not None else pd.DataFrame()
    out.key, out.assign, out.source, out.student, out.expert, out.corrections = key, assign, src, student, expert, corr
    if rep.ok:
        out.individual = build_individual(student, expert, key, corr)
        out.consensus = build_consensus(out.individual, key, tuple(a for a, _ in spec.annotators))
        n_ind = len(out.individual)
        rep.add("dataset.individual_count", n_ind == spec.n_judgments + spec.lawyer_en + spec.lawyer_hi, f"individual dataset holds {n_ind} human judgments (expected {spec.n_judgments + spec.lawyer_en + spec.lawyer_hi})")
    return out
