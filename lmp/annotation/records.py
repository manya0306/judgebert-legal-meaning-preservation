"""Long-format annotation records (one row per pair x annotator x pass) and validation.

annotator_type: 'ai' | 'human' | 'heuristic'. An 'ai' record is NEVER a human annotation. `independence_note` states how independent
the record is from other records for the same pair (e.g. 'same model family as generator, fresh context, labels hidden').
"""
from __future__ import annotations
import json
import pandas as pd
from ..config import score_to_category
from ..data.schema import ERROR_LABELS, CATEGORIES

COLUMNS = ["pair_id", "annotator_id", "annotator_type", "provider", "model", "model_version", "pass_id", "timestamp_utc",
           "language", "legal_domain", "lmp_score", "category", "error_labels", "changed_elements", "explanation",
           "evidence_original", "evidence_simplified", "confidence", "review_status", "prompt_id", "independence_note"]


def _labels(x) -> set[str]:
    return {t for t in str(x if pd.notna(x) else "").split("|") if t}


def validate_annotations(ann: pd.DataFrame, pairs: pd.DataFrame | None = None, thresholds: dict | None = None,
                         require_explanation: bool = True) -> pd.DataFrame:
    """Return a DataFrame of problems (pair_id, annotator_id, problem). Empty = clean."""
    probs = []
    def bad(r, msg): probs.append({"pair_id": r.pair_id, "annotator_id": r.annotator_id, "problem": msg})
    ptxt = pairs.set_index("pair_id") if pairs is not None else None
    seen = set()
    for r in ann.itertuples():
        key = (r.pair_id, r.annotator_id, getattr(r, "pass_id", ""))
        if key in seen: bad(r, "duplicate (pair, annotator, pass)")
        seen.add(key)
        if ptxt is not None and r.pair_id not in ptxt.index: bad(r, "pair_id not in pairs file"); continue
        if not (1 <= float(r.lmp_score) <= 10): bad(r, "score outside 1-10")
        if r.category not in CATEGORIES: bad(r, f"invalid category {r.category!r}")
        elif r.category != score_to_category(float(r.lmp_score), thresholds): bad(r, "category inconsistent with score")
        unk = _labels(r.error_labels) - set(ERROR_LABELS)
        if unk: bad(r, f"unknown labels {sorted(unk)}")
        if r.annotator_type not in ("ai", "human", "heuristic"): bad(r, "annotator_type must be ai|human|heuristic")
        expl = str(r.explanation) if pd.notna(r.explanation) else ""
        if require_explanation and r.annotator_type != "heuristic" and len(expl.strip()) < 8: bad(r, "missing explanation")
        if ptxt is not None:   # evidence must be a verbatim substring (unsupported evidence is flagged)
            o, s = ptxt.loc[r.pair_id, "original"], ptxt.loc[r.pair_id, "simplified"]
            eo = str(r.evidence_original) if pd.notna(r.evidence_original) else ""
            es = str(r.evidence_simplified) if pd.notna(r.evidence_simplified) else ""
            if eo and eo not in ("(none)", "(absent)") and eo not in o: bad(r, "evidence_original is not a substring of the original")
            if es and es not in ("(none)", "(absent)") and es not in s: bad(r, "evidence_simplified is not a substring of the simplified text")
        if r.category == "Not Preserved" and not _labels(r.error_labels): bad(r, "Not Preserved without any changed-element label")
    return pd.DataFrame(probs, columns=["pair_id", "annotator_id", "problem"])


def load_annotations(path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True) if str(path).endswith(".jsonl") else pd.read_csv(path, encoding="utf-8-sig")
    for c in COLUMNS:
        if c not in df.columns: df[c] = ""
    df["error_labels"] = df["error_labels"].fillna("").astype(str)
    return df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]


def save_annotations(df: pd.DataFrame, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
