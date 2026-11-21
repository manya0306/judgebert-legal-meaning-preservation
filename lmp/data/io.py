from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schema import ERROR_LABELS, REQUIRED, CATEGORIES, DATA_ORIGINS, LANGUAGES
from ..config import score_to_category
from ..preprocess import normalize

def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id values found")
    if not df["language"].isin(LANGUAGES).all():
        raise ValueError("language must be 'en' or 'hi'")
    if not df["lmp_score"].between(1, 10).all():
        raise ValueError("lmp_score must lie in [1, 10]")
    if not df["data_origin"].isin(DATA_ORIGINS).all():
        raise ValueError(f"data_origin must be one of {DATA_ORIGINS}")
    for c in ("original", "simplified"):
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
            raise ValueError(f"Empty or missing text in column '{c}'")
    if df["lmp_score"].isna().any():
        raise ValueError("lmp_score contains missing values")
    if "category" in df.columns:
        bad = set(df["category"].dropna()) - set(CATEGORIES)
        if bad:
            raise ValueError(f"Unknown category labels: {sorted(bad)}")
    for col in df["error_labels"].fillna(""):
        for lab in [x for x in str(col).split("|") if x]:
            if lab not in ERROR_LABELS:
                raise ValueError(f"Unknown error label: {lab}")
    return df

def load_pairs(path: str | Path) -> pd.DataFrame:
    """Load CSV or JSONL; normalise text; derive category if missing."""
    path = Path(path)
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path, encoding="utf-8")
    validate(df)
    for c in ("original", "simplified"):
        df[c] = df[c].map(normalize)
    if "category" not in df.columns or df["category"].isna().any():
        df["category"] = df["lmp_score"].map(score_to_category)
    df["error_labels"] = df.get("error_labels", pd.Series([""] * len(df))).fillna("")
    return df

def save_pairs(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")

def consolidate_annotations(ann_df: pd.DataFrame) -> pd.DataFrame:
    """Combine multi-annotator rows (pair_id, annotator_id, lmp_score, error_labels) into one record per pair:
    mean score and error labels chosen by majority (>= half of annotators)."""
    rows = []
    for pid, g in ann_df.groupby("pair_id"):
        n = len(g)
        counts: dict[str, int] = {}
        for s in g["error_labels"].fillna(""):
            for lab in {x for x in str(s).split("|") if x}:
                counts[lab] = counts.get(lab, 0) + 1
        labs = sorted(l for l, c in counts.items() if c * 2 >= n)
        rows.append({"pair_id": pid, "lmp_score": round(float(g["lmp_score"].mean()), 2),
                     "error_labels": "|".join(labs), "n_annotators": n})
    return pd.DataFrame(rows)
