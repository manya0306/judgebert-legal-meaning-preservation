"""Assemble the expanded dataset from (a) the original 48 author-written pairs and (b) AI-generated concepts.

Every row carries provenance. Nothing here is human-annotated: original rows have author-assigned scores,
AI-generated rows carry the labels intended by the generation pass (`label_source=ai_generation_intent`), which is
then audited by an independent-context second pass (see lmp/annotation and docs/DATASET_DOCUMENTATION.md).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
from ..config import ROOT, score_to_category
from ..preprocess import normalize
from .schema import COLUMNS

STRATEGY = {"P": "faithful_paraphrase", "H": "heavy_rewrite", "L": "lossy_simplification",
            "M": "minimal_meaning_edit", "X": "compound_change"}
GENERATOR = "claude-sonnet-5.5 (in-session authoring; no API call)"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_concepts(gen_dir: Path | None = None) -> list[tuple]:
    gen_dir = gen_dir or ROOT / "data" / "generated"
    out = []
    for f in sorted(gen_dir.glob("concepts_*.py")):
        out.extend(_load_module(f).CONCEPTS)
    ids = [c[0] for c in out]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate concept ids in generated concept files")
    return out


def _scores(s):
    return (s, s) if not isinstance(s, tuple) else s


def concepts_to_frame(concepts: list[tuple], thresholds: dict | None = None) -> pd.DataFrame:
    rows = []
    for cid, domain, en, hi, variants in concepts:
        for i, (kind, s_en, s_hi, score, labels) in enumerate(variants, 1):
            sc_en, sc_hi = _scores(score)
            for lang, orig, simp, sc in (("en", en, s_en, sc_en), ("hi", hi, s_hi, sc_hi)):
                rows.append({
                    "pair_id": f"AIG-{cid}-{lang.upper()}-{i:02d}", "language": lang, "legal_domain": domain,
                    "original": normalize(orig), "simplified": normalize(simp), "lmp_score": float(sc),
                    "category": score_to_category(float(sc), thresholds), "error_labels": labels,
                    "source_doc_id": f"AIG-{cid}", "group_id": f"G-{cid}", "concept_id": cid,
                    "data_origin": "ai_generated", "dataset_tier": "ai_generated",
                    "generation_strategy": STRATEGY[kind], "generator": GENERATOR,
                    "label_source": "ai_generation_intent", "annotation_status": "ai_single_pass_unreviewed",
                    "qa_flags": "",
                    "source_provenance": "AI-authored legal-style text; NOT quoted from any real statute, contract or judgment",
                    "license_note": "Project-original AI-generated text; no third-party rights asserted"})
    return pd.DataFrame(rows)


def original_frame(path: Path | None = None) -> pd.DataFrame:
    path = path or ROOT / "data" / "synthetic" / "synthetic_pairs.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["original"], df["simplified"] = df["original"].map(normalize), df["simplified"].map(normalize)
    norm = df["source_doc_id"].str.replace(r"^SYN-(?:EN|HI)-", "SYN-", regex=True)   # EN/HI counterparts share a group
    df["group_id"] = "G-" + norm
    df["concept_id"] = ""
    df["dataset_tier"] = "original_synthetic"
    df["generation_strategy"] = "author_handwritten"
    df["generator"] = "project author (human)"
    df["label_source"] = "author_assigned"
    df["annotation_status"] = "author_assigned_score_not_independent_annotation"
    df["qa_flags"] = ""
    return df


def build_expanded(gen_dir: Path | None = None, thresholds: dict | None = None) -> pd.DataFrame:
    orig = original_frame()
    new = concepts_to_frame(load_concepts(gen_dir), thresholds)
    df = pd.concat([orig, new], ignore_index=True)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]
