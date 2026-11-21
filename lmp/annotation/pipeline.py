"""Run passes, detect disagreement, adjudicate, consolidate. Nothing here invents annotations: every consolidated label is traceable to
records in the long-format annotation file, and every row carries the strongest status the evidence supports (never 'human')."""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..config import score_to_category
from .records import COLUMNS, _labels, save_annotations, load_annotations


def run_pass(provider, pairs: pd.DataFrame, out_path, resume: bool = True) -> pd.DataFrame:
    """Run a provider on pairs not yet annotated by the same annotator_id and append to `out_path` (CSV, long format)."""
    from pathlib import Path
    existing = load_annotations(out_path) if Path(out_path).exists() and resume else pd.DataFrame(columns=COLUMNS)
    aid = getattr(provider, "annotator_id", None)
    todo = pairs if aid is None else pairs[~pairs.pair_id.isin(set(existing[existing.annotator_id == aid].pair_id))]
    new = pd.DataFrame(provider.annotate(todo)) if len(todo) else pd.DataFrame(columns=COLUMNS)
    allr = pd.concat([existing, new], ignore_index=True)
    save_annotations(allr, out_path)
    return allr


def compare_annotators(ann: pd.DataFrame, a: str, b: str, score_tol: float = 2.0, label_jaccard_min: float = 0.5) -> pd.DataFrame:
    """Per-pair comparison between two annotators; `disagree=True` if categories differ, |score diff| >= score_tol, or label Jaccard < min."""
    A = ann[ann.annotator_id == a].drop_duplicates("pair_id").set_index("pair_id")
    B = ann[ann.annotator_id == b].drop_duplicates("pair_id").set_index("pair_id")
    ids = A.index.intersection(B.index)
    rows = []
    for pid in ids:
        la, lb = _labels(A.loc[pid, "error_labels"]), _labels(B.loc[pid, "error_labels"])
        jac = 1.0 if not (la | lb) else len(la & lb) / len(la | lb)
        sa, sb = float(A.loc[pid, "lmp_score"]), float(B.loc[pid, "lmp_score"])
        ca, cb = score_to_category(sa), score_to_category(sb)
        reasons = []
        if ca != cb: reasons.append("category")
        if abs(sa - sb) >= score_tol: reasons.append("score")
        if jac < label_jaccard_min: reasons.append("labels")
        rows.append({"pair_id": pid, f"score_{a}": sa, f"score_{b}": sb, "score_diff": sb - sa, f"category_{a}": ca, f"category_{b}": cb,
                     f"labels_{a}": "|".join(sorted(la)), f"labels_{b}": "|".join(sorted(lb)), "label_jaccard": jac,
                     "disagree": bool(reasons), "reasons": "|".join(reasons)})
    return pd.DataFrame(rows)


def adjudication_sheet(cmp: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    d = cmp[cmp.disagree].merge(pairs[["pair_id", "language", "legal_domain", "original", "simplified"]], on="pair_id")
    for c in ("final_score", "final_category", "final_error_labels", "adjudicator_id", "adjudicator_type", "rationale"):
        d[c] = ""
    return d


def consolidate(pairs: pd.DataFrame, ann: pd.DataFrame, primary: str, secondary: str, adjudications: pd.DataFrame | None = None,
                score_tol: float = 2.0) -> pd.DataFrame:
    """Final labels per pair. Rules (documented in docs/DATASET_DOCUMENTATION.md):
       * both AI passes agree        -> score = mean, labels = intersection, label_source=ai_pass2, status=ai_two_pass_agreed
       * disagree + adjudication     -> adjudicated values, label_source=ai_adjudicated (or human_adjudicated if adjudicator_type=human)
       * disagree, not adjudicated   -> generation-pass labels are kept but status=ai_disagreement_unadjudicated (excluded from the filtered set)
       * only one pass               -> unchanged, status=ai_single_pass_unreviewed
    No row is ever marked human unless an adjudicator_type == 'human' record exists."""
    out = pairs.copy()
    cmp = compare_annotators(ann, primary, secondary, score_tol).set_index("pair_id")
    adj = adjudications.set_index("pair_id") if adjudications is not None and len(adjudications) else pd.DataFrame()
    sa = ann[ann.annotator_id == primary].drop_duplicates("pair_id").set_index("pair_id")
    sb = ann[ann.annotator_id == secondary].drop_duplicates("pair_id").set_index("pair_id")
    for i, r in out.iterrows():
        pid = r.pair_id
        if pid not in cmp.index:
            continue
        c = cmp.loc[pid]
        if not c.disagree:
            sc = round((float(sa.loc[pid, "lmp_score"]) + float(sb.loc[pid, "lmp_score"])) / 2 * 2) / 2
            labs = sorted(_labels(sa.loc[pid, "error_labels"]) & _labels(sb.loc[pid, "error_labels"]))
            out.loc[i, ["lmp_score", "category", "error_labels", "label_source", "annotation_status"]] = [sc, score_to_category(sc), "|".join(labs), "ai_pass2", "ai_two_pass_agreed"]
        elif len(adj) and pid in adj.index and str(adj.loc[pid, "final_score"]) not in ("", "nan"):
            sc = float(adj.loc[pid, "final_score"])
            human = str(adj.loc[pid, "adjudicator_type"]) == "human"
            out.loc[i, ["lmp_score", "category", "error_labels", "label_source", "annotation_status"]] = [
                sc, score_to_category(sc), str(adj.loc[pid, "final_error_labels"]).replace("nan", ""),
                "human_adjudicated" if human else "ai_adjudicated", "human_adjudicated" if human else "ai_adjudicated_not_independent"]
        else:
            out.loc[i, "annotation_status"] = "ai_disagreement_unadjudicated"
    return out
