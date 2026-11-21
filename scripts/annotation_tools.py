"""Annotation workflow helpers.
  sheets       split a pairs CSV into one blank sheet per annotator (language/domain order randomised per annotator)
  agreement    inter-annotator agreement from a long-format CSV (pair_id, annotator_id, lmp_score, error_labels)
  consolidate  merge multi-annotator rows into one record per pair and join them onto the pairs file
Usage:  python scripts/annotation_tools.py sheets --pairs data/annotation/pairs_to_annotate.csv --annotators A B --out data/annotation/sheets
        python scripts/annotation_tools.py agreement --annotations data/annotation/all_annotations.csv
        python scripts/annotation_tools.py consolidate --pairs P.csv --annotations A.csv --out data/final.csv"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from lmp.data.agreement import pairwise_score_agreement, category_kappa, error_label_kappa
from lmp.data.io import consolidate_annotations, save_pairs
from lmp.config import score_to_category

ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("sheets"); s.add_argument("--pairs", required=True); s.add_argument("--annotators", nargs="+", required=True); s.add_argument("--out", required=True)
g = sub.add_parser("agreement"); g.add_argument("--annotations", required=True)
c = sub.add_parser("consolidate"); c.add_argument("--pairs", required=True); c.add_argument("--annotations", required=True); c.add_argument("--out", required=True)
a = ap.parse_args()
if a.cmd == "sheets":
    df = pd.read_csv(a.pairs); Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, ann in enumerate(a.annotators):
        sh = df[["pair_id", "language", "legal_domain", "original", "simplified"]].sample(frac=1, random_state=i).assign(annotator_id=ann, lmp_score="", error_labels="", notes="")
        sh.to_csv(Path(a.out) / f"sheet_{ann}.csv", index=False, encoding="utf-8-sig")
    print("wrote", len(a.annotators), "sheets")
elif a.cmd == "agreement":
    ann = pd.read_csv(a.annotations)
    for title, t in (("Score agreement (pairwise)", pairwise_score_agreement(ann)), ("Category kappa (3-class)", category_kappa(ann)), ("Error-label kappa", error_label_kappa(ann))):
        print(f"\n== {title} ==\n{t.to_string(index=False) if len(t) else 'not enough overlapping annotations'}")
else:
    pairs, ann = pd.read_csv(a.pairs), pd.read_csv(a.annotations)
    merged = pairs.drop(columns=[c for c in ("lmp_score", "error_labels", "category") if c in pairs.columns]).merge(consolidate_annotations(ann), on="pair_id")
    merged["category"] = merged["lmp_score"].map(score_to_category); save_pairs(merged, a.out); print("wrote", len(merged), "consolidated pairs ->", a.out)
