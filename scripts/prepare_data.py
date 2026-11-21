"""Reproducible data preparation (run from project root):  python scripts/prepare_data.py
build (original 48 + AI-generated concepts) -> QA flags -> consolidate AI passes/adjudications -> variants + report.
Writes data/expanded/{pairs_expanded.csv, pairs_consolidated.csv, pairs_filtered.csv, pairs_reviewed.csv, pairs_original_only.csv,
qa_flags.csv, rejected.csv, dataset_report.json}."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import pandas as pd
from lmp.data.build import build_expanded
from lmp.data.qa import run_qa, apply_flags, quality_filter, lexical_cue_reliance, distribution_report
from lmp.data.io import save_pairs
from lmp.annotation import records, pipeline

OUT = ROOT / "data" / "expanded"; ANN = ROOT / "data" / "annotation" / "ai"
UNRESOLVED_WARNINGS = {"lossy_variant_scored_preserved", "preserved_but_severe_label", "identical_tokens_low_score_unreviewed"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = build_expanded(); flags0 = run_qa(raw); raw = apply_flags(raw, flags0); save_pairs(raw, OUT / "pairs_expanded.csv")
    cons = raw.copy()
    if (ANN / "annotations_long.csv").exists():
        ann = records.load_annotations(ANN / "annotations_long.csv")
        adj = pd.read_csv(ANN / "adjudications.csv") if (ANN / "adjudications.csv").exists() else None
        cons = pipeline.consolidate(raw, ann, "claude-gen-intent", "claude-pass2-blind", adj)
    flags = run_qa(cons); cons = apply_flags(cons, flags); save_pairs(cons, OUT / "pairs_consolidated.csv"); flags.to_csv(OUT / "qa_flags.csv", index=False, encoding="utf-8-sig")
    kept, rej = quality_filter(cons, flags, drop_warnings=UNRESOLVED_WARNINGS)
    unres = kept.annotation_status == "ai_disagreement_unadjudicated"
    rej = pd.concat([rej, kept[unres].assign(rejection_reason="ai_disagreement_unadjudicated")]); kept = kept[~unres]
    save_pairs(kept, OUT / "pairs_filtered.csv"); rej.to_csv(OUT / "rejected.csv", index=False, encoding="utf-8-sig")
    reviewed = cons[cons.annotation_status.isin(["ai_two_pass_agreed", "ai_adjudicated_not_independent"]) | (cons.dataset_tier == "original_synthetic")]
    save_pairs(reviewed, OUT / "pairs_reviewed.csv"); save_pairs(cons[cons.dataset_tier == "original_synthetic"], OUT / "pairs_original_only.csv")
    rep = {"counts": {"original_synthetic_written_by_author": int((cons.dataset_tier == "original_synthetic").sum()),
                      "ai_generated_total": int((cons.dataset_tier == "ai_generated").sum()),
                      "ai_generated_with_second_blind_pass": int(cons.annotation_status.isin(["ai_two_pass_agreed", "ai_adjudicated_not_independent"]).sum()),
                      "ai_generated_single_pass_only": int((cons.annotation_status == "ai_single_pass_unreviewed").sum()),
                      "disagreements_found_and_adjudicated": int(len(pd.read_csv(ANN / "adjudications.csv"))) if (ANN / "adjudications.csv").exists() else None,
                      "accepted_in_filtered_set": len(kept), "rejected_from_filtered_set": len(rej),
                      "human_annotated": int((cons.dataset_tier == "human_annotated").sum()), "external_real_world": int((cons.dataset_tier == "external").sum())},
           "qa_flag_counts": flags.groupby(["severity", "check"]).size().rename("n").reset_index().to_dict("records"),
           "distribution_consolidated": distribution_report(cons), "distribution_filtered": distribution_report(kept),
           "lexical_cue_reliance_consolidated": lexical_cue_reliance(cons),
           "status_counts": cons.annotation_status.value_counts().to_dict()}
    (OUT / "dataset_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep["counts"], indent=1)); print("rejected reasons:", rej.rejection_reason.value_counts().to_dict() if len(rej) else {})
main()
