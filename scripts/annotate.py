"""AI-assisted annotation workflow CLI (run from project root).
  python scripts/annotate.py build-pass1   --pairs data/expanded/pairs_expanded.csv --out data/annotation/ai/annotations_long.csv
  python scripts/annotate.py import-blind  --raw data/annotation/ai/pass2_blind_raw.txt --key data/annotation/ai/pass2_blind_key.csv --pairs ... --out ...
  python scripts/annotate.py heuristic     --pairs ... --out ...
  python scripts/annotate.py api           --provider anthropic --model <model> --pairs ... --out ...   (needs ANTHROPIC_API_KEY; fails loudly otherwise)
  python scripts/annotate.py validate|compare|agreement|consolidate ...
"""
import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from lmp.config import score_to_category
from lmp.annotation import providers, pipeline, records
from lmp.annotation.evidence import diff_evidence
from lmp.data.agreement import agreement_report
from lmp.data.io import load_pairs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--pairs", default="data/expanded/pairs_expanded.csv"); ap.add_argument("--out", default="data/annotation/ai/annotations_long.csv")
    ap.add_argument("--raw"); ap.add_argument("--key"); ap.add_argument("--provider"); ap.add_argument("--model"); ap.add_argument("--base-url")
    ap.add_argument("--a", default="claude-gen-intent"); ap.add_argument("--b", default="claude-pass2-blind")
    ap.add_argument("--adjudications"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    pairs = load_pairs(a.pairs)
    if a.limit: pairs = pairs.head(a.limit)
    if a.cmd == "build-pass1":
        records.save_annotations(pipeline.run_pass(providers.GenerationIntentProvider(), pairs[pairs.dataset_tier == "ai_generated"], a.out), a.out)
    elif a.cmd == "heuristic":
        pipeline.run_pass(providers.HeuristicProvider(), pairs, a.out)
    elif a.cmd == "import-blind":
        key = pd.read_csv(a.key).set_index("blind_id")["pair_id"].to_dict(); P = pairs.set_index("pair_id"); rows = []
        conf = {"h": 0.9, "m": 0.7, "l": 0.5}
        for line in Path(a.raw).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            bid, sc, labs, c, expl = [x.strip() for x in line.split(";", 4)]
            pid = key[bid]; r = P.loc[pid]; eo, es = diff_evidence(r.original, r.simplified)
            rows.append({"pair_id": pid, "annotator_id": "claude-pass2-blind", "annotator_type": "ai", "provider": "in_session", "model": "claude-sonnet-5.5",
                         "model_version": "", "pass_id": "pass2_blind", "timestamp_utc": "", "language": r.language, "legal_domain": r.legal_domain,
                         "lmp_score": float(sc), "category": score_to_category(float(sc)), "error_labels": labs.replace("+", "|"), "changed_elements": "[]",
                         "explanation": expl, "evidence_original": eo, "evidence_simplified": es, "confidence": conf[c], "review_status": "unreviewed",
                         "prompt_id": "annotate_lmp_v1 (rubric applied in-session)",
                         "independence_note": "Same model family as the generator; written in a later step from shuffled, opaquely-ID'd text with the generation labels not consulted. Errors are correlated with the generator's; NOT independent expert or human annotation. Evidence spans computed automatically by token diff."})
        ex = records.load_annotations(a.out) if Path(a.out).exists() else pd.DataFrame(columns=records.COLUMNS)
        ex = ex[ex.annotator_id != "claude-pass2-blind"]
        records.save_annotations(pd.concat([ex, pd.DataFrame(rows)], ignore_index=True), a.out); print("imported", len(rows))
    elif a.cmd == "api":
        cls = providers.PROVIDERS[a.provider]
        prov = cls(a.model) if a.provider == "anthropic" else cls(a.base_url, a.model)   # raises ProviderUnavailable when unusable
        pipeline.run_pass(prov, pairs, a.out)
    elif a.cmd == "validate":
        ann = records.load_annotations(a.out); probs = records.validate_annotations(ann, pairs)
        print(probs.problem.value_counts().to_string() if len(probs) else "no problems"); probs.to_csv(Path(a.out).with_name("validation_problems.csv"), index=False)
    elif a.cmd == "compare":
        ann = records.load_annotations(a.out); cmp = pipeline.compare_annotators(ann, a.a, a.b)
        cmp.to_csv(Path(a.out).with_name("comparison_pass1_pass2.csv"), index=False, encoding="utf-8-sig")
        print(len(cmp), "compared;", int(cmp.disagree.sum()), "disagreements"); print(cmp.reasons.value_counts().to_string())
        pipeline.adjudication_sheet(cmp, pairs).to_csv(Path(a.out).with_name("adjudication_sheet.csv"), index=False, encoding="utf-8-sig")
    elif a.cmd == "agreement":
        ann = records.load_annotations(a.out); rep = agreement_report(ann, [a.a, a.b], "AI-vs-AI (same model family)")
        Path(a.out).with_name("agreement_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps(rep, indent=2, ensure_ascii=False))
    elif a.cmd == "consolidate":
        ann = records.load_annotations(a.out); adj = pd.read_csv(a.adjudications) if a.adjudications else None
        out = pipeline.consolidate(pairs, ann, a.a, a.b, adj); out.to_csv(a.pairs.replace(".csv", "_consolidated.csv"), index=False, encoding="utf-8-sig")
        print(out.annotation_status.value_counts().to_string())
main()
