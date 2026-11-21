import numpy as np, pandas as pd, pytest
from lmp.annotation import providers, pipeline, records
from lmp.data.agreement import krippendorff_alpha, weighted_kappa, agreement_report

P = pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "language": "en", "legal_domain": "tenancy", "original": ["A shall pay.", "B may leave.", "C must file."],
                  "simplified": ["A pays.", "B shall leave.", "C files."], "lmp_score": [9.0, 3.0, 8.0], "error_labels": ["", "changed_permission", ""]})

def test_api_provider_refuses_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(providers.ProviderUnavailable): providers.AnthropicProvider("some-model")

def test_local_provider_refuses_without_server():
    with pytest.raises(providers.ProviderUnavailable): providers.OpenAICompatibleProvider("http://127.0.0.1:9", "m")

def test_generation_intent_is_marked_not_independent():
    r = providers.GenerationIntentProvider().annotate(P); assert r[0]["annotator_type"] == "ai" and "NOT independent" in r[0]["independence_note"]

def _ann(pid, who, score, labels, typ="ai", expl="Some explanation text."):
    return {"pair_id": pid, "annotator_id": who, "annotator_type": typ, "provider": "x", "model": "m", "model_version": "", "pass_id": who, "timestamp_utc": "", "language": "en", "legal_domain": "tenancy",
            "lmp_score": score, "category": pipeline.score_to_category(score), "error_labels": labels, "changed_elements": "[]", "explanation": expl, "evidence_original": "", "evidence_simplified": "",
            "confidence": 0.9, "review_status": "unreviewed", "prompt_id": "t", "independence_note": ""}

def test_validation_catches_problems():
    ann = pd.DataFrame([_ann("p1", "a", 9, ""), _ann("p2", "a", 2, "", expl="x"), _ann("zz", "a", 5, "bogus")])
    ann.loc[0, "category"] = "Not Preserved"; probs = records.validate_annotations(ann, P).problem.str.cat(sep=";")
    assert "inconsistent" in probs and "missing explanation" in probs and "not in pairs" in probs and "Not Preserved without" in probs

def test_evidence_must_be_substring():
    a = pd.DataFrame([_ann("p1", "a", 9, "")]); a.loc[0, "evidence_original"] = "text not there"
    assert "not a substring" in records.validate_annotations(a, P).problem.str.cat(sep=";")

def test_disagreement_and_consolidation_statuses():
    ann = pd.DataFrame([_ann("p1", "a", 9, ""), _ann("p1", "b", 9, ""), _ann("p2", "a", 3, "changed_permission"), _ann("p2", "b", 8, ""), _ann("p3", "a", 8, "")])
    cmp = pipeline.compare_annotators(ann, "a", "b"); assert cmp.set_index("pair_id").disagree.to_dict() == {"p1": False, "p2": True}
    out = pipeline.consolidate(P.assign(annotation_status="x", label_source="y", category="z"), ann, "a", "b").set_index("pair_id")
    assert out.loc["p1", "annotation_status"] == "ai_two_pass_agreed" and out.loc["p2", "annotation_status"] == "ai_disagreement_unadjudicated" and out.loc["p3", "annotation_status"] == "x"
    adj = pipeline.adjudication_sheet(cmp, P); adj["final_score"], adj["final_error_labels"], adj["adjudicator_type"] = 3.0, "changed_permission", "ai"
    out = pipeline.consolidate(P.assign(annotation_status="x", label_source="y", category="z"), ann, "a", "b", adj).set_index("pair_id")
    assert out.loc["p2", "annotation_status"] == "ai_adjudicated_not_independent" and out.loc["p2", "label_source"] == "ai_adjudicated"   # never 'human'

def test_krippendorff_matches_reference_package_values():
    n = np.nan; M = np.array([[1,2,3,3,2,1,4,1,2,n,n,n,3,3,3],[1,2,3,3,2,2,4,1,2,5,n,3,n,3,3],[n,3,3,3,2,3,4,2,2,5,1,n,3,3,4],[1,2,3,3,2,4,4,1,2,5,1,n,n,n,n]], float)
    for lv, ref in (("nominal", 0.726744), ("ordinal", 0.820520), ("interval", 0.840864)): assert krippendorff_alpha(M, lv) == pytest.approx(ref, abs=1e-5)   # values from the `krippendorff` PyPI package
    assert krippendorff_alpha(np.array([[1, 2, 3, 4], [1, 2, 3, 4]], float), "interval") == pytest.approx(1.0)

def test_weighted_kappa_and_report_labels_ai_vs_ai():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    rows = [_ann(f"p{i}", w, s, "") for i, s in enumerate([1, 3, 5, 7, 9, 2, 8, 6]) for w in ("a", "b")]
    rep = agreement_report(pd.DataFrame(rows)); assert "NOT human" in rep["warning"] and rep["pairs"][0]["weighted_kappa_score_quadratic"] == pytest.approx(1.0)
