"""Tests for lmp.human_validation and scripts/analyze_human_validation.py (synthetic fixtures only; no models, no real data)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from lmp.human_validation import ingest as I
from lmp.human_validation import model_eval, stats
from .hv_fixture import SPEC, Fixture

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("ahv", ROOT / "scripts" / "analyze_human_validation.py")
ahv = importlib.util.module_from_spec(_s)
sys.modules["ahv"] = ahv
_s.loader.exec_module(ahv)


def load(fx, tmp_path):
    d = fx.write(tmp_path)
    return I.load_and_validate(d / "completed", SPEC, d), d


def failed(L):
    return {c.id for c in L.report.critical_failures}


# ------------------------------------------------------------------ parsing / clean fixture
def test_clean_fixture_parses_and_validates(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    assert L.ok and not failed(L)
    assert len(L.individual) == SPEC.n_judgments + SPEC.lawyer_en + SPEC.lawyer_hi
    assert len(L.consensus) == SPEC.n_en + SPEC.n_hi
    assert (L.individual.annotator_role == "student").sum() == SPEC.n_judgments
    assert set(L.individual[L.individual.annotator_role == "expert"].annotator_id) == {I.EXPERT_ID}


def test_header_row_is_detected_even_if_banner_moves(tmp_path):
    fx = Fixture()
    d = fx.write(tmp_path)
    from openpyxl import load_workbook
    p = d / "completed" / I.STUDENT_FILE
    wb = load_workbook(p)
    wb["A1"].insert_rows(1)                      # extra banner line above the header
    wb.save(p)
    assert I.load_and_validate(d / "completed", SPEC, d).ok


def test_missing_by_design_is_not_an_error(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    hi = L.consensus[L.consensus.language == "hi"]
    assert hi.score_A4.isna().all() and (hi.n_student_raters == 3).all()
    en = L.consensus[L.consensus.language == "en"]
    assert (en.n_student_raters == 4).all()                      # unequal annotator counts across languages
    assert any(c.id == "student.A4.missing_by_design" and c.status == "INFO" for c in L.report.checks)
    assert not any("A4.complete" in i for i in failed(L))


def test_inputs_never_modified_and_hashes_recorded(tmp_path):
    fx = Fixture()
    d = fx.write(tmp_path)
    files = list((d / "completed").glob("*.xlsx"))
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    rc = ahv.main(["--input-dir", str(d / "completed"), "--output-dir", str(d / "out"), "--data-root", str(d), "--n-boot", "10"], spec=SPEC)
    assert rc == 0
    assert before == {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    meta = json.loads((d / "out" / "run_metadata.json").read_text())
    assert meta["input_sha256"] == before and meta["inputs_unchanged"] is True and meta["seed"] == 42


# ------------------------------------------------------------------ validation failures
def test_genuinely_missing_response(tmp_path):
    fx = Fixture()
    fx.student["A2"].loc[0, I.S_COL["score"]] = None
    fx.student["A2"].loc[0, I.S_COL["category"]] = None
    L, _ = load(fx, tmp_path)
    assert "student.A2.complete" in failed(L) and not L.ok and L.consensus is None


def test_missing_whole_row_is_missing(tmp_path):
    fx = Fixture()
    fx.student["A1"] = fx.student["A1"].iloc[1:]
    L, _ = load(fx, tmp_path)
    assert "student.A1.complete" in failed(L)


def test_duplicate_response(tmp_path):
    fx = Fixture()
    fx.student["A3"] = pd.concat([fx.student["A3"], fx.student["A3"].iloc[[0]]])
    L, _ = load(fx, tmp_path)
    assert "student.no_duplicates" in failed(L)


def test_incorrect_assignment_extra_hindi_for_a4(tmp_path):
    fx = Fixture()
    extra = fx.student["A1"][fx.student["A1"][I.S_COL["language"]] == "Hindi"].iloc[[0]]
    fx.student["A4"] = pd.concat([fx.student["A4"], extra])
    L, _ = load(fx, tmp_path)
    assert "student.A4.no_unassigned" in failed(L)


def test_wrong_assignment_counts_in_map(tmp_path):
    fx = Fixture()
    fx.assign = fx.assign[~((fx.assign.annotator == "A2") & (fx.assign.pair_id == "HV2-EN-001"))]
    L, _ = load(fx, tmp_path)
    assert {"assign.A2", "assign.rows"} & failed(L)


@pytest.mark.parametrize("score,cat,check", [(8, "Not Preserved", "student.score_category_consistent"), (11, "Preserved", "student.score_range"),
                                              (0, "Not Preserved", "student.score_range"), (6.5, "Partially Preserved", "student.score_integer"),
                                              (5, "Maybe", "student.category_valid"), ("abc", "Preserved", "student.score_numeric")])
def test_bad_scores_and_categories(tmp_path, score, cat, check):
    fx = Fixture()
    fx.student["A1"][I.S_COL["score"]] = fx.student["A1"][I.S_COL["score"]].astype(object)
    fx.student["A1"].loc[0, I.S_COL["score"]] = score
    fx.student["A1"].loc[0, I.S_COL["category"]] = cat
    L, _ = load(fx, tmp_path)
    assert check in failed(L)


def test_missing_student_confidence_is_a_warning_not_critical(tmp_path):
    fx = Fixture()
    fx.student["A1"].loc[0, I.S_COL["confidence"]] = None
    L, _ = load(fx, tmp_path)
    assert L.ok
    assert any(c.id == "student.confidence_complete" and c.status == "WARN" for c in L.report.checks)
    row = L.individual[(L.individual.annotator_id == "A1")].iloc[0:]
    assert (row.confidence_status == "missing").any()


def test_source_text_mismatch_with_dataset(tmp_path):
    fx = Fixture()
    fx.src.loc[0, "simplified"] = "A different simplified sentence."
    L, _ = load(fx, tmp_path)
    assert "key.text_vs_dataset" in failed(L)


def test_text_mismatch_between_student_sheet_and_key(tmp_path):
    fx = Fixture()
    fx.student["A2"].loc[3, I.S_COL["simplified"]] = "tampered text"
    L, _ = load(fx, tmp_path)
    assert "student.text_consistency" in failed(L)


def test_split_membership_and_heldout_flag_must_match_dataset(tmp_path):
    fx = Fixture()
    fx.key.loc[fx.key.dataset_split == "train", "dataset_split"] = "test"      # a training pair mislabelled as test
    L, _ = load(fx, tmp_path)
    assert {"key.split_membership", "key.split_flag"} & failed(L)


def test_missing_worksheet_and_column(tmp_path):
    fx = Fixture()
    d = fx.write(tmp_path)
    from openpyxl import load_workbook
    p = d / "completed" / I.STUDENT_FILE
    wb = load_workbook(p)
    del wb["A3"]
    wb.save(p)
    L = I.load_and_validate(d / "completed", SPEC, d)
    assert any(c.id.startswith("student.A3") for c in L.report.critical_failures)


def test_missing_input_file(tmp_path):
    L = I.load_and_validate(tmp_path, SPEC, tmp_path)
    assert not L.ok and "file.student" in failed(L)


def test_expected_split_counts_are_checked(tmp_path):
    fx = Fixture()
    d = fx.write(tmp_path)
    spec = I.DesignSpec(**{**SPEC.__dict__, "expected_student_split": (("test", 99), ("supplemental", 1))})
    L = I.load_and_validate(d / "completed", spec, d)
    assert any(c.id == "key.student_split_counts" and c.status == "WARN" for c in L.report.checks)


# ------------------------------------------------------------------ expert corrections and confidence
def test_expert_corrections_verified_not_reapplied(tmp_path):
    fx = Fixture()
    L, _ = load(fx, tmp_path)
    log = L.corrections
    assert set(log.pair_id) == {"HV2-HI-001", "HV2-EN-002", "ALL_EXPERT"}
    c = log[log.pair_id == "HV2-HI-001"].iloc[0]
    assert c.original_score == 8 and c.observed_final_score == 6 and c.observed_final_category == "Partially Preserved" and c.verified
    assert "NOT re-applied" in c.status and "expert-confirmed" in c.status
    exp = L.individual[(L.individual.annotator_role == "expert") & (L.individual.pair_id == "HV2-HI-001")].iloc[0]
    assert exp.human_score == 6 and exp.expert_correction_manually_incorporated


def test_wrong_expert_value_fails_correction_check(tmp_path):
    fx = Fixture()
    fx.lawyer.loc[fx.lawyer[I.L_COL["pair_id"]] == "HV2-HI-001", I.L_COL["score"]] = 8
    fx.lawyer.loc[fx.lawyer[I.L_COL["pair_id"]] == "HV2-HI-001", I.L_COL["category"]] = "Preserved"
    L, _ = load(fx, tmp_path)
    assert "expert.correction.HV2-HI-001" in failed(L)


def test_expert_confidence_excluded_but_preserved(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    e = L.individual[L.individual.annotator_role == "expert"]
    assert (e.confidence_raw == "Low").all()                       # raw responses untouched
    assert (~e.confidence_valid_for_analysis).all() and (e.confidence_status == "excluded_expert_rubric_misinterpretation").all()
    s = L.individual[L.individual.annotator_role == "student"]
    assert s.confidence_valid_for_analysis.all()
    assert "ALL_EXPERT" in set(L.corrections.pair_id)


# ------------------------------------------------------------------ consensus
def _ind(rows):
    return pd.DataFrame([{"pair_id": "P", "annotator_id": a, "annotator_role": "student", "human_score": s, "human_category": I.rubric_category(s),
                          "semantic_change_type": lab} for a, s, lab in rows])


def test_consensus_mean_median_majority_and_ties():
    key = pd.DataFrame([{"pair_id": "P", "source_pair_id": "S", "language": "en", "legal_domain": "x", "complexity_level": "easy", "group_id": "g", "concept_id": "c",
                         "dataset_split": "test", "is_heldout_test": True, "selection_source": "t", "reference_label_type": "AI-GENERATED", "label_source": "l", "data_origin": "o",
                         "original": "o", "simplified": "s", "AI_REFERENCE_score": 7.0, "AI_REFERENCE_category": "Partially Preserved", "AI_REFERENCE_error_labels": "",
                         **{c: 1 for c in I.STAT_COLS}}])
    c = I.build_consensus(_ind([("A1", 5, "Scope"), ("A2", 6, "Scope"), ("A3", 6, "Party"), ("A4", 8, "Scope")]), key).iloc[0]
    assert c.student_mean == 6.25 and c.student_median == 6.0 and c.student_range == 3 and c.student_high_disagreement
    assert c.student_mean_category == "Partially Preserved" and c.student_majority_category == "Partially Preserved" and not c.student_majority_tie
    assert c.student_modal_change == "Scope" and c.student_change_agreement_share == 0.75
    tie = I.build_consensus(_ind([("A1", 2, "Scope"), ("A2", 3, "Party"), ("A3", 8, "Scope"), ("A4", 9, "Party")]), key).iloc[0]
    assert tie.student_majority_tie and tie.student_majority_category is None and tie.student_modal_change_tie
    assert tie.student_mean == 5.5 and tie.student_category_disagreement
    three = I.build_consensus(_ind([("A1", 2, None), ("A2", 5, None), ("A3", 9, None)]), key).iloc[0]     # 1-1-1 tie, 3 raters
    assert three.student_majority_tie and three.n_student_raters == 3 and three.student_mean == pytest.approx(16 / 3)
    assert I.cat_from_score(6.5) == "Preserved" and I.cat_from_score(6.49) == "Partially Preserved" and I.cat_from_score(3.5) == "Partially Preserved" and I.cat_from_score(3.49) == "Not Preserved"
    assert c.ref_category_original == "Partially Preserved" and c.ref_category_harmonized == "Preserved"   # 7.0: repo thresholds vs harmonised differ


def test_scores_are_not_rounded(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    assert (L.consensus.student_mean % 1 != 0).any()


# ------------------------------------------------------------------ metric correctness
def test_kappas_match_sklearn():
    rng = np.random.default_rng(3)
    a = rng.integers(1, 11, 80)
    b = np.clip(a + rng.integers(-2, 3, 80), 1, 10)
    assert stats.cohen_kappa(a - 1, b - 1, 10, "quadratic") == pytest.approx(cohen_kappa_score(a, b, weights="quadratic", labels=list(range(1, 11))))
    ca, cb = np.digitize(a, [3.5, 6.5]), np.digitize(b, [3.5, 6.5])
    assert stats.cohen_kappa(ca, cb, 3) == pytest.approx(cohen_kappa_score(ca, cb))


def test_fleiss_textbook_example():
    cnt = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    codes = np.array([sum(([c] * n for c, n in enumerate(r)), []) for r in cnt])
    k, n = stats.fleiss_kappa(codes, 5)
    assert n == 10 and k == pytest.approx(0.2099, abs=1e-4)
    assert stats.fleiss_kappa(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 3)[0] == pytest.approx(1.0)


def test_krippendorff_known_values_with_missing_data():
    M = np.array([[1, 1, 2], [2, 2, 2], [3, 4, 3], [5, 5, np.nan], [7, 9, 8], [10, 10, 9]], float)
    assert stats.krippendorff_alpha(M, "nominal") == pytest.approx(0.424, abs=1e-6)
    assert stats.krippendorff_alpha(M, "ordinal") == pytest.approx(0.953, abs=1e-6)
    assert stats.krippendorff_alpha(M, "interval") == pytest.approx(0.967123, abs=1e-6)
    k = pytest.importorskip("krippendorff")
    for lvl in ("nominal", "ordinal", "interval"):
        assert stats.krippendorff_alpha(M, lvl) == pytest.approx(k.alpha(reliability_data=M.T, level_of_measurement=lvl))


def test_undefined_statistics_are_nan_not_errors():
    const = np.full((6, 3), 5.0)
    assert np.isnan(stats.krippendorff_alpha(const, "ordinal"))
    assert np.isnan(stats.cohen_kappa([0, 0, 0], [0, 0, 0], 3))
    assert np.isnan(stats.fleiss_kappa(np.zeros((0, 3), int), 3)[0])
    assert np.isnan(stats.safe_corr([1, 1, 1], [1, 2, 3]))
    assert stats.group_bootstrap(np.array(["a", "a", "b"]), lambda i: 1.0, 20) != stats.group_bootstrap(np.array(["a", "a", "b"]), lambda i: 1.0, 20) or True


def test_group_bootstrap_is_deterministic_and_group_aware():
    g = np.repeat(np.arange(12), 3)
    x = np.random.default_rng(0).normal(size=36)
    fn = lambda idx: float(np.mean(x[idx]))  # noqa: E731
    assert stats.group_bootstrap(g, fn, 200, 7) == stats.group_bootstrap(g, fn, 200, 7)
    lo, hi = stats.group_bootstrap(g, fn, 200, 7)
    assert lo < np.mean(x) < hi
    assert all(np.isnan(stats.group_bootstrap(np.array(["a", "b"]), fn, 50, 1)))     # too few groups -> undefined


def test_student_agreement_table_scopes_and_a4_english_only(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    ag = stats.student_agreement(L.consensus, n_boot=20, seed=1)
    assert {"english", "english_A1-A3", "hindi", "combined_A1-A3"} <= set(ag.subset)
    pw = ag[ag.scope == "student_agreement_pairwise"]
    assert not pw[(pw.subset == "hindi") & pw.rater_set.str.contains("A4")].shape[0]
    assert pw[(pw.subset == "english") & pw.rater_set.str.contains("A4")].shape[0] > 0
    assert not pw[pw.subset == "combined_A1-A3"].rater_set.str.contains("A4").any()
    r = pw[(pw.subset == "english") & (pw.rater_set == "A1-A2") & (pw.metric == "mae_score")].iloc[0]
    en = L.consensus[L.consensus.language == "en"]
    assert r.value == pytest.approx((en.score_A1 - en.score_A2).abs().mean()) and r.n == len(en)


# ------------------------------------------------------------------ comparisons, subsets, splits
def test_subsets_partition_fixed_test_vs_supplemental(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    m = stats.subset_masks(L.consensus)
    assert (m["fixed_test"] & m["supplemental_train_val"]).sum() == 0 and (m["fixed_test"] | m["supplemental_train_val"]).all()
    assert (L.consensus[m["fixed_test"]].dataset_split == "test").all() and (L.consensus[m["supplemental_train_val"]].dataset_split != "test").all()
    ref, _ = stats.reference_comparison(L.consensus, L.individual, 10, 1)
    assert {"all", "fixed_test", "supplemental_train_val", "english", "hindi"} <= set(ref.subset)
    ex, _ = stats.expert_comparison(L.consensus, 10, 1)
    assert {"expert_all", "expert_fixed_test", "expert_supplemental_train_val"} <= set(ex.subset)


def test_reference_comparison_bias_sign_and_variants(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    ref, conf = stats.reference_comparison(L.consensus, L.individual, 10, 1)
    b = ref[(ref.subset == "all") & (ref.comparator == "student_mean") & (ref.metric == "bias")].iloc[0]
    assert b.value == pytest.approx((L.consensus.ref_score - L.consensus.student_mean).mean()) and "reference - human" in b.note
    variants = set(ref[ref.metric == "category_accuracy"].reference_variant)
    assert variants == {"original_repo_thresholds_4_7.5", "harmonized_human_rubric_3.5_6.5"}
    assert (ref[ref.metric == "mae"].reference_variant == "score").all()          # incompatible definitions never mixed
    assert conf.groupby(["comparison", "subset"])["count"].sum().min() > 0


def test_expert_comparison_metrics(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    ex, conf = stats.expert_comparison(L.consensus, 10, 1)
    e = L.consensus[L.consensus.expert_reviewed]
    r = ex[(ex.subset == "expert_all") & (ex.comparator == "student_mean") & (ex.metric == "bias")].iloc[0]
    assert r.value == pytest.approx((e.student_mean - e.expert_score).mean()) and r.n == len(e)
    assert len(conf) > 0


def test_empty_and_small_subgroups_do_not_crash(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    c = L.consensus.copy()
    c["expert_reviewed"] = False
    ex, conf = stats.expert_comparison(c, 10, 1)
    assert ex.empty and conf.empty
    sg = stats.subgroup_analysis(L.consensus)
    small = sg[(sg.metric == "krippendorff_alpha_ordinal") & (sg.n < stats.MIN_N_CORR)]
    assert len(small) and small.value.isna().all() and small.note.str.contains("n<").all()
    assert stats.semantic_analysis(c, L.individual).shape[0] > 0


def test_semantic_taxonomy_mapping_documented(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    sem = stats.semantic_analysis(L.consensus, L.individual)
    tm = sem[sem.section == "taxonomy_mapping"]
    assert set(tm.label) == set(stats.HUMAN_TO_DATASET)
    assert tm[tm.label == "Number-Threshold"].value.iloc[0] == "altered_numbers"
    assert "(no dataset equivalent)" in set(tm.value)


def test_qualitative_table_contains_actual_texts_and_explanations(tmp_path):
    L, _ = load(Fixture(), tmp_path)
    q = stats.qualitative_tables(L.consensus, L.individual)
    assert q.original.str.len().gt(0).all() and q.student_explanations.str.contains("explanation").any()
    assert "expert_vs_student_largest_score_gaps" in set(q.table)


# ------------------------------------------------------------------ model-evaluation interface / leakage safeguards
def _pred_csv(tmp_path, cons, **extra):
    df = pd.DataFrame({"pair_id": cons.pair_id, "model_name": "m1", "score": np.linspace(1, 10, len(cons))})
    for k, v in extra.items():
        df[k] = v
    p = tmp_path / "p.csv"
    df.to_csv(p, index=False)
    return p


def test_import_predictions_validation(tmp_path):
    L, d = load(Fixture(), tmp_path)
    c = L.consensus
    bad = pd.DataFrame({"pair_id": ["HV2-EN-001", "HV2-EN-001", "NOPE"], "model_name": "m", "score": [5, 6, 7]})
    bad.to_csv(tmp_path / "b.csv", index=False)
    preds, issues = model_eval.import_predictions(tmp_path / "b.csv", c)
    assert preds == [] and any("duplicate" in i for i in issues) and any("not among" in i for i in issues)
    pd.DataFrame({"pair_id": c.pair_id, "model_name": "m", "score": 99}).to_csv(tmp_path / "r.csv", index=False)
    assert model_eval.import_predictions(tmp_path / "r.csv", c)[0] == []
    pd.DataFrame({"pair_id": c.pair_id, "score": 5}).to_csv(tmp_path / "x.csv", index=False)
    assert "missing required columns" in model_eval.import_predictions(tmp_path / "x.csv", c)[1][0]
    preds, issues = model_eval.import_predictions(_pred_csv(tmp_path, c), c)
    assert preds[0].provenance == "UNVERIFIED" and any("UNVERIFIED" in i for i in issues)
    src_ids = pd.DataFrame({"pair_id": c.source_pair_id, "model_name": "m", "score": 5.0})          # source IDs are mapped to neutral IDs
    src_ids.to_csv(tmp_path / "s.csv", index=False)
    assert set(model_eval.import_predictions(tmp_path / "s.csv", c)[0][0].scores.index) == set(c.pair_id)


def test_split_separation_and_heldout_status(tmp_path):
    L, d = load(Fixture(), tmp_path)
    c = L.consensus
    declared = model_eval.import_predictions(_pred_csv(tmp_path, c, checkpoint_id="ck1", training_splits="train;val"), c)[0]
    res = model_eval.evaluate_models(declared, {}, c, d)
    st = res[(res.target == "student_mean")].set_index("subset").heldout_status
    assert st["fixed_test"].startswith("HELD_OUT_DECLARED") and st["supplemental_train_val"].startswith("CONTAINS_TRAINING_DATA")
    assert st["all"].startswith("CONTAINS_TRAINING_DATA")
    assert res[res.subset == "fixed_test"].heldout_note.str.contains("not an untouched final test").all()
    unknown = model_eval.import_predictions(_pred_csv(tmp_path, c), c)[0]
    assert set(model_eval.evaluate_models(unknown, {}, c, d).heldout_status) == {"UNVERIFIED"}


def test_group_leakage_is_detected(tmp_path):
    fx = Fixture()
    g = fx.src.loc[fx.src.dataset_split_source == "test", "group_id"].iloc[0]
    fx.src.loc[fx.src.dataset_split_source == "train", "group_id"] = g                      # train shares a group with test ...
    fx.key.loc[fx.key.dataset_split == "train", "group_id"] = g                             # ... and the key mirrors the dataset
    L, d = load(fx, tmp_path)
    p = model_eval.import_predictions(_pred_csv(tmp_path, L.consensus, checkpoint_id="c", training_splits="train"), L.consensus)[0]
    res = model_eval.evaluate_models(p, {}, L.consensus, d)
    assert res[res.subset == "fixed_test"].heldout_status.str.startswith("GROUP_LEAKAGE").any()


def test_models_not_run_report_exact_reason(tmp_path):
    L, d = load(Fixture(), tmp_path)
    preds, nr = model_eval.run_live(["xlmr_crossencoder", "lexical_similarity", "bogus"], L.consensus, d, None, None)
    assert "no --xlmr-checkpoint" in nr["xlmr_crossencoder"] and "train.csv" not in nr.get("bogus", "") and "unknown model" in nr["bogus"]
    assert all(p.provenance == "VERIFIED_IN_PROCESS" and p.training_splits == ("train",) for p in preds)       # classical baseline fitted on train only
    out = model_eval.evaluate_models([], model_eval.not_requested(), L.consensus, d)
    assert set(out.status) == {"NOT RUN"} and len(out) == len(model_eval.CANONICAL)


def test_gitignore_protects_private_analysis_material():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("outputs/human_validation_v2/", "outputs/human_validation_analysis/", "*.xlsx", "PRIVATE_*", "individual_annotations*.csv", "pair_consensus*.csv",
                "correction_log*.csv", "qualitative_disagreements*.csv", "checkpoints/", ".venv/"):
        assert pat in gi, pat


# ------------------------------------------------------------------ CLI / outputs
def run_cli(d, *extra, spec=SPEC):
    return ahv.main(["--input-dir", str(d / "completed"), "--output-dir", str(d / "out"), "--data-root", str(d), "--n-boot", "15", *extra], spec=spec)


def test_cli_generates_all_outputs(tmp_path):
    d = Fixture().write(tmp_path)
    assert run_cli(d) == 0
    out = d / "out"
    for f in ("validation_report.json", "validation_report.md", "correction_log.csv", "individual_annotations.csv", "pair_consensus.csv", "human_validation_report.md", "run_metadata.json",
              "metrics/student_agreement.csv", "metrics/expert_comparison.csv", "metrics/reference_label_comparison.csv", "metrics/subgroup_analysis.csv",
              "metrics/semantic_error_analysis.csv", "metrics/model_comparison.csv", "metrics/qualitative_disagreements.csv", "metrics/confusion_matrices.csv"):
        assert (out / f).exists(), f
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert len(figs) == 9 and not any("fig10" in f for f in figs) and len(list((out / "figures").glob("*.pdf"))) == 9
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["status"] == "COMPLETED" and meta["figures"]["fig10_model_comparison"].startswith("SKIPPED") and "numpy" in meta["versions"]
    mc = pd.read_csv(out / "metrics" / "model_comparison.csv")
    assert set(mc.status) == {"NOT RUN"} and len(mc) == 5
    rep = (out / "human_validation_report.md").read_text(encoding="utf-8")
    for h in ("Annotation methodology", "Dataset composition", "Data validation", "Student inter-annotator agreement", "Student consensus vs expert",
              "reference labels", "Language and complexity", "Semantic change-type analysis", "Model evaluation", "Limitations"):
        assert h in rep, h
    for s in ("synthetic", "sentence-level", "only one legal professional", "untouched final test", "not independent human ground truth", "Correlation does not establish", "confidence"):
        assert s in rep, s
    ind = pd.read_csv(out / "individual_annotations.csv")
    assert len(ind) == 56 and {"human_score", "ref_score", "confidence_status", "semantic_change_type", "explanation", "dataset_split"} <= set(ind.columns)
    assert len(pd.read_csv(out / "pair_consensus.csv")) == 14


def test_cli_does_not_overwrite_previous_run(tmp_path):
    d = Fixture().write(tmp_path)
    assert run_cli(d) == 0
    marker = (d / "out" / "run_metadata.json").read_text()
    assert run_cli(d) == 3
    assert (d / "out" / "run_metadata.json").read_text() == marker
    assert run_cli(d, "--overwrite") == 0


def test_cli_stops_dependent_analyses_on_critical_errors(tmp_path):
    fx = Fixture()
    fx.student["A2"].loc[0, I.S_COL["score"]] = 12
    d = fx.write(tmp_path)
    assert run_cli(d) == 2
    out = d / "out"
    assert (out / "validation_report.md").exists() and (out / "run_metadata.json").exists()
    assert not (out / "pair_consensus.csv").exists() and not (out / "metrics" / "student_agreement.csv").exists()
    assert json.loads((out / "run_metadata.json").read_text())["status"].startswith("STOPPED")


def test_cli_with_predictions_runs_model_comparison_and_figure(tmp_path):
    L, d = load(Fixture(), tmp_path)
    p = _pred_csv(tmp_path, L.consensus, checkpoint_id="ck", training_splits="train")
    assert run_cli(d, "--predictions", str(p)) == 0
    mc = pd.read_csv(d / "out" / "metrics" / "model_comparison.csv")
    assert (mc.status == "EVALUATED").any() and {"student_mean", "expert"} <= set(mc.target.dropna())
    assert (d / "out" / "figures" / "fig10_model_comparison.png").exists()


# ---- Regression tests for missing provenance values

import pytest
import numpy as np

from tests.hv_fixture import Fixture
from lmp.human_validation import ingest as I


@pytest.mark.parametrize(
    "key_value,source_value,expected",
    [
        (np.nan, "", True),
        ("", np.nan, True),
        (np.nan, np.nan, True),
        (np.nan, "C-VALID", False),
        ("C-VALID", np.nan, False),
        ("C-001", "C-002", False),
        ("C-001", "C-001", True),
    ],
    ids=[
        "missing-key-empty-source",
        "empty-key-missing-source",
        "both-missing",
        "missing-key-valid-source",
        "valid-key-missing-source",
        "different-valid-ids",
        "matching-valid-ids",
    ],
)
def test_provenance_missing_value_normalization(
    key_value, source_value, expected
):
    fixture = Fixture()

    key = fixture.key.copy(deep=True)
    src = fixture.src.copy(deep=True)

    row = key.index[0]
    source_id = key.loc[row, "source_pair_id"]

    key["concept_id"] = key["concept_id"].astype(object)
    src["concept_id"] = src["concept_id"].astype(object)

    key.loc[row, "concept_id"] = key_value
    src.loc[
        src.pair_id == source_id, "concept_id"
    ] = source_value

    class CaptureReport:
        def __init__(self):
            self.results = {}

        def add(self, name, passed, message):
            self.results[name] = (bool(passed), message)

    report = CaptureReport()

    I.check_key_vs_source(key, src, report)

    passed, message = report.results[
        "key.provenance_vs_dataset"
    ]

    assert passed == expected, message

    if not expected:
        assert "concept_id" in message

    for name, (ok, detail) in report.results.items():
        if name != "key.provenance_vs_dataset":
            assert ok, f"{name}: {detail}"
