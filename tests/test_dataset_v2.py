import numpy as np, pandas as pd, pytest
from lmp.data.build import build_expanded, load_concepts
from lmp.data.io import load_pairs, validate
from lmp.data import qa
from lmp.data.splits import grouped_split, grouped_kfold, merge_near_duplicate_groups, assert_no_leakage, split_report
from lmp.data.augment import augment_training_set, assert_augmentation_isolated
from lmp.data.schema import DATASET_TIERS

@pytest.fixture(scope="module")
def df(): return build_expanded()

def test_build_counts_and_tiers(df):
    assert len(df) >= 500 and set(df.dataset_tier) <= set(DATASET_TIERS)
    assert (df.dataset_tier == "human_annotated").sum() == 0 and (df.dataset_tier == "external").sum() == 0   # no fabricated human/external data
    assert set(df.language) == {"en", "hi"} and df.pair_id.is_unique
    assert not df[df.dataset_tier == "ai_generated"].label_source.isin(["human", "human_adjudicated"]).any()

def test_every_concept_has_both_languages_and_one_group(df):
    g = df[df.concept_id != ""].groupby("concept_id")
    assert (g.language.nunique() == 2).all() and (g.group_id.nunique() == 1).all()

def test_category_matches_score(df):
    assert qa.run_qa(df).query("check == 'score_category_mismatch'").empty

def _row(**kw):
    base = dict(pair_id="X1", language="en", legal_domain="tenancy", original="The tenant shall pay rent monthly.", simplified="The tenant pays rent each month.", lmp_score=9.0,
                category="Preserved", error_labels="", source_doc_id="d", group_id="g", concept_id="", data_origin="ai_generated", dataset_tier="ai_generated", generation_strategy="faithful_paraphrase")
    base.update(kw); return base

def test_qa_flags_invalid_label_and_mismatch_and_range():
    d = pd.DataFrame([_row(pair_id="a", error_labels="bogus"), _row(pair_id="b", category="Not Preserved"), _row(pair_id="c", lmp_score=11.0)])
    f = qa.run_qa(d); assert {"invalid_label", "score_category_mismatch", "invalid_score"} <= set(f.check)

def test_qa_detects_exact_duplicates_and_contradictory_scores():
    d = pd.DataFrame([_row(pair_id="a"), _row(pair_id="b", lmp_score=2.0, category="Not Preserved", error_labels="contradiction")])
    f = qa.run_qa(d); assert "exact_duplicate" in set(f.check) and "contradictory_annotation" in set(f.check)

def test_qa_hindi_orthography_and_script():
    d = pd.DataFrame([_row(pair_id="h1", language="hi", original="किरायेदार किराया देगा।", simplified="tenant किराया देगा।"),
                      _row(pair_id="h2", language="hi", original="किरायेदार किराया किराया देगा।", simplified="किरायेदार देगा।")])
    f = qa.run_qa(d); assert {"hindi_script_ratio"} & set(f[f.pair_id == "h1"].check) or "hindi_orthography" in set(f.check)
    assert "hindi_orthography" in set(f[f.pair_id == "h2"].check)

def test_qa_missing_values_rejected_by_validate():
    with pytest.raises(ValueError): validate(pd.DataFrame([_row(original="  ")]))
    with pytest.raises(ValueError): validate(pd.DataFrame([_row(language="fr")]))
    with pytest.raises(ValueError): validate(pd.DataFrame([_row(lmp_score=np.nan)]))

def test_quality_filter_never_silent(df):
    f = qa.run_qa(df); kept, rej = qa.quality_filter(df, f, {"lossy_variant_scored_preserved"}); assert len(kept) + len(rej) == len(df) and "rejection_reason" in rej

def test_lexical_cue_reliance_runs(df): assert "spearman_jaccard_vs_score" in qa.lexical_cue_reliance(df)

def test_grouped_split_leak_free_and_reproducible(df):
    a = grouped_split(df, seed=3); b = grouped_split(df, seed=3); assert_no_leakage(*a)
    assert list(a[2].pair_id) == list(b[2].pair_id) and all(len(x) > 0 for x in a)
    rep = split_report(*a); assert rep["leakage_check"].startswith("passed")

def test_translation_pairs_stay_together(df):
    tr, va, te = grouped_split(df, seed=1)
    for s in (tr, va, te):
        for cid, g in s[s.concept_id != ""].groupby("concept_id"): assert g.language.nunique() == 2
    assert not (set(tr.concept_id) & set(te.concept_id) - {""})

def test_near_duplicate_groups_are_merged():
    a = _row(pair_id="a", group_id="G1", original="The landlord shall give the tenant at least twenty four hours notice before entering the premises.")
    b = _row(pair_id="b", group_id="G2", original="The landlord shall give the tenant at least twenty four hours notice before entering the premises.", simplified="Different simplified text here.")
    c = _row(pair_id="c", group_id="G3", original="Completely unrelated sentence about copyright duration of literary works after death.")
    sg = merge_near_duplicate_groups(pd.DataFrame([a, b, c])); assert sg.iloc[0] == sg.iloc[1] != sg.iloc[2]

def test_leakage_assertion_fires():
    a = pd.DataFrame([_row(pair_id="a", split_group="G1")]); b = pd.DataFrame([_row(pair_id="b", split_group="G1")])
    with pytest.raises(AssertionError): assert_no_leakage(a, b)

def test_grouped_kfold_disjoint_groups(df):
    seen = []
    for f, a, b in grouped_kfold(df, 4, 0):
        assert not (set(df.iloc[a].group_id) & set(df.iloc[b].group_id)); seen += list(b)
    assert sorted(seen) == list(range(len(df)))

def test_augmentation_only_in_train_and_guard(df):
    tr, va, te = grouped_split(df, seed=2); aug = augment_training_set(tr, 0, 0.2); assert_augmentation_isolated(aug, va, te)
    assert aug.is_augmented.sum() > 0 and set(aug[aug.is_augmented].label_source) == {"augmentation_rule"} and not aug[~aug.is_augmented].pair_id.str.startswith("AUG").any()
    leaked = pd.concat([va, aug[aug.is_augmented].head(1)]); 
    with pytest.raises(AssertionError): assert_augmentation_isolated(aug, leaked)

def test_loader_roundtrip_expanded():
    d = load_pairs("data/expanded/pairs_consolidated.csv"); assert {"group_id", "dataset_tier", "annotation_status"} <= set(d.columns)
