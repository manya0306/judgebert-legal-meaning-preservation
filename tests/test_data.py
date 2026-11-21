import numpy as np, pandas as pd, pytest
from lmp.config import ROOT
from lmp.data.io import load_pairs, validate, consolidate_annotations
from lmp.data.splits import document_split, domain_holdout_split
from lmp.data.agreement import pairwise_score_agreement, category_kappa, error_label_kappa
from lmp.data.augment import augment_with_sanity_pairs

DATA = ROOT / "data/synthetic/synthetic_pairs.csv"

def test_synthetic_dataset_is_labelled_synthetic():
    df = load_pairs(DATA)
    assert len(df) == 48 and (df.data_origin == "synthetic").all()
    assert set(df.language) == {"en", "hi"} and df.lmp_score.between(1, 10).all()

def test_validation_rejects_bad_rows():
    df = pd.read_csv(DATA)
    with pytest.raises(ValueError): validate(df.assign(lmp_score=11))
    with pytest.raises(ValueError): validate(df.assign(language="fr"))
    with pytest.raises(ValueError): validate(df.assign(error_labels="made_up_label"))
    with pytest.raises(ValueError): validate(df.drop(columns=["source_doc_id"]))

def test_document_split_has_no_leakage():
    df = load_pairs(DATA)
    tr, va, te = document_split(df, seed=1)
    assert not set(tr.source_doc_id) & set(te.source_doc_id) and not set(tr.source_doc_id) & set(va.source_doc_id) and not set(va.source_doc_id) & set(te.source_doc_id)
    assert len(tr) + len(va) + len(te) == len(df)

def test_domain_holdout_never_trains_on_heldout_domain():
    df = load_pairs(DATA)
    tr, va, te = domain_holdout_split(df, "tenancy")
    assert (te.legal_domain == "tenancy").all() and not (tr.legal_domain == "tenancy").any() and not (va.legal_domain == "tenancy").any()

def test_augmentation_only_uses_training_rows():
    df = load_pairs(DATA); tr, _, _ = document_split(df, seed=3)
    aug = augment_with_sanity_pairs(tr, 3)
    new = aug[aug.pair_id.str.startswith("AUG-")]
    assert len(new) > 0 and set(new.source_doc_id) <= set(tr.source_doc_id)
    assert (new[new.pair_id.str.startswith("AUG-ID")].original == new[new.pair_id.str.startswith("AUG-ID")].simplified).all()

def _annotations(noise):
    rng = np.random.RandomState(0); df = load_pairs(DATA); rows = []
    for ann in ("A", "B"):
        for _, r in df.iterrows():
            rows.append({"pair_id": r.pair_id, "annotator_id": ann, "lmp_score": float(np.clip(r.lmp_score + rng.normal(0, noise), 1, 10)), "error_labels": r.error_labels})
    return pd.DataFrame(rows)

def test_agreement_high_for_low_noise_and_lower_for_high_noise():
    lo, hi = pairwise_score_agreement(_annotations(0.3)), pairwise_score_agreement(_annotations(4.0))
    assert lo.spearman.iloc[0] > 0.9 and lo.spearman.iloc[0] > hi.spearman.iloc[0]
    assert len(category_kappa(_annotations(0.3))) == 1 and len(error_label_kappa(_annotations(0.3))) > 0

def test_consolidation_majority_labels():
    ann = pd.DataFrame({"pair_id": ["p"] * 3, "annotator_id": list("ABC"), "lmp_score": [2, 4, 3], "error_labels": ["omission|altered_numbers", "omission", ""]})
    out = consolidate_annotations(ann).iloc[0]
    assert out.lmp_score == 3.0 and out.error_labels == "omission" and out.n_annotators == 3
