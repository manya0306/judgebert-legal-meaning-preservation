"""Optional, ISOLATED training-set augmentation (OFF by default; evaluated as an ablation, see docs/EXPERIMENTS.md).

Audit of the previous implementation (docs/PROJECT_AUDIT.md section 6):
  * it added (original, original)->10 and (original, unrelated sentence)->1 rows for ~100% of the training size;
  * the labels are trivially separable and sit at the extremes of the scale, so a model can reach low loss on them by detecting
    "same text" / "different topic" while learning nothing about SUBTLE legal changes (the actual task);
  * rows were not structurally tied to their parent group and could not be separated from real rows afterwards;
  * the app trained on augmented data.
This version: (1) never touches val/test (callers pass the TRAIN split only), (2) marks each augmented row (`is_augmented=True`,
`parent_pair_id`, inherited `group_id`/`split_group`, `label_source=augmentation_rule`, `dataset_tier=augmented`), (3) caps the added
fraction, (4) offers `assert_augmentation_isolated` as an automated guard, (5) draws unrelated sentences from the training pool only.
These rows are rule-generated, NOT annotations of any kind.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

IDENTITY_SCORE = 10.0
UNRELATED_SCORE = 1.0


def augment_training_set(train: pd.DataFrame, seed: int = 42, frac: float = 0.25, strategies=("identity", "unrelated")) -> pd.DataFrame:
    """Return train + augmented rows. `frac` = fraction of training rows used as parents (per strategy)."""
    train = train.copy()
    if "is_augmented" not in train.columns:
        train["is_augmented"] = False
    train["parent_pair_id"] = train.get("parent_pair_id", pd.Series("", index=train.index)).fillna("")
    parents = train[~train["is_augmented"]].sample(frac=min(1.0, max(0.0, frac)), random_state=seed)
    rng = np.random.RandomState(seed)
    rows = []
    for _, r in parents.iterrows():
        base = r.to_dict()
        common = {"is_augmented": True, "parent_pair_id": r.pair_id, "dataset_tier": "augmented", "data_origin": "synthetic",
                  "label_source": "augmentation_rule", "annotation_status": "rule_generated_not_annotated", "qa_flags": "", "error_labels": ""}
        if "identity" in strategies:
            rows.append(base | common | {"pair_id": "AUG-ID-" + r.pair_id, "simplified": r.original, "lmp_score": IDENTITY_SCORE,
                                         "category": "Preserved", "generation_strategy": "aug_identity"})
        if "unrelated" in strategies:
            pool = train[(~train["is_augmented"]) & (train.language == r.language) & (train.legal_domain != r.legal_domain)]
            if len(pool):
                rows.append(base | common | {"pair_id": "AUG-UN-" + r.pair_id, "simplified": pool.iloc[rng.randint(len(pool))].simplified,
                                             "lmp_score": UNRELATED_SCORE, "category": "Not Preserved", "generation_strategy": "aug_unrelated"})
    if not rows:
        return train
    return pd.concat([train, pd.DataFrame(rows)], ignore_index=True)


def augment_with_sanity_pairs(train: pd.DataFrame, seed: int = 42, frac: float = 0.5) -> pd.DataFrame:
    """Backward-compatible name for the previous API (both strategies)."""
    return augment_training_set(train, seed, frac)


def assert_augmentation_isolated(aug_train: pd.DataFrame, *eval_splits: pd.DataFrame) -> None:
    """Guard: augmented rows exist only in the training frame; their parents are training rows; no augmented sentence pair or
    text leaked into any evaluation split."""
    aug = aug_train[aug_train["is_augmented"]]
    real_ids = set(aug_train.loc[~aug_train["is_augmented"], "pair_id"])
    assert set(aug["parent_pair_id"]) <= real_ids, "augmented row whose parent is not in the training split"
    for ev in eval_splits:
        if "is_augmented" in ev.columns:
            assert not ev["is_augmented"].any(), "augmented row found in an evaluation split"
        assert not (set(aug.pair_id) & set(ev.pair_id)), "augmented pair_id in evaluation split"
        ev_pairs = set(zip(ev.original.str.lower(), ev.simplified.str.lower()))
        assert not (set(zip(aug.original.str.lower(), aug.simplified.str.lower())) & ev_pairs), "augmented text pair in evaluation split"
        if "split_group" in aug.columns and "split_group" in ev.columns:
            assert not (set(aug.split_group) & set(ev.split_group)), "augmented row shares a group with an evaluation split"
