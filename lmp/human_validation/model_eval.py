"""Optional evaluation of model predictions against human judgments (never required by the annotation analysis).

No training or downloading happens here. Classical baselines may be re-fitted IN MEMORY on data/splits/train.csv only
(exactly how the repository's own fallback predictor does it) when explicitly requested; XLM-R is loaded from an
existing local checkpoint. Anything unavailable is reported as NOT RUN / UNVERIFIED with the exact reason."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lmp.evaluation.metrics import full_metrics

from .ingest import HARMONIZED
from .stats import subset_masks

CANONICAL = {"lexical_similarity": "TF-IDF cosine similarity, linearly calibrated on train", "rule_features_ridge": "rule-based features + ridge (train only)",
             "sentence_transformer": "multilingual Sentence-Transformer cosine, calibrated on train", "lexrule_gbm": "lexical+rule features gradient boosting (train only)",
             "xlmr_crossencoder": "XLM-R cross-encoder checkpoint"}
SK_KIND = {"lexical_similarity": "tfidf_cal", "rule_features_ridge": "rule_ridge", "lexrule_gbm": "lexrule_gbm"}
SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
COLS = ["model_name", "status", "reason", "checkpoint_id", "identity_status", "training_provenance", "training_splits", "target", "subset", "n", "mae", "rmse",
        "spearman", "pearson", "bias", "accuracy", "f1_macro", "f1_weighted", "confusion_rows_truth_cols_pred", "heldout_status", "heldout_note"]
TEST_NOTE = "the fixed test split was already used in exploratory model evaluation: not an untouched final test set"


@dataclass
class ModelPred:
    name: str
    scores: pd.Series                      # index = neutral pair_id
    checkpoint_id: str | None = None
    training_splits: tuple | None = None
    provenance: str = "UNVERIFIED"         # VERIFIED_IN_PROCESS | DECLARED | UNVERIFIED
    identity: str = "UNVERIFIED"
    notes: str = ""


def _norm_splits(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
        return None
    parts = [p.strip().lower() for p in str(x).replace(",", ";").split(";") if p.strip()]
    return tuple(sorted({SPLIT_ALIASES.get(p, p) for p in parts}))


def training_groups(data_root, splits) -> set:
    out = set()
    for s in splits or ():
        p = Path(data_root) / "data" / "splits" / f"{'val' if s == 'validation' else s}.csv"
        if p.exists():
            d = pd.read_csv(p, keep_default_na=False)
            for c in ("group_id", "concept_id"):
                if c in d:
                    out |= {x for x in d[c].astype(str) if x}
    return out


def import_predictions(path, cons: pd.DataFrame):
    """Read a predictions CSV (pair_id, model_name, score[, checkpoint_id, training_splits]); returns (ModelPred list, issues)."""
    issues, preds = [], []
    df = pd.read_csv(path)
    need = {"pair_id", "model_name", "score"}
    if not need <= set(df.columns):
        return [], [f"{path}: missing required columns {sorted(need - set(df.columns))}"]
    src_map = dict(zip(cons.source_pair_id, cons.pair_id))
    df["pair_id"] = df["pair_id"].astype(str).map(lambda p: p if p in set(cons.pair_id) else src_map.get(p, p))
    for name, g in df.groupby("model_name"):
        bad_ids = sorted(set(g.pair_id) - set(cons.pair_id))
        if bad_ids:
            issues.append(f"{name}: {len(bad_ids)} pair IDs not among the annotated pairs (e.g. {bad_ids[:3]}); ignored")
            g = g[~g.pair_id.isin(bad_ids)]
        if g.pair_id.duplicated().any():
            issues.append(f"{name}: duplicate pair IDs; model skipped")
            continue
        sc = pd.to_numeric(g.score, errors="coerce")
        if sc.isna().any() or ((sc < 1) | (sc > 10)).any():
            issues.append(f"{name}: non-numeric or out-of-range (1-10) scores; model skipped")
            continue
        cov = len(g) / len(cons)
        if cov < 1:
            issues.append(f"{name}: covers {len(g)}/{len(cons)} annotated pairs; metrics use the covered subset")
        ck = g.get("checkpoint_id", pd.Series(dtype=object)).dropna().astype(str).unique()
        ts = _norm_splits(g["training_splits"].dropna().iloc[0]) if "training_splits" in g and g["training_splits"].notna().any() else None
        if "training_splits" in g and g["training_splits"].dropna().astype(str).nunique() > 1:
            issues.append(f"{name}: inconsistent training_splits values; provenance treated as UNVERIFIED")
            ts = None
        preds.append(ModelPred(name, pd.Series(sc.to_numpy(float), index=g.pair_id.to_numpy()), ck[0] if len(ck) == 1 else None, ts,
                               "DECLARED" if ts else "UNVERIFIED", "DECLARED" if len(ck) == 1 else "UNVERIFIED", f"imported from {Path(path).name}"))
        if ts is None:
            issues.append(f"{name}: training_splits missing -> training provenance UNVERIFIED (no held-out claims)")
        if len(ck) != 1:
            issues.append(f"{name}: checkpoint_id missing/ambiguous -> checkpoint identity UNVERIFIED")
    return preds, issues


def run_live(requested, cons, data_root, xlmr_checkpoint=None, xlmr_provenance=None):
    """Evaluate requested models in-process. Returns (ModelPred list, {model: NOT RUN reason})."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    preds, not_run = [], {}
    root = Path(data_root)
    df = cons[["pair_id", "original", "simplified", "language"]].copy()
    train_path = root / "data" / "splits" / "train.csv"
    train = pd.read_csv(train_path, keep_default_na=False) if train_path.exists() else None
    if train is not None:
        train["lmp_score"] = pd.to_numeric(train["lmp_score"])
    for name in requested:
        try:
            if name in SK_KIND:
                if train is None:
                    not_run[name] = f"NOT RUN: {train_path} not found"
                    continue
                from lmp.models.sk_models import SKScorer
                s = SKScorer(SK_KIND[name]).fit(train).predict(df)
                preds.append(ModelPred(name, pd.Series(np.asarray(s, float), index=df.pair_id), f"in-process fit of {SK_KIND[name]}", ("train",), "VERIFIED_IN_PROCESS", "VERIFIED_IN_PROCESS", "fitted in memory on data/splits/train.csv only"))
            elif name == "sentence_transformer":
                from lmp.config import load_config
                from lmp.evaluation.metrics import calibrate_linear
                from lmp.models.st_baseline import SentenceTransformerBaseline
                cfg = load_config()
                if train is None:
                    not_run[name] = f"NOT RUN: {train_path} not found"
                    continue
                st = SentenceTransformerBaseline(cfg.get("sentence_transformer_local_path") or cfg["sentence_transformer_model"], device="cpu")
                cal = calibrate_linear(st.predict(train.original.tolist(), train.simplified.tolist()), train.lmp_score.values)
                preds.append(ModelPred(name, pd.Series(cal(st.predict(df.original.tolist(), df.simplified.tolist())), index=df.pair_id), "sentence-transformer + linear calibration on train", ("train",), "VERIFIED_IN_PROCESS", "UNVERIFIED", "model weights loaded from local cache only (offline mode)"))
            elif name == "xlmr_crossencoder":
                if not xlmr_checkpoint or not Path(xlmr_checkpoint).exists():
                    not_run[name] = "NOT RUN: no --xlmr-checkpoint directory supplied/found"
                    continue
                preds.append(_run_xlmr(Path(xlmr_checkpoint), df, xlmr_provenance))
            else:
                not_run[name] = f"NOT RUN: unknown model name {name!r}"
        except Exception as e:  # noqa: BLE001
            not_run[name] = f"NOT RUN: {type(e).__name__}: {str(e)[:200]}"
    return preds, not_run


def _run_xlmr(ck: Path, df, prov_path):
    meta = json.loads((ck / "meta.json").read_text(encoding="utf-8")) if (ck / "meta.json").exists() else {}
    weights = ck / "model.pt"
    cid = f"{ck.name}:{hashlib.sha256(weights.read_bytes()).hexdigest()[:12]}" if weights.exists() else None
    if meta.get("architecture") == "LMPCrossEncoder":
        from lmp.models.train_crossenc import load_checkpoint, predict
        model, tok, m = load_checkpoint(ck)
        scores = predict(model, tok, df[["original", "simplified", "language"]], m["max_length"], 4)[0]
    else:
        from lmp.models import trainer
        model, tok, m = trainer.load_checkpoint(ck)
        scores = trainer.predict(model, tok, df.original.tolist(), df.simplified.tolist(), max_length=m["max_length"], batch_size=4, device="cpu")
    ts, prov = None, "UNVERIFIED"
    if prov_path and Path(prov_path).exists():
        pj = json.loads(Path(prov_path).read_text(encoding="utf-8"))
        ts, prov = _norm_splits(";".join(pj.get("training_splits", []))), "DECLARED"
        cid = pj.get("checkpoint_id") or cid
    note = "checkpoint meta.json carries no training-split provenance" if prov == "UNVERIFIED" else f"training splits declared in {Path(prov_path).name}"
    if meta.get("pretrained_weights_used") is False:
        note += "; randomly initialised (NOT pretrained) - diagnostic only"
    return ModelPred("xlmr_crossencoder", pd.Series(np.asarray(scores, float), index=df.pair_id), cid, ts, prov, "VERIFIED_IN_PROCESS" if cid else "UNVERIFIED", note)


def _heldout(mp: ModelPred, sub: pd.DataFrame, data_root):
    if mp.training_splits is None:
        return "UNVERIFIED", "training splits unknown: no held-out claim possible"
    trained = sub.dataset_split.isin(mp.training_splits)
    groups = training_groups(data_root, mp.training_splits)
    leak = ~trained & (sub.group_id.astype(str).isin(groups) | sub.concept_id.astype(str).isin(groups))
    if trained.any():
        return f"CONTAINS_TRAINING_DATA ({int(trained.sum())}/{len(sub)} pairs in {','.join(mp.training_splits)})", "not independent generalisation"
    if leak.any():
        return f"GROUP_LEAKAGE ({int(leak.sum())}/{len(sub)} pairs share a group with training data)", "not held-out"
    tag = "HELD_OUT_VERIFIED_SPLIT_AND_GROUP_DISJOINT" if mp.provenance == "VERIFIED_IN_PROCESS" else "HELD_OUT_DECLARED_SPLIT_AND_GROUP_DISJOINT (training provenance declared, not independently verified)"
    return tag, TEST_NOTE if (sub.dataset_split == "test").all() else ""


def evaluate_models(preds, not_run, cons, data_root, thresholds=HARMONIZED) -> pd.DataFrame:
    rows = []
    for name, why in not_run.items():
        rows.append({"model_name": name, "status": "NOT RUN", "reason": why})
    for mp in preds:
        c = cons[cons.pair_id.isin(mp.scores.index)].copy()
        c["pred"] = mp.scores.reindex(c.pair_id).to_numpy()
        for target, col, base in (("student_mean", "student_mean", c), ("expert", "expert_score", c[c.expert_reviewed])):
            for sname, mask in subset_masks(base, ("all", "fixed_test", "supplemental_train_val", "english", "hindi", "english_fixed_test", "hindi_fixed_test")).items():
                sub = base[mask]
                if len(sub) < 3:
                    continue
                m = full_metrics(sub[col].to_numpy(float), sub.pred.to_numpy(float), thresholds)
                ho, note = _heldout(mp, sub, data_root)
                rows.append({"model_name": mp.name, "status": "EVALUATED", "reason": mp.notes, "checkpoint_id": mp.checkpoint_id, "identity_status": mp.identity,
                             "training_provenance": mp.provenance, "training_splits": ";".join(mp.training_splits) if mp.training_splits else None, "target": target,
                             "subset": sname if target == "student_mean" else f"expert_{sname}", "n": m["n"], **{k: m[k] for k in ("mae", "rmse", "spearman", "pearson", "bias", "accuracy", "f1_macro", "f1_weighted")},
                             "confusion_rows_truth_cols_pred": json.dumps(m["confusion"]), "heldout_status": ho, "heldout_note": note})
    out = pd.DataFrame(rows)
    for c in COLS:
        if c not in out:
            out[c] = None
    return out[COLS]


def not_requested(reason="model evaluation not requested (run with --run-models / --predictions)"):
    return {k: f"NOT RUN: {reason}" for k in CANONICAL}
