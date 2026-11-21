"""Experiment framework (v2). Everything reported is computed here from actual runs; nothing is typed in by hand.
E1  repeated grouped k-fold CV of all runnable models on the chosen dataset (out-of-fold predictions; group-cluster bootstrap CIs)
E2  per-language and per-domain breakdowns of E1
E3  DATASET ABLATION on a FIXED grouped test set: train on original-48 / expanded / filtered / reviewed, identical test rows
E4  AUGMENTATION ablation (off vs on), train-only, same fixed test
E5  sanity checks (identical pair, shuffled/unrelated pair) per model
E6  explainer label-detection and false-alarm analysis
E7  error analysis tables from out-of-fold predictions
Models that cannot run here are listed in meta.model_status with the exact reason (never silently dropped, never given placeholder numbers).
"""
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from ..config import set_seed, score_to_category
from ..data.io import load_pairs
from ..data.splits import repeated_grouped_kfold, grouped_split, merge_near_duplicate_groups, assert_no_leakage
from ..data.augment import augment_training_set, assert_augmentation_isolated
from ..explain.analyzer import analyze
from ..models.sk_models import SKScorer, SK_KINDS
from .metrics import full_metrics, label_detection_metrics, calibrate_linear

NOT_RUN = {
    "sentence_transformer_cosine": "NOT RUN: Hugging Face Hub unreachable from the build sandbox (HTTP 000); run `python scripts/run_experiments.py --st-path <local model dir or cached model name>` where the weights are available",
    "xlmr_cross_encoder_pretrained": "NOT RUN: pretrained xlm-roberta-base weights not obtainable here; train with scripts/train_crossenc.py on a machine with Hub access",
}


def _boot_ci(y, p, groups, fn, n_boot=300, seed=0):
    """Cluster (group) bootstrap 95% CI of fn(y_true, y_pred)."""
    rng = np.random.RandomState(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try: v = fn(y[ii], p[ii])
        except Exception: v = np.nan
        if v == v: vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))

def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0] if np.std(b) > 0 else np.nan

ST_SIM: dict = {}      # pair_id -> cosine similarity from an optional Sentence-Transformer (filled by register_sentence_transformer)

def register_sentence_transformer(df, model_path: str) -> str:
    """Load a LOCAL/cached sentence-transformer and cache cosine similarities for every pair. Returns a status string; never fakes success."""
    try:
        from ..models.st_baseline import SentenceTransformerBaseline
        st = SentenceTransformerBaseline(model_path)
        sims = st.similarity(list(df.original), list(df.simplified))
    except Exception as e:                                  # missing package, missing weights, no network, ...
        return f"NOT RUN: could not load sentence-transformer from {model_path!r} ({type(e).__name__}: {str(e)[:100]})"
    ST_SIM.update(dict(zip(df.pair_id, map(float, sims)))); return f"run (model={model_path}; similarity -> sign-free linear calibration on training folds)"

def _fit_predict(kind, tr, te):
    if kind == "st_cal":
        f = lambda d: 1 + 9 * np.clip(np.array([ST_SIM[i] for i in d.pair_id]), 0, 1)
        return calibrate_linear(f(tr), tr.lmp_score.values, allow_negative=True)(f(te))
    if kind == "tfidf_cal_signfree":
        m = SKScorer("tfidf_raw").fit(tr); raw = m.predict(tr); cal = calibrate_linear(raw, tr.lmp_score.values, allow_negative=True); return cal(m.predict(te))
    return SKScorer(kind).fit(tr).predict(te)

MODELS = ["tfidf_raw", "tfidf_cal", "tfidf_cal_signfree", "lex_ridge", "rule_ridge", "lexrule_ridge", "lexrule_gbm"]

def run_cv(df, cfg, models=MODELS):
    k, reps, seed = cfg["cv"]["k"], cfg["cv"]["repeats"], cfg["seed"]; df = df.reset_index(drop=True)
    oof = {m: [np.full(len(df), np.nan) for _ in range(reps)] for m in models}
    for r, f, a, b in repeated_grouped_kfold(df, k, reps, seed):
        tr, te = df.iloc[a], df.iloc[b]
        assert not (set(tr.group_id) & set(te.group_id)), "group leakage inside CV fold"
        for m in models: oof[m][r][b] = _fit_predict(m, tr, te)
    return oof

def summarize_cv(df, oof, thr, models):
    sg = merge_near_duplicate_groups(df.reset_index(drop=True)).values; y = df.lmp_score.values; res = {}
    for m in models:
        per_rep = [full_metrics(y, p, thr) for p in oof[m]]
        keys = ["mae", "rmse", "pearson", "spearman", "accuracy", "f1_macro", "f1_weighted", "collapse_index", "bias"]
        pm = np.mean(np.vstack(oof[m]), axis=0)
        res[m] = {k_: {"mean": float(np.nanmean([r[k_] for r in per_rep])), "sd_across_repeats": float(np.nanstd([r[k_] for r in per_rep]))} for k_ in keys}
        res[m]["spearman_ci95_cluster_bootstrap"] = _boot_ci(y, oof[m][0], sg, _spearman); res[m]["rmse_ci95_cluster_bootstrap"] = _boot_ci(y, oof[m][0], sg, lambda a, b: np.sqrt(np.mean((a - b) ** 2)))
        res[m]["per_class_rep0"], res[m]["confusion_rep0"] = per_rep[0]["per_class"], per_rep[0]["confusion"]
        res[m]["by_language"] = {l: {k_: float(full_metrics(y[(df.language == l).values], oof[m][0][(df.language == l).values], thr)[k_]) for k_ in ("spearman", "rmse", "mae", "f1_macro", "accuracy", "collapse_index")} for l in ("en", "hi")}
        res[m]["by_domain_spearman"] = {d: float(_spearman(y[(df.legal_domain == d).values], oof[m][0][(df.legal_domain == d).values])) for d in sorted(df.legal_domain.unique()) if (df.legal_domain == d).sum() >= 12}
    return res

def run_dataset_ablation(data_dir: Path, cfg, models=("tfidf_cal_signfree", "lex_ridge", "lexrule_ridge", "lexrule_gbm"), n_rep=5):
    thr = cfg["category_thresholds"]; full = load_pairs(data_dir / "pairs_consolidated.csv"); variants = {n: load_pairs(data_dir / f) for n, f in
        {"original_only(48)": "pairs_original_only.csv", "expanded": "pairs_consolidated.csv", "expanded_filtered": "pairs_filtered.csv", "reviewed_subset": "pairs_reviewed.csv"}.items()}
    out = {n: {m: [] for m in models} for n in list(variants) + ["expanded+augmentation"]}; sizes = {}
    for r in range(n_rep):
        _, _, te = grouped_split(full.assign(), val_frac=0.05, test_frac=0.25, seed=cfg["seed"] + r)
        test_groups = set(te.split_group); ev = full[merge_near_duplicate_groups(full).isin(test_groups).values]
        for name, v in variants.items():
            trn = v[~merge_near_duplicate_groups(v).isin(test_groups).values]; sizes.setdefault(name, []).append(len(trn))
            for m in models: out[name][m].append(full_metrics(ev.lmp_score, _fit_predict(m, trn, ev), thr))
        base = variants["expanded"]; trn = base[~merge_near_duplicate_groups(base).isin(test_groups).values].assign(split_group=lambda d: merge_near_duplicate_groups(d).values)
        aug = augment_training_set(trn, cfg["seed"] + r, cfg["augment"]["frac"], tuple(cfg["augment"]["strategies"])); assert_augmentation_isolated(aug, ev.assign(is_augmented=False))
        sizes.setdefault("expanded+augmentation", []).append(len(aug))
        for m in models: out["expanded+augmentation"][m].append(full_metrics(ev.lmp_score, _fit_predict(m, aug, ev), thr))
    agg = {n: {m: {k: {"mean": float(np.nanmean([x[k] for x in runs])), "sd": float(np.nanstd([x[k] for x in runs]))} for k in ("spearman", "rmse", "f1_macro", "accuracy", "collapse_index")}
               for m, runs in d.items()} for n, d in out.items()}
    return {"n_repeats": n_rep, "mean_train_sizes": {k: float(np.mean(v)) for k, v in sizes.items()}, "results": agg, "test_note": "fixed grouped test set per repeat, identical for every variant"}

def sanity_checks(df, cfg, models=MODELS):
    tr, va, te = grouped_split(df, seed=cfg["seed"]); res = {"n_test": len(te)}; rng = np.random.RandomState(0)
    ident = te.assign(simplified=te.original); shuf = te.assign(simplified=te.simplified.sample(frac=1, random_state=0).values)
    shuf = shuf[shuf.original != shuf.simplified]
    for m in models:
        f = (lambda d, m=m: _fit_predict(m, tr, d))
        res[m] = {"identical_mean": float(f(ident).mean()), "shuffled_unrelated_mean": float(f(shuf).mean()), "test_mean": float(f(te).mean())}
    return res

def explainer_eval(df):
    d = df[df.language.isin(["en", "hi"])].copy(); gold = [set(x for x in str(s).split("|") if x) for s in d.error_labels]
    an = [analyze(o, s, l, min_confidence=0.0) for o, s, l in zip(d.original, d.simplified, d.language)]
    out = {"n": len(d), "label_detection_all_findings": label_detection_metrics(gold, [a.labels for a in an])}
    for thr in (0.0, 0.6, 0.75):
        flag = np.array([any(f.confidence >= thr for f in a.findings) for a in an]); d["flag"] = flag
        out[f"flag_rate_by_category_conf>={thr}"] = d.groupby("category").flag.mean().round(3).to_dict()
        out[f"flag_rate_by_language_category_conf>={thr}"] = {f"{l}/{c}": float(v) for (l, c), v in d.groupby(["language", "category"]).flag.mean().items()}
        out[f"label_micro_f1_conf>={thr}"] = label_detection_metrics(gold, [{f.category for f in a.findings if f.confidence >= thr} for a in an])["micro"]["f1"]
    return out

def error_analysis(df, oof_pred, name, out_md: Path, thr, k=6):
    d = df.reset_index(drop=True).assign(pred=oof_pred, err=lambda x: x.pred - x.lmp_score); d["abs"] = d.err.abs()
    from ..preprocess import tokenize
    d["jac"] = [len(set(tokenize(a)) & set(tokenize(b))) / max(1, len(set(tokenize(a)) | set(tokenize(b)))) for a, b in zip(d.original, d.simplified)]
    L = [f"# Error analysis - {name} (out-of-fold predictions, repeat 0, grouped CV)", "", "All rows are real dataset rows (AI-generated or original synthetic); predictions are genuine out-of-fold outputs. Labels are AI-assisted, not human.", ""]
    def block(title, sub):
        L.extend([f"## {title}  (n={len(sub)})", ""])
        if not len(sub): L.extend(["_No rows in this category._", ""]); return
        L.append(f"mean predicted {sub.pred.mean():.2f} vs mean reference {sub.lmp_score.mean():.2f}; MAE {sub['abs'].mean():.2f}"); L.append("")
        for r in sub.sort_values("abs", ascending=False).head(3).itertuples():
            L.extend([f"- `{r.pair_id}` ({r.language}, {r.legal_domain}) ref {r.lmp_score:g} / pred {r.pred:.1f}", f"  - O: {r.original}", f"  - S: {r.simplified}"])
        L.append("")
    for lab, title in [("changed_obligation", "Obligation changed"), ("changed_permission", "Permission/obligation swapped"), ("contradiction", "Contradiction / negation / reversal"),
                       ("missing_exception", "Exception removed"), ("altered_condition", "Condition altered"), ("altered_entities", "Actor / party changed"), ("scope_change", "Scope / quantifier changed"),
                       ("altered_dates", "Time limit changed"), ("altered_numbers", "Numerical change"), ("altered_reference", "Reference changed")]:
        block(title, d[d.error_labels.fillna("").str.contains(lab)])
    block("HIGH lexical similarity (Jaccard>=0.8) but NOT preserved", d[(d.jac >= 0.8) & (d.lmp_score < 4)])
    block("LOW lexical similarity (Jaccard<0.5) but Preserved", d[(d.jac < 0.5) & (d.lmp_score >= 7.5)])
    for l in ("hi", "en"): block(f"Largest errors, language={l}", d[d.language == l])
    out_md.write_text("\n".join(L), encoding="utf-8"); return d
