"""The five experiments from the proposal. Everything reported is computed at run time; a model that cannot be run in the
current environment is reported as NOT RUN with the reason, never with placeholder numbers."""
from __future__ import annotations
import json, platform, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd
from .metrics import regression_metrics, category_metrics, label_detection_metrics, calibrate_linear
from ..config import set_seed, score_to_category
from ..data.io import load_pairs
from ..data.augment import augment_with_sanity_pairs
from ..data.splits import document_split, domain_holdout_split
from ..explain.analyzer import analyze
from ..logging_utils import get_logger
from ..models.tfidf import TfidfBaseline
from ..models.feature_regressor import FeatureRegressor
from .. import viz

log = get_logger("lmp.exp")

class ModelZoo:
    """Builds all scorers for one train/val split. Each scorer: name -> callable(df) -> np.ndarray, or a NOT RUN reason."""
    def __init__(self, cfg, train, val, use_transformers=True, ckpt_dir=None):
        self.cfg, self.status, self.scorers, self.history = cfg, {}, {}, None
        if cfg.get("augment_sanity_pairs", True):
            train = augment_with_sanity_pairs(train, cfg["seed"])
        texts = list(train.original) + list(train.simplified)
        self.tfidf = TfidfBaseline().fit(texts)
        raw_tr = self.tfidf.predict(list(train.original), list(train.simplified))
        cal = calibrate_linear(raw_tr, train.lmp_score.values) if train.lmp_score.nunique() > 1 else (lambda x: x)
        self.scorers["TF-IDF cosine (raw)"] = lambda d: self.tfidf.predict(list(d.original), list(d.simplified))
        self.scorers["TF-IDF cosine (calibrated)"] = lambda d: cal(self.tfidf.predict(list(d.original), list(d.simplified)))
        fr = FeatureRegressor(self.tfidf).fit(train)
        self.scorers["Feature-ridge (offline fallback)"] = fr.predict_df
        self.status["TF-IDF cosine (raw)"] = self.status["TF-IDF cosine (calibrated)"] = self.status["Feature-ridge (offline fallback)"] = "run"
        if use_transformers:
            self._try_st(cfg, train)
            self._try_regressor(cfg, train, val, ckpt_dir)
        else:
            for n in ("Sentence-Transformer cosine (raw)", "JUDGEBERT-inspired regressor"): self.status[n] = "NOT RUN: transformers disabled (--no-transformers)"

    def _try_st(self, cfg, train):
        name = "Sentence-Transformer cosine (raw)"
        try:
            from ..models.st_baseline import SentenceTransformerBaseline
            st = SentenceTransformerBaseline(cfg["sentence_transformer_model"])
            self.scorers[name] = lambda d: st.predict(list(d.original), list(d.simplified)); self.status[name] = "run"
            raw = st.predict(list(train.original), list(train.simplified))
            cal = calibrate_linear(raw, train.lmp_score.values)
            self.scorers["Sentence-Transformer cosine (calibrated)"] = lambda d: cal(st.predict(list(d.original), list(d.simplified)))
            self.status["Sentence-Transformer cosine (calibrated)"] = "run"
        except Exception as e:
            self.status[name] = f"NOT RUN: {type(e).__name__}: {str(e)[:160]}"

    def _try_regressor(self, cfg, train, val, ckpt_dir):
        name = "JUDGEBERT-inspired regressor"
        try:
            from transformers import AutoTokenizer
            from ..models import trainer
            tok = AutoTokenizer.from_pretrained(cfg["regressor"]["backbone"])
            model, hist = trainer.fit(train, val, cfg, cfg["seed"], tok, ckpt_dir=ckpt_dir)  # `train` is already augmented
            self.history = hist
            mx = cfg["regressor"]["max_length"]
            self.scorers[name] = lambda d: trainer.predict(model, tok, list(d.original), list(d.simplified), mx)
            self.status[name] = "run"
        except Exception as e:
            self.status[name] = f"NOT RUN: {type(e).__name__}: {str(e)[:160]}"

def _eval(zoo, df, thr):
    res = {}
    for name, fn in zoo.scorers.items():
        p = fn(df)
        res[name] = {**regression_metrics(df.lmp_score, p), **category_metrics(df.lmp_score, p, thr)}
    return res

def run_all(cfg, out_dir: Path, use_transformers=True, data_path=None):
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "figures").mkdir(exist_ok=True)
    seed, thr = cfg["seed"], cfg["category_thresholds"]; set_seed(seed)
    df = load_pairs(data_path or cfg["data_path"])
    synthetic_only = bool((df.data_origin == "synthetic").all())
    train, val, test = document_split(df, seed=seed)
    log.info("split by document: train=%d val=%d test=%d pairs", len(train), len(val), len(test))
    zoo = ModelZoo(cfg, train, val, use_transformers, ckpt_dir=Path(cfg["checkpoint_dir"]) / "judgebert_regressor")
    R = {"meta": {"seed": seed, "n_pairs": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test), "synthetic_only": synthetic_only,
                  "python": sys.version.split()[0], "platform": platform.platform(), "config": cfg, "model_status": zoo.status}}
    # Exp 1 + 2
    R["exp1_test"] = _eval(zoo, test, thr)
    R["exp1_all_pairs_unsupervised_view"] = _eval(zoo, df, thr)
    R["exp2_language_test"] = {l: _eval(zoo, test[test.language == l], thr) for l in ("en", "hi") if (test.language == l).sum() >= 3}
    R["exp2_language_all_pairs"] = {l: _eval(zoo, df[df.language == l], thr) for l in ("en", "hi")}
    # Exp 3: leave-one-domain-out
    exp3 = {}
    for dom in sorted(df.legal_domain.unique()):
        tr, va, te = domain_holdout_split(df, dom, seed=seed)
        z = ModelZoo(cfg, tr, va, use_transformers=False)
        exp3[dom] = {"n_train": len(tr), "n_test": len(te), "results": _eval(z, te, thr)}
    R["exp3_domain_holdout"] = exp3
    R["exp3_note"] = "Transformer regressor is not retrained per fold in this script (cost); run scripts/train_regressor.py --holdout-domain <d> for it."
    # Exp 4: sanity checks
    heldout = pd.concat([val, test], ignore_index=True)   # documents never used for training or augmentation
    R["exp4_sanity"] = sanity_checks(zoo, heldout, seed)
    # explainer evaluation
    gold = [set(x for x in s.split("|") if x) for s in df.error_labels]; pred = [analyze(o, s, l).labels for o, s, l in zip(df.original, df.simplified, df.language)]
    R["explainer_label_detection_all_pairs"] = label_detection_metrics(gold, pred)
    # Exp 5
    exp5_md = qualitative(zoo, df)
    (out_dir / "exp5_error_analysis.md").write_text(exp5_md, encoding="utf-8")
    # figures
    sc = {n: f(test) for n, f in zoo.scorers.items()}
    viz.scatter_models(sc, test.lmp_score.values, out_dir / "figures" / "exp1_scatter_test.png")
    viz.bar_metrics({n: v for n, v in R["exp1_test"].items()}, "rmse", out_dir / "figures" / "exp1_rmse_test.png", "RMSE on test split (lower is better)")
    viz.bar_metrics({n: v for n, v in R["exp1_all_pairs_unsupervised_view"].items()}, "spearman", out_dir / "figures" / "exp1_spearman_all.png", "Spearman, all pairs (synthetic)")
    viz.grouped_bar({n: {l: R["exp2_language_all_pairs"][l][n]["spearman"] for l in R["exp2_language_all_pairs"]} for n in zoo.scorers}, "Spearman", out_dir / "figures" / "exp2_language_spearman.png", "Per-language Spearman (all pairs)")
    if zoo.history: viz.loss_curves(zoo.history, out_dir / "figures" / "training_loss.png")
    (out_dir / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    write_tables(R, out_dir)
    return R

def sanity_checks(zoo, df, seed):
    rng = np.random.RandomState(seed)
    ident = df.assign(simplified=df.original)
    # unrelated: pair each original with a simplified sentence of a DIFFERENT language-matched domain
    shuf = []
    for _, r in df.iterrows():
        other = df[(df.language == r.language) & (df.legal_domain != r.legal_domain)]
        shuf.append(other.iloc[rng.randint(len(other))].simplified)
    unrel = df.assign(simplified=shuf)
    # subtle: minimal edits that SHOULD change legal meaning
    subtle_rows = []
    for _, r in df[df.language == "en"].iterrows():
        s = r.original
        for a, b in (("shall not", "may"), ("shall", "may"), ("must", "may"), ("within", "after")):
            if a in s: subtle_rows.append(r.copy().to_dict() | {"simplified": s.replace(a, b, 1)}); break
    for _, r in df[df.language == "hi"].iterrows():
        s = r.original
        for a, b in (("होगा", "सकता है"), ("होगी", "सकती है"), ("नहीं करेगा", "कर सकता है")):
            if a in s: subtle_rows.append(r.copy().to_dict() | {"simplified": s.replace(a, b, 1)}); break
    subtle = pd.DataFrame(subtle_rows)
    out = {"n_identical": len(ident), "n_unrelated": len(unrel), "n_subtle": len(subtle), "models": {}}
    for n, f in zoo.scorers.items():
        out["models"][n] = {"identical_mean": float(f(ident).mean()), "unrelated_mean": float(f(unrel).mean()), "subtle_mean": float(f(subtle).mean()) if len(subtle) else None}
    out["explainer_flags_subtle"] = float(np.mean([bool(analyze(o, s, l).findings) for o, s, l in zip(subtle.original, subtle.simplified, subtle.language)])) if len(subtle) else None
    out["explainer_flags_identical"] = float(np.mean([bool(analyze(o, s, l).findings) for o, s, l in zip(ident.original, ident.simplified, ident.language)]))
    return out

CHALLENGE = ["changed_obligation", "contradiction", "changed_permission", "altered_condition", "missing_exception", "altered_numbers", "altered_dates", "scope_change"]
def qualitative(zoo, df, n_per_label=2):
    picked, seen = [], set()
    for lab in CHALLENGE:
        for _, r in df[df.error_labels.str.contains(lab)].iterrows():
            if r.pair_id not in seen and sum(lab in p for p in [x.error_labels for x in picked]) < n_per_label:
                picked.append(r); seen.add(r.pair_id)
    # add hard negatives: paraphrases scored high by author (does the analyzer over-flag?)
    picked += [r for _, r in df[df.lmp_score >= 9].iterrows() if r.pair_id not in seen][:4]
    lines = ["# Experiment 5 - Qualitative error analysis", "",
             f"{len(picked)} pairs. **All pairs are SYNTHETIC** (author-written); 'Reference' is the author-assigned score, not a human annotation.",
             "Scores are produced by the models that could be run in this environment (see results.json -> meta.model_status).", ""]
    for r in picked:
        a = analyze(r.original, r.simplified, r.language)
        preds = {n: float(f(pd.DataFrame([r]))[0]) for n, f in zoo.scorers.items()}
        lines += [f"## {r.pair_id} ({r.language}, {r.legal_domain})", f"- Original: {r.original}", f"- Simplified: {r.simplified}",
                  f"- Reference LMP (author): {r.lmp_score:g} - gold labels: `{r.error_labels or 'none'}`",
                  "- Model scores: " + "; ".join(f"{n}: {v:.2f}" for n, v in preds.items()),
                  f"- Explainer labels: `{'|'.join(sorted(a.labels)) or 'none'}`", ""]
        for f in a.findings:
            lines.append(f"  - **{f.category}** (conf {f.confidence:.2f}, rule `{f.rule}`): detected: {f.detected_change} Interpretation: {f.interpretation}")
        match = a.labels == set(x for x in r.error_labels.split("|") if x)
        lines += ["", f"- Explainer vs gold labels: {'exact match' if match else 'DIFFERS - inspect (false positive / false negative / label ambiguity)'}", ""]
    return "\n".join(lines)

def _fmt(v): return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
def write_tables(R, out_dir: Path):
    L = ["# Results (generated by scripts/run_experiments.py)", "",
         "> **READ FIRST.** These numbers come from a 48-pair SYNTHETIC dataset whose scores were assigned by the project author. "
         "They verify that the pipeline runs end to end. They are NOT evidence about real-world model quality and must not be reported as research findings.", "",
         f"Pairs: {R['meta']['n_pairs']} (train {R['meta']['n_train']} / val {R['meta']['n_val']} / test {R['meta']['n_test']}), seed {R['meta']['seed']}", "",
         "## Model status", ""] + [f"- **{k}**: {v}" for k, v in R["meta"]["model_status"].items()]
    def tbl(title, d):
        cols = ["n", "pearson", "spearman", "rmse", "mae", "precision_macro", "recall_macro", "f1_macro", "kappa"]
        out = ["", f"## {title}", "", "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for m, v in d.items(): out.append(f"| {m} | " + " | ".join(str(v["n"]) if c == "n" else _fmt(v[c]) for c in cols) + " |")
        return out
    L += tbl("Exp 1 - baseline comparison (test split; tiny n)", R["exp1_test"])
    L += tbl("Exp 1b - all pairs (descriptive; unsupervised baselines never saw labels, trained models did)", R["exp1_all_pairs_unsupervised_view"])
    for l, d in R["exp2_language_all_pairs"].items(): L += tbl(f"Exp 2 - language = {l} (all pairs, descriptive)", d)
    for dom, d in R["exp3_domain_holdout"].items(): L += tbl(f"Exp 3 - held-out domain = {dom} (train n={d['n_train']}, test n={d['n_test']})", d["results"])
    s = R["exp4_sanity"]
    L += ["", "## Exp 4 - sanity checks on held-out documents (mean predicted LMP; identical should be ~10, unrelated ~1)", "", "| Model | identical | unrelated | subtle legal edit |", "|---|---|---|---|"]
    for m, v in s["models"].items(): L.append(f"| {m} | {_fmt(v['identical_mean'])} | {_fmt(v['unrelated_mean'])} | {_fmt(v['subtle_mean'])} |")
    L += ["", f"Explainer flags >=1 difference in {s['explainer_flags_subtle']:.0%} of the {s['n_subtle']} subtle edits and in {s['explainer_flags_identical']:.0%} of identical pairs.", "",
          "## Explainer: legal-error label detection vs author labels (all 48 pairs)", "", "| Label | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for lab, v in R["explainer_label_detection_all_pairs"].items(): L.append(f"| {lab} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |")
    L += ["", "See `exp5_error_analysis.md` for Experiment 5 and `figures/` for plots."]
    (out_dir / "results.md").write_text("\n".join(L), encoding="utf-8")
