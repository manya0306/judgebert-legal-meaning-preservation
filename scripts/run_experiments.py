"""Run all feasible experiments (from project root):  python scripts/run_experiments.py [--data data/expanded] [--out outputs] [--quick]
Legacy 48-pair experiments: python scripts/run_experiments.py --legacy"""
import argparse, json, sys, time, platform
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lmp.config import load_config, set_seed
from lmp.data.io import load_pairs
from lmp.evaluation import framework as F

def _fmt(x): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/expanded"); ap.add_argument("--out", default="outputs"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--legacy", action="store_true"); ap.add_argument("--st-path", help="local dir or cached name of a sentence-transformers model (optional baseline)"); a = ap.parse_args()
    if a.legacy:
        from lmp.evaluation.experiments import run_all
        cfg = load_config(); cfg["data_path"] = "data/synthetic/synthetic_pairs.csv"; run_all(cfg, ROOT / a.out / "legacy_48", use_transformers=False); return
    cfg = load_config(); set_seed(cfg["seed"]); thr = cfg["category_thresholds"]; out = ROOT / a.out; (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.quick: cfg["cv"] = {"k": 3, "repeats": 1}
    data = ROOT / a.data; df = load_pairs(data / "pairs_consolidated.csv").reset_index(drop=True); t0 = time.time()
    R = {"meta": {"seed": cfg["seed"], "n_pairs": len(df), "dataset": str(a.data), "tiers": df.dataset_tier.value_counts().to_dict(), "label_source_note": "ALL labels are author-assigned or AI-assisted; none are human annotations",
                  "python": sys.version.split()[0], "platform": platform.platform(), "cv": cfg["cv"], "thresholds": thr, "model_status": {**{m: "run" for m in F.MODELS}, **F.NOT_RUN}}}
    models = list(F.MODELS)
    if a.st_path:
        st = F.register_sentence_transformer(df, a.st_path); R["meta"]["model_status"]["sentence_transformer_cosine"] = st
        if st.startswith("run"): models.append("st_cal")
    F.MODELS = models
    print("E1/E2 grouped CV ..."); oof = F.run_cv(df, cfg, models); R["e1_cv"] = F.summarize_cv(df, oof, thr, models)
    print("E3/E4 dataset + augmentation ablations ..."); R["e3_e4_ablation"] = F.run_dataset_ablation(data, cfg, n_rep=2 if a.quick else 5)
    print("E5 sanity ..."); R["e5_sanity"] = F.sanity_checks(df, cfg, models)
    print("E6 explainer ..."); R["e6_explainer"] = F.explainer_eval(df)
    rev = load_pairs(data / "pairs_reviewed.csv"); R["e6_explainer_reviewed_subset"] = F.explainer_eval(rev)
    best = max(F.MODELS, key=lambda m: R["e1_cv"][m]["spearman"]["mean"] if R["e1_cv"][m]["spearman"]["mean"] == R["e1_cv"][m]["spearman"]["mean"] else -9)
    R["best_model_by_cv_spearman"] = best; d = F.error_analysis(df, oof[best][0], best, out / "error_analysis.md", thr)
    pd.DataFrame({"pair_id": df.pair_id, "language": df.language, "legal_domain": df.legal_domain, "reference": df.lmp_score, **{m: oof[m][0] for m in F.MODELS}}).to_csv(out / "oof_predictions.csv", index=False)
    (out / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    # ---- figures (all from R / oof) ----
    names = F.MODELS; sp = [R["e1_cv"][m]["spearman"]["mean"] for m in names]; ci = [R["e1_cv"][m]["spearman_ci95_cluster_bootstrap"] for m in names]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.bar(names, sp, yerr=[[s - c[0] for s, c in zip(sp, ci)], [c[1] - s for s, c in zip(sp, ci)]], capsize=3, color="#4a78b5"); ax.axhline(0, c="k", lw=.6)
    ax.set_ylabel("Spearman (OOF, grouped CV)"); ax.set_title("Model comparison - 95% cluster-bootstrap CI"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out / "figures/e1_spearman.png", dpi=130); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, t in ((axs[0], "rmse", "RMSE (lower better)"), (axs[1], "f1_macro", "Macro-F1")):
        ax.bar(names, [R["e1_cv"][m][key]["mean"] for m in names], yerr=[R["e1_cv"][m][key]["sd_across_repeats"] for m in names], color="#6aa36a"); ax.set_title(t); ax.tick_params(axis="x", rotation=35)
    plt.tight_layout(); plt.savefig(out / "figures/e1_rmse_f1.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(9, 4)); w = .38; x = np.arange(len(names))
    for i, l in enumerate(("en", "hi")): ax.bar(x + (i - .5) * w, [R["e1_cv"][m]["by_language"][l]["spearman"] for m in names], w, label=l.upper())
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.legend(); ax.set_ylabel("Spearman"); ax.set_title("English vs Hindi (OOF)"); plt.tight_layout(); plt.savefig(out / "figures/e2_language.png", dpi=130); plt.close()
    cm = np.array(R["e1_cv"][best]["confusion_rep0"]); fig, ax = plt.subplots(figsize=(4.4, 4)); ax.imshow(cm, cmap="Blues"); labs = ["Not", "Partial", "Preserved"]
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs); ax.set_xlabel("predicted"); ax.set_ylabel("reference"); ax.set_title(f"Confusion - {best}")
    for i in range(3):
        for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="k")
    plt.tight_layout(); plt.savefig(out / "figures/e1_confusion_best.png", dpi=130); plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(9, 4)); axs[0].scatter(df.lmp_score + np.random.RandomState(0).uniform(-.15, .15, len(df)), oof[best][0], s=8, alpha=.5); axs[0].plot([1, 10], [1, 10], "r--", lw=1)
    axs[0].set_xlabel("reference"); axs[0].set_ylabel("OOF prediction"); axs[0].set_title(best); axs[1].hist(oof[best][0], bins=20, alpha=.6, label="predicted"); axs[1].hist(df.lmp_score, bins=20, alpha=.4, label="reference"); axs[1].legend(); axs[1].set_title("Prediction distribution")
    plt.tight_layout(); plt.savefig(out / "figures/e1_scatter_distribution.png", dpi=130); plt.close()
    ab = R["e3_e4_ablation"]["results"]; vs = list(ab); ms = list(next(iter(ab.values()))); fig, ax = plt.subplots(figsize=(10, 4)); w = .8 / len(ms)
    for i, m in enumerate(ms): ax.bar(np.arange(len(vs)) + i * w, [ab[v][m]["spearman"]["mean"] for v in vs], w, yerr=[ab[v][m]["spearman"]["sd"] for v in vs], label=m, capsize=2)
    ax.set_xticks(np.arange(len(vs)) + .4 - w / 2); ax.set_xticklabels(vs, rotation=15); ax.set_ylabel("Spearman on fixed grouped test"); ax.legend(fontsize=7); ax.set_title("Dataset / augmentation ablation"); plt.tight_layout(); plt.savefig(out / "figures/e3_ablation.png", dpi=130); plt.close()
    # ---- markdown report ----
    L = ["# Results (generated by scripts/run_experiments.py)", "", "> **READ FIRST.** Labels are AI-assisted (generation pass + partial blind second pass by the same model family) or author-assigned. No human annotation exists. "
         "Numbers show how methods behave on THIS dataset; they are not evidence about real legal text, real annotators, or pretrained transformers.", "",
         f"Dataset: `{a.data}/pairs_consolidated.csv`, {len(df)} pairs, {df.group_id.nunique()} groups. CV: grouped, k={cfg['cv']['k']}, repeats={cfg['cv']['repeats']}. Runtime {time.time()-t0:.0f}s.", "", "## Model status", ""]
    L += [f"- **{k}**: {v}" for k, v in R["meta"]["model_status"].items()]
    cols = ["spearman", "pearson", "rmse", "mae", "accuracy", "f1_macro", "f1_weighted", "collapse_index"]
    L += ["", "## E1 - grouped cross-validation (out-of-fold; mean over repeats; Spearman 95% CI = group-cluster bootstrap, repeat 0)", "", "| model | " + " | ".join(cols) + " | Spearman CI95 |", "|---|" + "---|" * (len(cols) + 1)]
    for m in names: r = R["e1_cv"][m]; L.append(f"| {m} | " + " | ".join(_fmt(r[c]["mean"]) for c in cols) + f" | [{_fmt(r['spearman_ci95_cluster_bootstrap'][0])}, {_fmt(r['spearman_ci95_cluster_bootstrap'][1])}] |")
    L += ["", "## E2 - by language (repeat 0 OOF)", "", "| model | lang | Spearman | RMSE | MAE | macro-F1 | accuracy |", "|---|---|---|---|---|---|---|"]
    for m in names:
        for l in ("en", "hi"): r = R["e1_cv"][m]["by_language"][l]; L.append(f"| {m} | {l} | {_fmt(r['spearman'])} | {_fmt(r['rmse'])} | {_fmt(r['mae'])} | {_fmt(r['f1_macro'])} | {_fmt(r['accuracy'])} |")
    L += ["", "## E3/E4 - dataset and augmentation ablation (fixed grouped test per repeat, mean +- sd over repeats)", "", f"Mean training sizes: {json.dumps(R['e3_e4_ablation']['mean_train_sizes'])}", "", "| train set | model | Spearman | RMSE | macro-F1 | accuracy |", "|---|---|---|---|---|---|"]
    for v, d_ in ab.items():
        for m, r in d_.items(): L.append(f"| {v} | {m} | {_fmt(r['spearman']['mean'])} +- {_fmt(r['spearman']['sd'])} | {_fmt(r['rmse']['mean'])} | {_fmt(r['f1_macro']['mean'])} | {_fmt(r['accuracy']['mean'])} |")
    L += ["", "## E5 - sanity checks (mean predicted score; identical should be high, unrelated low)", "", "| model | identical | shuffled/unrelated | test mean |", "|---|---|---|---|"]
    for m in names: r = R["e5_sanity"][m]; L.append(f"| {m} | {_fmt(r['identical_mean'])} | {_fmt(r['shuffled_unrelated_mean'])} | {_fmt(r['test_mean'])} |")
    e = R["e6_explainer"]; L += ["", "## E6 - rule-based explainer vs dataset labels (NOT a validated explanation system)", "", f"n={e['n']}. Micro-F1 of changed-element labels at confidence>=0 / 0.6 / 0.75: {_fmt(e['label_micro_f1_conf>=0.0'])} / {_fmt(e['label_micro_f1_conf>=0.6'])} / {_fmt(e['label_micro_f1_conf>=0.75'])}", "",
          "Flag rate (>=1 finding) by reference category, confidence>=0.0: " + json.dumps(e["flag_rate_by_category_conf>=0.0"]), "", "Flag rate by category, confidence>=0.6: " + json.dumps(e["flag_rate_by_category_conf>=0.6"]), "",
          "| label | precision | recall | F1 | support |", "|---|---|---|---|---|"] + [f"| {k} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |" for k, v in e["label_detection_all_findings"].items()]
    L += ["", f"Best model by CV Spearman: **{best}**. Error analysis: `error_analysis.md`. Per-pair OOF predictions: `oof_predictions.csv`. Figures: `figures/`."]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8"); print("done", round(time.time() - t0), "s")
main()
