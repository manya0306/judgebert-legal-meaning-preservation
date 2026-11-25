"""Publication-style figures from computed results (PNG 300 dpi + vector PDF). Uninformative plots are skipped, never faked."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .ingest import CATEGORIES, STUDENT_SHEETS  # noqa: E402
from .stats import pairwise_matrix, safe_corr  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})
LANGC = {"en": "#1f77b4", "hi": "#d95f02"}
LANGN = {"en": "English", "hi": "Hindi"}
SHORT = {"Not Preserved": "Not\nPreserved", "Partially Preserved": "Partially\nPreserved", "Preserved": "Preserved"}


def _save(fig, out: Path, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _cm(ax, truth, pred, title, xlabel, ylabel):
    t = [a for a, b in zip(truth, pred) if isinstance(a, str) and isinstance(b, str)]
    p = [b for a, b in zip(truth, pred) if isinstance(a, str) and isinstance(b, str)]
    cm = confusion_matrix(t, p, labels=list(CATEGORIES))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), [SHORT[c] for c in CATEGORIES])
    ax.set_yticks(range(3), [SHORT[c] for c in CATEGORIES])
    ax.set_title(f"{title}\n(n={len(t)}, accuracy={np.mean(np.array(t) == np.array(p)):.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def make_figures(cons, ind, tables, out_dir, model_df=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    ann = [a for a in STUDENT_SHEETS if f"score_{a}" in cons]

    def run(name, fn):
        try:
            r = fn()
            status[name] = r if r else "saved"
        except Exception as e:  # noqa: BLE001
            status[name] = f"SKIPPED: {type(e).__name__}: {e}"
    # 1 heatmap
    def f1():
        langs = [("en", ann), ("hi", [a for a in ann if a != "A4"])]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9))
        for ax, (lg, rs) in zip(axes, langs):
            M = pairwise_matrix(cons, lg, rs)
            ax.imshow(np.ma.masked_invalid(M.to_numpy(float)), cmap="Blues", vmin=0, vmax=1)
            for i in range(len(rs)):
                for j in range(len(rs)):
                    v = M.iloc[i, j]
                    ax.text(j, i, "" if v != v else f"{v:.2f}", ha="center", va="center", color="white" if v == v and v > 0.6 else "black")
            ax.set_xticks(range(len(rs)), rs)
            ax.set_yticks(range(len(rs)), rs)
            ax.set_title(f"{LANGN[lg]} (n={int((cons.language == lg).sum())} pairs)")
        fig.suptitle("Pairwise quadratic weighted kappa between student annotators (scores 1-10)")
        return _save(fig, out, "fig01_annotator_agreement_heatmap")
    run("fig01_annotator_agreement_heatmap", f1)
    # 2 distributions
    def f2():
        stu, exp = ind[ind.annotator_role == "student"], ind[ind.annotator_role == "expert"]
        groups = [(f"{a}\n(n={int((stu.annotator_id == a).sum())})", stu[stu.annotator_id == a].human_score.to_numpy()) for a in ann]
        groups += [(f"Expert\n(n={len(exp)})", exp.human_score.to_numpy()), (f"Reference\n(n={len(cons)})", cons.ref_score.to_numpy())]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.boxplot([g[1] for g in groups], tick_labels=[g[0] for g in groups], widths=0.55, showmeans=True)
        rng = np.random.default_rng(0)
        for k, (_, v) in enumerate(groups, 1):
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=7, alpha=0.35, color="k")
        ax.set_ylabel("Preservation score")
        ax.set_title("Score distributions by annotator (reference = AI-generated / author-assigned label)")
        return _save(fig, out, "fig02_score_distributions_by_annotator")
    run("fig02_score_distributions_by_annotator", f2)
    # 3 reference vs student mean
    def f3():
        fig, ax = plt.subplots(figsize=(5.2, 5))
        for lg in ("en", "hi"):
            for ai, mk in ((True, "o"), (False, "^")):
                d = cons[(cons.language == lg) & (cons.reference_label_type.astype(str).str.startswith("AI-GENERATED") == ai)]
                ax.scatter(d.student_mean, d.ref_score, c=LANGC[lg], marker=mk, alpha=0.7, s=28, label=f"{LANGN[lg]}, {'AI-generated' if ai else 'author-assigned'} (n={len(d)})")
        ax.plot([1, 10], [1, 10], "k--", lw=0.8)
        d = cons
        ax.set_title(f"Reference label vs student mean (n={len(d)})\nSpearman={safe_corr(d.student_mean, d.ref_score):.2f}, MAE={np.mean(np.abs(d.ref_score - d.student_mean)):.2f}, bias={np.mean(d.ref_score - d.student_mean):+.2f}")
        ax.set_xlabel("Student mean score")
        ax.set_ylabel("Reference score")
        ax.legend(fontsize=7, loc="upper left")
        return _save(fig, out, "fig03_reference_vs_student_consensus")
    run("fig03_reference_vs_student_consensus", f3)
    # 4 student vs expert
    def f4():
        e = cons[cons.expert_reviewed]
        if len(e) < 5:
            return "SKIPPED: fewer than 5 expert-reviewed pairs"
        fig, ax = plt.subplots(figsize=(5.2, 5))
        for lg in ("en", "hi"):
            d = e[e.language == lg]
            ax.scatter(d.student_mean, d.expert_score, c=LANGC[lg], alpha=0.8, s=32, label=f"{LANGN[lg]} (n={len(d)})")
        ax.plot([1, 10], [1, 10], "k--", lw=0.8)
        ax.set_title(f"Student mean vs expert (n={len(e)})\nSpearman={safe_corr(e.student_mean, e.expert_score):.2f}, MAE={np.mean(np.abs(e.student_mean - e.expert_score)):.2f}, bias(student-expert)={np.mean(e.student_mean - e.expert_score):+.2f}")
        ax.set_xlabel("Student mean score")
        ax.set_ylabel("Expert score (single legal professional)")
        ax.legend(fontsize=8)
        return _save(fig, out, "fig04_student_consensus_vs_expert")
    run("fig04_student_consensus_vs_expert", f4)
    # 5 bias distribution
    def f5():
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bins = np.arange(-6, 6.6, 0.5)
        for lg in ("en", "hi"):
            b = (cons[cons.language == lg].ref_score - cons[cons.language == lg].student_mean)
            ax.hist(b, bins=bins, alpha=0.55, color=LANGC[lg], label=f"{LANGN[lg]} (n={len(b)}, mean {b.mean():+.2f})")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Reference score - student mean (positive = reference more favourable)")
        ax.set_ylabel("Pairs")
        ax.legend(fontsize=8)
        ax.set_title("AI/author reference vs student consensus: signed bias")
        return _save(fig, out, "fig05_reference_bias_distribution")
    run("fig05_reference_bias_distribution", f5)
    # 6 confusion matrices
    def f6():
        e = cons[cons.expert_reviewed]
        n = 3 if len(e) >= 5 else 2
        fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
        _cm(axes[0], cons.student_mean_category, cons.ref_category_harmonized, "Reference (harmonised 3.5/6.5)", "Reference category", "Student-mean category")
        _cm(axes[1], cons.student_mean_category, cons.ref_category_original, "Reference (repo thresholds 4/7.5)", "Reference category", "Student-mean category")
        if n == 3:
            _cm(axes[2], e.expert_category, e.student_mean_category, "Student mean vs expert", "Student-mean category", "Expert category")
        fig.tight_layout()
        return _save(fig, out, "fig06_category_confusion_matrices")
    run("fig06_category_confusion_matrices", f6)
    # 7 english vs hindi
    def f7():
        ag = tables["student_agreement"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        mets = [("krippendorff_alpha_ordinal_score", "Krippendorff\nalpha (ordinal)"), ("mean_pairwise_qwk_score", "Mean pairwise\nQWK"), ("fleiss_kappa_category", "Fleiss kappa\n(category)")]
        for k, (sub, lg) in enumerate((("english_A1-A3", "en"), ("hindi", "hi"))):
            vals, err = [], []
            for m, _ in mets:
                r = ag[(ag.scope == "student_agreement") & (ag.subset == sub) & (ag.metric == m)]
                v = float(r.value.iloc[0]) if len(r) else np.nan
                lo, hi = (float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])) if len(r) else (np.nan, np.nan)
                vals.append(v)
                err.append([max(v - lo, 0) if lo == lo else 0, max(hi - v, 0) if hi == hi else 0])
            axes[0].bar(np.arange(3) + (k - 0.5) * 0.36, vals, 0.34, yerr=np.array(err).T, color=LANGC[lg], capsize=3, label=f"{LANGN[lg]} (A1-A3, n={int((cons.language == lg).sum())})")
        axes[0].set_xticks(range(3), [m[1] for m in mets])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Student agreement (95% group-bootstrap CI)")
        axes[0].legend(fontsize=7)
        w = np.arange(2)
        for k, lg in enumerate(("en", "hi")):
            d = cons[cons.language == lg]
            axes[1].bar(w + (k - 0.5) * 0.36, [np.mean(np.abs(d.ref_score - d.student_mean)), abs(np.mean(d.ref_score - d.student_mean))], 0.34, color=LANGC[lg], label=LANGN[lg])
        axes[1].set_xticks(w, ["MAE", "|bias|"])
        axes[1].set_title("Reference vs student mean (score points)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        return _save(fig, out, "fig07_english_vs_hindi")
    run("fig07_english_vs_hindi", f7)
    # 8 complexity vs disagreement
    def f8():
        order = ["easy", "moderate", "difficult"]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for k, lg in enumerate(("en", "hi")):
            data = [cons[(cons.language == lg) & (cons.complexity == c)].student_sd.dropna().to_numpy() for c in order]
            pos = np.arange(3) * 3 + k
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showmeans=True)
            for b in bp["boxes"]:
                b.set_facecolor(LANGC[lg])
                b.set_alpha(0.5)
            for p_, d_ in zip(pos, data):
                ax.text(p_, ax.get_ylim()[0], f"n={len(d_)}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(3) * 3 + 0.5, [c.title() for c in order])
        ax.set_ylabel("Student score SD per pair")
        ax.set_title("Student disagreement by complexity (blue English, orange Hindi)")
        return _save(fig, out, "fig08_complexity_vs_disagreement")
    run("fig08_complexity_vs_disagreement", f8)
    # 9 semantic label agreement
    def f9():
        d = cons.dropna(subset=["student_modal_change"])
        g = d.groupby("student_modal_change").agg(n=("pair_id", "size"), share=("student_change_agreement_share", "mean")).sort_values("share")
        if len(g) < 2:
            return "SKIPPED: fewer than 2 modal semantic labels"
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(g) + 1.5))
        ax.barh([f"{i} (n={r.n})" for i, r in g.iterrows()], g.share, color="#4c72b0")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean share of student raters choosing the modal label")
        ax.set_title("Semantic change-type agreement among students, by modal label")
        return _save(fig, out, "fig09_semantic_label_agreement")
    run("fig09_semantic_label_agreement", f9)
    # 10 model comparison
    def f10():
        if model_df is None or model_df.empty or not (model_df.status == "EVALUATED").any():
            return "SKIPPED: no model predictions available"
        d = model_df[(model_df.status == "EVALUATED") & (model_df.target == "student_mean") & (model_df.subset == "all")]
        if d.empty:
            return "SKIPPED: no evaluated models on the full annotated set"
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, col, t in ((axes[0], "mae", "MAE vs student mean (lower better)"), (axes[1], "spearman", "Spearman vs student mean")):
            bars = ax.barh(d.model_name, d[col], color=["#4c72b0" if "VERIFIED" in str(h) and "NOT" not in str(h) else "#c44e52" for h in d.heldout_status])
            ax.set_title(t)
        axes[0].text(0.0, -0.25, "Red = held-out status unverified or contains training data; all pairs (n=%d)" % int(d.n.max()), transform=axes[0].transAxes, fontsize=7)
        fig.tight_layout()
        return _save(fig, out, "fig10_model_comparison")
    run("fig10_model_comparison", f10)
    return status
