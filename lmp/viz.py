from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def scatter_models(preds: dict, y, path, title="Predicted vs reference LMP"):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.2 * len(preds), 4), squeeze=False)
    for ax, (name, p) in zip(axes[0], preds.items()):
        ax.scatter(y, p, alpha=.7); ax.plot([1, 10], [1, 10], "k--", lw=.8)
        ax.set_xlabel("Reference LMP"); ax.set_ylabel("Predicted LMP"); ax.set_title(name, fontsize=9); ax.set_xlim(0.5, 10.5); ax.set_ylim(0.5, 10.5)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def bar_metrics(table: dict, metric: str, path, title=None):
    names = list(table); vals = [table[n].get(metric, np.nan) for n in names]
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 3.6)); ax.bar(names, vals, color="#2a6f7f")
    ax.set_ylabel(metric); ax.set_title(title or metric); plt.xticks(rotation=20, ha="right"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def grouped_bar(groups: dict, metric: str, path, title=None):
    """groups: {model: {group: value}}"""
    models = list(groups); cats = sorted({g for m in groups.values() for g in m}); w = .8 / len(models)
    fig, ax = plt.subplots(figsize=(1.5 * len(cats) + 3, 3.8))
    for i, m in enumerate(models):
        ax.bar(np.arange(len(cats)) + i * w, [groups[m].get(c, np.nan) for c in cats], w, label=m)
    ax.set_xticks(np.arange(len(cats)) + w * (len(models) - 1) / 2); ax.set_xticklabels(cats); ax.set_ylabel(metric); ax.legend(fontsize=7)
    ax.set_title(title or metric); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def loss_curves(history: dict, path):
    fig, ax = plt.subplots(figsize=(5, 3.5)); ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="train"); ax.plot(ep, history["val_loss"], label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
