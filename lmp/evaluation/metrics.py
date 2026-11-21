from __future__ import annotations
import warnings
import numpy as np
warnings.filterwarnings("ignore", message="An input array is constant")
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from ..config import score_to_category
from ..data.schema import ERROR_LABELS, CATEGORIES

def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"n": int(len(y_true)), "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))), "mae": float(np.mean(np.abs(y_true - y_pred)))}
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson"], out["spearman"] = float(pearsonr(y_true, y_pred)[0]), float(spearmanr(y_true, y_pred)[0])
    else:
        out["pearson"] = out["spearman"] = float("nan")
    return out

def category_metrics(y_true, y_pred, thresholds=None) -> dict:
    t = [score_to_category(s, thresholds) for s in y_true]; p = [score_to_category(s, thresholds) for s in y_pred]
    pr, rc, f1, _ = precision_recall_fscore_support(t, p, labels=CATEGORIES, average="macro", zero_division=0)
    return {"precision_macro": float(pr), "recall_macro": float(rc), "f1_macro": float(f1),
            "accuracy": float(np.mean([a == b for a, b in zip(t, p)])), "kappa": float(cohen_kappa_score(t, p)) if len(set(t) | set(p)) > 1 else float("nan")}

def label_detection_metrics(gold: list[set], pred: list[set]) -> dict:
    """Multi-label precision/recall/F1 for legal-error detection (per label + micro + macro)."""
    rows = {}
    for lab in ERROR_LABELS:
        tp = sum(lab in g and lab in p for g, p in zip(gold, pred)); fp = sum(lab not in g and lab in p for g, p in zip(gold, pred))
        fn = sum(lab in g and lab not in p for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * pr * rc / (pr + rc) if pr == pr and rc == rc and pr + rc > 0 else float("nan")
        rows[lab] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    tp, fp, fn = (sum(r[k] for r in rows.values()) for k in ("tp", "fp", "fn"))
    pr, rc = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    rows["micro"] = {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return rows

def calibrate_linear(train_pred, train_true, allow_negative: bool = False):
    """Least-squares map of a raw similarity score to the 1-10 scale (fit on TRAIN only) so RMSE/MAE of baselines are fair."""
    x, y = np.asarray(train_pred, float), np.asarray(train_true, float)
    a, b = np.polyfit(x, y, 1) if np.std(x) > 0 else (0.0, y.mean())
    if a < 0 and not allow_negative:   # a similarity score cannot sensibly be negatively related to preservation; fall back to the train mean
        a, b = 0.0, float(y.mean())
    return lambda x: np.clip(a * np.asarray(x, float) + b, 1, 10)


def full_metrics(y_true, y_pred, thresholds=None) -> dict:
    """Regression + classification metrics for one prediction vector (categories derived from scores with the SAME thresholds for all models)."""
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    t = [score_to_category(s, thresholds) for s in yt]; p = [score_to_category(s, thresholds) for s in yp]
    out = regression_metrics(yt, yp)
    out.update({"bias": float(np.mean(yp - yt)), "pred_std": float(np.std(yp)), "target_std": float(np.std(yt)), "collapse_index": float(np.std(yp) / max(1e-8, np.std(yt))),
                "accuracy": float(accuracy_score(t, p)), "f1_macro": float(f1_score(t, p, labels=CATEGORIES, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(t, p, labels=CATEGORIES, average="weighted", zero_division=0))})
    pr, rc, f1, sup = precision_recall_fscore_support(t, p, labels=CATEGORIES, zero_division=0)
    out["precision_macro"], out["recall_macro"] = float(pr.mean()), float(rc.mean())
    out["per_class"] = {c: {"precision": float(a), "recall": float(b), "f1": float(f), "support": int(n)} for c, a, b, f, n in zip(CATEGORIES, pr, rc, f1, sup)}
    out["confusion"] = confusion_matrix(t, p, labels=CATEGORIES).tolist()
    return out
