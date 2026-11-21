"""OFFLINE FALLBACK regressor (not JUDGEBERT-style): ridge regression on cheap lexical features + rule-based legal-difference
features. It is trained on LMP annotations, so it learns from human-style judgments, but it is NOT a transformer and its
features come from the same rules used by the explainer. Use only when no pretrained transformer can be downloaded, and
interpret results with caution (see docs/METHODOLOGY.md)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from ..data.schema import ERROR_LABELS
from ..explain.analyzer import analyze
from ..preprocess import tokenize

def featurize(originals, simplified, langs, tfidf) -> np.ndarray:
    cos = tfidf.similarity(originals, simplified)
    rows = []
    for o, s, l, c in zip(originals, simplified, langs, cos):
        a = analyze(o, s, l)
        counts = [sum(f.category == lab for f in a.findings) for lab in ERROR_LABELS]
        to, ts = tokenize(o), tokenize(s)
        jacc = len(set(to) & set(ts)) / max(1, len(set(to) | set(ts)))
        rows.append([c, jacc, len(ts) / max(1, len(to)), len(a.findings), max([f.confidence for f in a.findings], default=0.0), *counts])
    return np.array(rows, float)

class FeatureRegressor:
    name = "Feature-ridge (offline fallback)"
    def __init__(self, tfidf, alpha: float = 3.0):
        self.tfidf, self.model = tfidf, Ridge(alpha=alpha)
    def fit(self, df):
        self.model.fit(featurize(list(df.original), list(df.simplified), list(df.language), self.tfidf), df.lmp_score.values); return self
    def predict_df(self, df):
        return np.clip(self.model.predict(featurize(list(df.original), list(df.simplified), list(df.language), self.tfidf)), 1, 10)
