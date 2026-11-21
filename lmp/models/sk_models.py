"""CPU/offline scikit-learn scorers used in fair comparisons. All are trained on the TRAIN portion only (calibration included).
  tfidf_raw   : 1 + 9*cosine(word TF-IDF)                       (unsupervised, no labels)
  tfidf_cal   : linear calibration of the above on train
  lex_ridge   : ridge on LEXICAL pair features only (word/char cosine, Jaccard, length ratio, removed/added token fractions)
  rule_ridge  : ridge on the rule-analyzer features only         (circularity caveat: same lexicon as the explainer)
  lexrule_ridge / lexrule_gbm : both feature groups
The rule-based features never see gold labels, but their design was informed by the same kinds of examples the labels describe, so
rule_* results are NOT independent semantic validation (docs/METHODOLOGY.md).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ..data.schema import ERROR_LABELS
from ..explain.analyzer import analyze
from ..preprocess import tokenize
from .tfidf import TfidfBaseline
from ..evaluation.metrics import calibrate_linear

_RULE_CACHE: dict = {}

def rule_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for r in df.itertuples():
        key = (r.language, r.original, r.simplified)
        if key not in _RULE_CACHE:
            a = analyze(r.original, r.simplified, r.language, min_confidence=0.0)
            _RULE_CACHE[key] = [sum(f.category == lab for f in a.findings) for lab in ERROR_LABELS] + [len(a.findings), max([f.confidence for f in a.findings], default=0.0), sum(f.confidence for f in a.findings)]
        rows.append(_RULE_CACHE[key])
    return np.array(rows, float)

def lexical_features(df: pd.DataFrame, word: TfidfBaseline, char: TfidfVectorizer) -> np.ndarray:
    o, s = list(df.original), list(df.simplified)
    cw = word.similarity(o, s)
    Xo, Xs = char.transform(o), char.transform(s); cc = np.asarray(Xo.multiply(Xs).sum(axis=1)).ravel()
    rows = []
    for a, b, w, c in zip(o, s, cw, cc):
        to, ts = tokenize(a), tokenize(b); A, B = set(to), set(ts)
        rows.append([w, c, len(A & B) / max(1, len(A | B)), len(ts) / max(1, len(to)), len(A - B) / max(1, len(A)), len(B - A) / max(1, len(B))])
    return np.array(rows, float)

class SKScorer:
    def __init__(self, kind: str, alpha: float = 3.0):
        self.kind, self.alpha = kind, alpha
    @property
    def name(self): return self.kind
    def fit(self, df: pd.DataFrame):
        self.word = TfidfBaseline().fit(list(df.original) + list(df.simplified))
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(list(df.original) + list(df.simplified))
        if self.kind in ("tfidf_raw", "tfidf_cal"):
            raw = self.word.predict(list(df.original), list(df.simplified))
            self.cal = calibrate_linear(raw, df.lmp_score.values) if self.kind == "tfidf_cal" else None
            return self
        X = self._X(df); self.sc = StandardScaler().fit(X); Xs = self.sc.transform(X)
        self.m = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0) if self.kind.endswith("gbm") else Ridge(alpha=self.alpha)
        self.m.fit(Xs, df.lmp_score.values); return self
    def _X(self, df):
        parts = []
        if self.kind.startswith("lex"): parts.append(lexical_features(df, self.word, self.char))
        if "rule" in self.kind: parts.append(rule_features(df))
        return np.hstack(parts)
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind.startswith("tfidf"):
            p = self.word.predict(list(df.original), list(df.simplified)); return self.cal(p) if self.cal else p
        return np.clip(self.m.predict(self.sc.transform(self._X(df))), 1, 10)

SK_KINDS = ["tfidf_raw", "tfidf_cal", "lex_ridge", "rule_ridge", "lexrule_ridge", "lexrule_gbm"]
