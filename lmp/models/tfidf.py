"""Baseline A: TF-IDF cosine similarity (lexical). Score = 1 + 9*cosine, so identical text -> 10."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ..preprocess import tokenize

class TfidfBaseline:
    name = "TF-IDF cosine"

    def __init__(self):
        # Custom tokenizer keeps Devanagari tokens intact (sklearn's default regex splits on matras).
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, texts: list[str]) -> "TfidfBaseline":
        self.vec.fit(texts)
        return self

    def similarity(self, originals: list[str], simplified: list[str]) -> np.ndarray:
        a, b = self.vec.transform(originals), self.vec.transform(simplified)
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()  # rows are L2-normalised

    def predict(self, originals: list[str], simplified: list[str]) -> np.ndarray:
        return 1.0 + 9.0 * np.clip(self.similarity(originals, simplified), 0, 1)
