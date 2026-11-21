"""Baseline B: multilingual Sentence-Transformer cosine similarity.
Requires `sentence-transformers` and a one-time download of the model from the Hugging Face Hub."""
from __future__ import annotations
import numpy as np

class SentenceTransformerBaseline:
    name = "Sentence-Transformer cosine"

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def similarity(self, originals: list[str], simplified: list[str]) -> np.ndarray:
        a = self.model.encode(originals, normalize_embeddings=True, show_progress_bar=False)
        b = self.model.encode(simplified, normalize_embeddings=True, show_progress_bar=False)
        return (a * b).sum(axis=1)

    def predict(self, originals: list[str], simplified: list[str]) -> np.ndarray:
        return 1.0 + 9.0 * np.clip(self.similarity(originals, simplified), 0, 1)
