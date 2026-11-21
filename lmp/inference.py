"""Single inference entry point used by the CLI and the Streamlit app.

Backends (always reported in `Prediction.backend`, never silently substituted):
  transformer : LMPCrossEncoder checkpoint (only if a compatible checkpoint directory loads)
  lexrule_gbm : offline fallback - gradient boosting on lexical + rule features (NOT a transformer)
  tfidf       : last-resort unsupervised TF-IDF cosine (NOT calibrated to labels; known to be inversely related to the labels on this dataset)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, pandas as pd
from .config import ROOT, load_config, score_to_category
from .data.io import load_pairs
from .explain.analyzer import analyze, Analysis
from .langid import identify_language
from .preprocess import normalize

class InferenceError(ValueError):
    pass

@dataclass
class Prediction:
    score: float
    category: str
    backend: str
    is_transformer: bool
    language: str
    analysis: Analysis
    class_probs: dict | None = None
    warnings: list[str] = field(default_factory=list)
    backend_note: str = ""


class LMPPredictor:
    def __init__(self, cfg: dict | None = None, checkpoint_dir: str | Path | None = None, data_path: str | Path | None = None, prefer: str = "auto"):
        self.cfg = cfg or load_config(); self.thr = self.cfg["category_thresholds"]; self.model = self.tok = None; self.notes: list[str] = []
        self.backend = None
        ck = Path(checkpoint_dir) if checkpoint_dir else ROOT / self.cfg["checkpoint_dir"] / "crossenc"
        if prefer in ("auto", "transformer") and (ck / "meta.json").exists():
            try:
                from .models.train_crossenc import load_checkpoint
                self.model, self.tok, self.meta = load_checkpoint(ck); self.backend = "transformer"; self.max_length = self.meta["max_length"]
                self.pretrained = bool(self.meta.get("pretrained_weights_used", True))
            except Exception as e:
                self.notes.append(f"Transformer checkpoint at {ck} could not be used ({type(e).__name__}: {str(e)[:120]}); using fallback.")
        elif prefer == "transformer":
            raise InferenceError(f"No transformer checkpoint found at {ck}")
        elif prefer == "auto":
            self.notes.append("No transformer checkpoint found (train one with scripts/train_crossenc.py); using the labelled fallback model.")
        if self.backend is None:
            from .models.sk_models import SKScorer
            kind = "tfidf_raw" if prefer == "tfidf" else "lexrule_gbm"
            train = load_pairs(data_path or ROOT / self.cfg["data_path"])
            self.sk = SKScorer(kind).fit(train); self.backend = kind if kind == "tfidf_raw" else "lexrule_gbm"; self.n_train = len(train)

    def describe(self) -> str:
        if self.backend == "transformer":
            return "Transformer cross-encoder" + ("" if self.pretrained else " (randomly initialised - NOT pretrained; diagnostic only)")
        return {"lexrule_gbm": "Fallback: gradient boosting on lexical+rule features (NOT a transformer)", "tfidf_raw": "TF-IDF cosine (uncalibrated, NOT a transformer)"}[self.backend]

    def predict(self, original: str, simplified: str, language: str | None = None) -> Prediction:
        o, s = normalize(original or ""), normalize(simplified or "")
        if not o or not s: raise InferenceError("Both the original and the simplified sentence must be non-empty.")
        detected = identify_language(o)
        if language in (None, "auto"): language = detected if detected in ("en", "hi") else "en"
        if language not in ("en", "hi"): raise InferenceError(f"Unsupported language {language!r}; use 'en' or 'hi'.")
        warns = []
        if detected in ("en", "hi") and detected != language: warns.append(f"Selected language '{language}' differs from the detected script ('{detected}').")
        if identify_language(s) != detected and identify_language(s) != "unknown": warns.append("Original and simplified sentences appear to be in different languages; this task assumes the same language.")
        if len(o.split()) < 4 or len(s.split()) < 3: warns.append("Very short input: scores on short fragments are unreliable.")
        if len(o.split()) > 60 or len(s.split()) > 60: warns.append("Long input: the model was developed on single sentences and may truncate.")
        probs = None
        if o.lower() == s.lower():
            score = 10.0; warns.append("Texts are identical (score set by rule).")
        elif self.backend == "transformer":
            from .models.train_crossenc import predict as tpred
            sc, pr, _ = tpred(self.model, self.tok, pd.DataFrame({"original": [o], "simplified": [s], "language": [language]}), self.max_length)
            score, probs = float(sc[0]), {c: float(p) for c, p in zip(["Not Preserved", "Partially Preserved", "Preserved"], pr[0])}
        else:
            score = float(self.sk.predict(pd.DataFrame({"original": [o], "simplified": [s], "language": [language]}))[0])
        analysis = analyze(o, s, language)
        cat = score_to_category(score, self.thr)
        if min(abs(score - self.thr["not_preserved_below"]), abs(score - self.thr["preserved_from"])) < 0.75: warns.append("Score is close to a category boundary: treat the category as uncertain.")
        strong = [f for f in analysis.findings if f.confidence >= 0.75]
        if cat == "Preserved" and strong: warns.append("The model says 'Preserved' but a high-confidence rule found a change - check the highlighted findings.")
        if cat == "Not Preserved" and not analysis.findings: warns.append("The model says 'Not Preserved' but no rule explains why - the explanation list is incomplete, not the verdict wrong.")
        if self.backend != "transformer": warns.append("Fallback model in use (not a transformer). It was trained on a small AI-assisted dataset and exploits lexical patterns; do not treat the score as a legal judgement.")
        return Prediction(round(score, 2), cat, self.backend, self.backend == "transformer", language, analysis, probs, warns + self.notes, self.describe())
