"""Configuration loading and global reproducibility helpers."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.json"

def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def score_to_category(score: float, thresholds: dict | None = None) -> str:
    """Map a 1-10 LMP score to a category. Thresholds are provisional and must be
    validated against real human annotations (see docs/METHODOLOGY.md)."""
    t = thresholds or load_config()["category_thresholds"]
    if score < t["not_preserved_below"]:
        return "Not Preserved"
    if score >= t["preserved_from"]:
        return "Preserved"
    return "Partially Preserved"
