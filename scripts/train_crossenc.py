"""Train / evaluate the multi-task cross-encoder (from project root).
Pretrained XLM-R (needs the Hugging Face Hub once, or --local-model-path to a cached dir):
    python scripts/train_crossenc.py --backbone xlm-roberta-base --data data/expanded/pairs_consolidated.csv
Offline smoke run with a tiny RANDOMLY-INITIALISED encoder (verifies the pipeline only; says nothing about pretrained quality):
    python scripts/train_crossenc.py --tiny --epochs 5
Checkpoints go to checkpoints/crossenc (consumed by lmp.inference.LMPPredictor). Test metrics are computed ONCE, after model selection on the validation split."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from lmp.config import load_config, set_seed
from lmp.data.io import load_pairs
from lmp.data.splits import grouped_split, split_report
from lmp.evaluation.metrics import full_metrics
from lmp.models import train_crossenc as T

ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--backbone"); ap.add_argument("--local-model-path"); ap.add_argument("--tiny", action="store_true")
ap.add_argument("--epochs", type=int); ap.add_argument("--seed", type=int); ap.add_argument("--out", default="checkpoints/crossenc"); a = ap.parse_args()
cfg = load_config(); rc = cfg["crossenc"]; seed = a.seed if a.seed is not None else cfg["seed"]; set_seed(seed)
if a.backbone: rc["backbone"] = a.backbone
if a.epochs: rc["epochs"] = a.epochs
df = load_pairs(ROOT / (a.data or cfg["data_path"])); tr, va, te = grouped_split(df, seed=seed)
run_dir = ROOT / a.out
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "split_report.json").write_text(
    json.dumps(split_report(tr, va, te), indent=2, ensure_ascii=False),
    encoding="utf-8"
)
if a.tiny:
    from lmp.models.tiny import build_tokenizer, tiny_config; from lmp.models.crossenc import LMPCrossEncoder
    tok = build_tokenizer(list(tr.original) + list(tr.simplified)); rc.update(pretrained=False, backbone="tiny-random", max_length=96, encoder_lr=1e-3, head_lr=2e-3, freeze_layers=0, batch_size=16, grad_accum=1, patience=6)
    model = LMPCrossEncoder(config=tiny_config(len(tok)))
else:
    from transformers import AutoTokenizer
    src = a.local_model_path or rc.get("local_model_path") or rc["backbone"]
    try: tok = AutoTokenizer.from_pretrained(src)
    except Exception as e: sys.exit(f"Cannot load pretrained weights/tokenizer from {src!r}: {type(e).__name__}. No model was trained. Provide --local-model-path or Hub access.")
    from lmp.models.crossenc import LMPCrossEncoder
    model = LMPCrossEncoder(src, rc["dropout"], use_language_embedding=rc["language_embedding"], freeze_layers=rc["freeze_layers"])
model, hist = T.fit(tr, va, cfg, seed, tok, model=model, ckpt_dir=run_dir)
pred, probs, cats = T.predict(model, tok, te, rc["max_length"])
res = full_metrics(te.lmp_score, pred, cfg["category_thresholds"]); res["tiny_random_init"] = bool(a.tiny)
(run_dir / "test_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k not in ("per_class", "confusion")}, indent=1)); print("best epoch", hist.get("best_epoch"), "| collapse index (val, last):", round(hist["collapse_index"][-1], 3))
