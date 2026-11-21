"""Fine-tune the JUDGEBERT-inspired regressor and evaluate it on the test split.
Usage: python scripts/train_regressor.py [--config configs/default.json] [--data PATH] [--holdout-domain tenancy] [--backbone bert-base-multilingual-cased]
Needs torch + transformers and a one-time download of the backbone from the Hugging Face Hub."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from lmp.config import load_config, set_seed
from lmp.data.io import load_pairs
from lmp.data.splits import document_split, domain_holdout_split
from lmp.data.augment import augment_with_sanity_pairs
from lmp.evaluation.metrics import regression_metrics, category_metrics
from lmp.models import trainer
from lmp import viz

ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--data"); ap.add_argument("--holdout-domain"); ap.add_argument("--backbone"); ap.add_argument("--seed", type=int)
ap.add_argument("--no-augment", action="store_true")
a = ap.parse_args(); cfg = load_config(a.config)
if a.backbone: cfg["regressor"]["backbone"] = a.backbone
if a.seed is not None: cfg["seed"] = a.seed
set_seed(cfg["seed"]); df = load_pairs(a.data or cfg["data_path"])
train, val, test = domain_holdout_split(df, a.holdout_domain, seed=cfg["seed"]) if a.holdout_domain else document_split(df, seed=cfg["seed"])
if cfg.get("augment_sanity_pairs", True) and not a.no_augment: train = augment_with_sanity_pairs(train, cfg["seed"])
tag = f"{cfg['regressor']['backbone'].replace('/', '_')}_seed{cfg['seed']}" + (f"_holdout-{a.holdout_domain}" if a.holdout_domain else "")
ckpt = Path(cfg["checkpoint_dir"]) / tag; out = Path(cfg["output_dir"]) / "regressor" / tag; out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(cfg["regressor"]["backbone"])
model, hist = trainer.fit(train, val, cfg, cfg["seed"], tok, ckpt_dir=ckpt)
pred = trainer.predict(model, tok, list(test.original), list(test.simplified), cfg["regressor"]["max_length"])
res = {**regression_metrics(test.lmp_score, pred), **category_metrics(test.lmp_score, pred, cfg["category_thresholds"]), "tag": tag, "n_train": len(train), "n_test": len(test)}
(out / "test_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8"); viz.loss_curves(hist, out / "loss_curves.png")
print(json.dumps(res, indent=2)); print("checkpoint:", ckpt)
