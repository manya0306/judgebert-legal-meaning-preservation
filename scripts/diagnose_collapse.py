"""Controlled diagnosis of the prediction-collapse failure (tiny randomly-initialised encoder; NO pretrained weights).
Compares, on the same grouped split and the same encoder:
  legacy@2e-5 : the shipped design and learning rate        legacy@1e-3 : shipped design at an LR a tiny model can use
  new design  : LMPCrossEncoder (linear heads, multi-task, differential LR, rank-based checkpoint selection)
Writes outputs/collapse_diagnosis.json. Interpretation limits are in docs/EXPERIMENTS.md."""
import json, sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.stats import spearmanr
from lmp.config import load_config, ROOT
from lmp.data.io import load_pairs
from lmp.data.splits import grouped_split
from lmp.models.tiny import build_tokenizer, tiny_config
from lmp.models import trainer as legacy
from lmp.models.judgebert_regressor import LMPRegressor
from lmp.models.crossenc import LMPCrossEncoder
from lmp.models import train_crossenc as new
from lmp.evaluation.metrics import regression_metrics, category_metrics

def main(seeds=(0, 1, 2), epochs=30, data="data/expanded/pairs_consolidated.csv"):
    torch.set_num_threads(1); cfg0 = load_config(); df = load_pairs(ROOT / data); fp = ROOT / "outputs/collapse_diagnosis.json"
    out = json.loads(fp.read_text()) if (fp.exists() and "--fresh" not in sys.argv and "--smoke" not in sys.argv) else {"note": "tiny random-init encoder; diagnostic only", "runs": []}
    out["runs"] = [r for r in out["runs"] if r["seed"] not in seeds]
    for seed in seeds:
        tr, va, te = grouped_split(df, seed=seed); tok = build_tokenizer(list(tr.original) + list(tr.simplified)); tc = tiny_config(len(tok))
        for name in ("legacy@2e-5", "legacy@1e-3", "new"):
            cfg = copy.deepcopy(cfg0); t0 = time.time(); torch.manual_seed(seed)
            if name.startswith("legacy"):
                cfg["regressor"].update(epochs=epochs, batch_size=16, max_length=96, patience=3, device="cpu", learning_rate=2e-5 if "2e-5" in name else 1e-3)
                model, h = legacy.fit(tr, va, cfg, seed, tok, model=LMPRegressor(config=tc)); pred = legacy.predict(model, tok, list(te.original), list(te.simplified), 96)
                ident = legacy.predict(model, tok, list(te.original), list(te.original), 96); unrel = legacy.predict(model, tok, list(te.original), list(te.simplified[::-1]), 96)
                ep = len(h["train_loss"])
            else:
                cfg["crossenc"].update(epochs=epochs, batch_size=16, grad_accum=1, max_length=96, patience=6, device="cpu", encoder_lr=1e-3, head_lr=2e-3, freeze_layers=0, pretrained=False)
                model, h = new.fit(tr, va, cfg, seed, tok, model=LMPCrossEncoder(config=tc)); pred = new.predict(model, tok, te, 96)[0]
                ident = new.predict(model, tok, te.assign(simplified=te.original), 96)[0]; unrel = new.predict(model, tok, te.assign(simplified=list(te.simplified[::-1])), 96)[0]
                ep = len(h["train_loss"])
            m = {**regression_metrics(te.lmp_score, pred), **category_metrics(te.lmp_score, pred)}
            out["runs"].append({"seed": seed, "design": name, "epochs_run": ep, "test_n": len(te), "pred_std": float(np.std(pred)), "target_std": float(te.lmp_score.std()),
                                "collapse_index": float(np.std(pred) / te.lmp_score.std()), "identical_mean": float(np.mean(ident)), "shuffled_mean": float(np.mean(unrel)),
                                "spearman": m["spearman"], "rmse": m["rmse"], "f1_macro": m["f1_macro"], "seconds": round(time.time() - t0, 1)})
            print(out["runs"][-1], flush=True)
    Path(ROOT / "outputs").mkdir(exist_ok=True); (ROOT / "outputs/collapse_diagnosis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
if __name__ == "__main__":
    if "--smoke" in sys.argv: main(seeds=(0,), epochs=3)
    else:
        sd = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [0, 1, 2]
        main(seeds=tuple(sd))
