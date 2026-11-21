"""Training / inference for the LMP regressor: seeds, early stopping, checkpoints, history."""
from __future__ import annotations
import json, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
from .judgebert_regressor import LMPRegressor, to_unit, from_unit
from ..config import set_seed
from ..logging_utils import get_logger

log = get_logger("lmp.train")

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        enc = tokenizer(list(df["original"]), list(df["simplified"]), truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        self.enc = enc
        y = df["lmp_score"].values if "lmp_score" in df.columns else np.full(len(df), 1.0)
        self.y = torch.tensor(to_unit(y.astype("float32")))
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.y[i]
        return item

def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)

def _predict_unit(model, loader, device):
    model.eval(); out = []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            out.append(model(**b).cpu())
    return torch.cat(out).numpy()

def fit(train_df, val_df, cfg: dict, seed: int, tokenizer, model=None, ckpt_dir: str | Path | None = None):
    """Returns (best_model, history dict). Early stopping on validation RMSE (1-10 scale)."""
    set_seed(seed)
    rc = cfg["regressor"]; device = pick_device(rc.get("device", "auto"))
    model = (model or LMPRegressor(rc["backbone"], rc["dropout"])).to(device)
    tr = DataLoader(PairDataset(train_df, tokenizer, rc["max_length"]), batch_size=rc["batch_size"], shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    va = DataLoader(PairDataset(val_df, tokenizer, rc["max_length"]), batch_size=rc["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=rc["learning_rate"], weight_decay=rc["weight_decay"])
    total = max(1, rc["epochs"] * len(tr)); warm = int(rc["warmup_ratio"] * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    loss_fn = torch.nn.HuberLoss(delta=0.2) if rc["loss"] == "huber" else torch.nn.MSELoss()
    yv = val_df["lmp_score"].values
    hist = {"train_loss": [], "val_loss": [], "val_rmse": [], "seed": seed, "backbone": rc["backbone"], "config": rc}
    best, best_state, bad = float("inf"), None, 0
    for ep in range(rc["epochs"]):
        model.train(); losses = []
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            y = b.pop("labels")
            loss = loss_fn(model(**b), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); losses.append(loss.item())
        pv = _predict_unit(model, va, device)
        vl = float(loss_fn(torch.tensor(pv), torch.tensor(to_unit(yv.astype("float32")))))
        rmse = float(np.sqrt(np.mean((from_unit(pv) - yv) ** 2)))
        hist["train_loss"].append(float(np.mean(losses))); hist["val_loss"].append(vl); hist["val_rmse"].append(rmse)
        log.info("epoch %d train_loss=%.4f val_loss=%.4f val_rmse=%.3f", ep + 1, hist["train_loss"][-1], vl, rmse)
        if rmse < best - 1e-4:
            best, bad, best_state = rmse, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= rc["patience"]:
                log.info("early stopping at epoch %d", ep + 1); break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_rmse"] = best
    if ckpt_dir:
        p = Path(ckpt_dir); p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), p / "model.pt")
        (p / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
        tokenizer.save_pretrained(p)
        model.encoder.config.save_pretrained(p)   # lets load_checkpoint rebuild the encoder without re-downloading weights
        (p / "meta.json").write_text(json.dumps({"backbone": rc["backbone"], "dropout": rc["dropout"], "max_length": rc["max_length"], "seed": seed}), encoding="utf-8")
    return model, hist

def predict(model, tokenizer, originals, simplified, max_length=128, batch_size=16, device=None) -> np.ndarray:
    device = device or next(model.parameters()).device
    df = pd.DataFrame({"original": originals, "simplified": simplified})
    ds = PairDataset(df, tokenizer, max_length)
    return from_unit(_predict_unit(model, DataLoader(ds, batch_size=batch_size), device))

def load_checkpoint(ckpt_dir: str | Path, device="cpu"):
    from transformers import AutoTokenizer, AutoConfig
    p = Path(ckpt_dir); meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(p)
    model = LMPRegressor(meta["backbone"], meta["dropout"], config=AutoConfig.from_pretrained(p))
    model.load_state_dict(torch.load(p / "model.pt", map_location=device)); model.to(device).eval()
    return model, tok, meta
