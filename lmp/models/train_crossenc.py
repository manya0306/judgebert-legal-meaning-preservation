"""Trainer for LMPCrossEncoder: seeds, differential learning rates, gradient accumulation, collapse monitoring, checkpointing by a
rank-based validation criterion (a constant predictor cannot win), history/metadata saving, prediction + category decoding."""
from __future__ import annotations
import copy, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from .crossenc import LMPCrossEncoder, to_z, from_z, class_names
from ..config import set_seed, score_to_category
from ..logging_utils import get_logger

log = get_logger("lmp.crossenc")
CAT_IDX = {c: i for i, c in enumerate(class_names())}
LANG_IDX = {"en": 0, "hi": 1}


class PairDS(Dataset):
    def __init__(self, df, tok, max_length):
        enc = tok(list(df.original), list(df.simplified), truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        self.enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        has = "lmp_score" in df.columns
        y = df.lmp_score.values.astype("float32") if has else np.full(len(df), 5.5, "float32")
        self.z = torch.tensor(to_z(y)); self.c = torch.tensor([CAT_IDX[score_to_category(s)] for s in y])
        self.lang = torch.tensor([LANG_IDX.get(l, 0) for l in df.language])
    def __len__(self): return len(self.z)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}; d.update(z=self.z[i], c=self.c[i], lang=self.lang[i]); return d


def _batch_inputs(b, device):
    kw = {k: b[k].to(device) for k in ("input_ids", "attention_mask") if k in b}
    if "token_type_ids" in b: kw["token_type_ids"] = b["token_type_ids"].to(device)
    kw["lang_ids"] = b["lang"].to(device)
    return kw


@torch.no_grad()
def predict_raw(model, loader, device):
    model.eval(); zs, ls = [], []
    for b in loader:
        z, lg = model(**_batch_inputs(b, device)); zs.append(z.cpu()); ls.append(lg.cpu())
    return torch.cat(zs).numpy(), torch.cat(ls).numpy()


def collapse_index(pred, y) -> float:
    """std(pred)/std(target). ~0 = constant output (collapse); ~1 = healthy spread."""
    return float(np.std(pred) / max(1e-8, np.std(y)))


def fit(train_df, val_df, cfg: dict, seed: int, tok, model=None, ckpt_dir=None):
    set_seed(seed); rc = cfg["crossenc"]; device = torch.device("cuda" if (rc.get("device", "auto") == "auto" and torch.cuda.is_available()) else ("cpu" if rc.get("device", "auto") in ("auto", "cpu") else rc["device"]))
    torch.set_num_threads(int(rc.get("threads", 1)))
    model = (model or LMPCrossEncoder(rc["backbone"], rc["dropout"], use_language_embedding=rc.get("language_embedding", False), freeze_layers=rc.get("freeze_layers", 0))).to(device)
    model.init_regression_bias(float(np.mean(to_z(train_df.lmp_score.values))))
    tr = DataLoader(PairDS(train_df, tok, rc["max_length"]), batch_size=rc["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed))
    va = DataLoader(PairDS(val_df, tok, rc["max_length"]), batch_size=rc["batch_size"])
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": rc["encoder_lr"]}, {"params": head_params, "lr": rc["head_lr"]}], weight_decay=rc["weight_decay"])
    accum = max(1, rc.get("grad_accum", 1)); steps = max(1, rc["epochs"] * int(np.ceil(len(tr) / accum))); warm = int(rc["warmup_ratio"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    counts = np.bincount(PairDS(train_df, tok, 8).c.numpy(), minlength=3).astype(float); w = torch.tensor(counts.sum() / np.maximum(counts, 1) / 3, dtype=torch.float32).to(device)
    huber, ce, lam = torch.nn.HuberLoss(delta=rc.get("huber_delta", 0.5)), torch.nn.CrossEntropyLoss(weight=w), rc.get("cls_weight", 1.0)
    yv, hist = val_df.lmp_score.values, {"epoch": [], "train_loss": [], "val_rmse": [], "val_spearman": [], "val_macro_f1": [], "collapse_index": [], "seed": seed, "config": rc}
    best, best_state, bad = -1e9, None, 0
    best_state_path = None

    if ckpt_dir:
        best_dir = Path(ckpt_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        best_state_path = best_dir / "best_state.tmp.pt"
    for ep in range(rc["epochs"]):
        model.train(); losses = []; opt.zero_grad()
        for i, b in enumerate(tr):
            z, lg = model(**_batch_inputs(b, device))
            group_size = min(accum, len(tr) - (i // accum) * accum)
            loss = (huber(z, b["z"].to(device)) + lam * ce(lg, b["c"].to(device))) / group_size
            loss.backward(); losses.append(loss.item() * accum)
            if (i + 1) % accum == 0 or i + 1 == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
        zs, lg = predict_raw(model, va, device); pred = np.clip(from_z(zs), 1, 10)
        rmse = float(np.sqrt(np.mean((pred - yv) ** 2))); rho = float(spearmanr(pred, yv)[0]) if np.std(pred) > 1e-9 else 0.0
        f1 = float(f1_score([CAT_IDX[score_to_category(s)] for s in yv], lg.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)); ci = collapse_index(pred, yv)
        for k, v in zip(("epoch", "train_loss", "val_rmse", "val_spearman", "val_macro_f1", "collapse_index"), (ep + 1, float(np.mean(losses)), rmse, rho, f1, ci)): hist[k].append(v)
        log.info("ep %d loss=%.3f rmse=%.2f rho=%.2f f1=%.2f collapse=%.2f", ep + 1, hist["train_loss"][-1], rmse, rho, f1, ci)
        crit = rho + f1 - 0.001 * rmse - (1.0 if ci < 0.15 else 0.0)          # rank/F1-based; collapsed epochs are penalised
        if crit > best + 1e-4:
            best, bad = crit, 0
            hist["best_epoch"] = ep + 1

            if best_state_path is not None:
                torch.save(model.state_dict(), best_state_path)
                best_state = True
            else:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= rc["patience"]:
                log.info("early stop at epoch %d", ep + 1)
                break
    if best_state_path is not None and best_state:
        state = torch.load(best_state_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        del state
        best_state_path.unlink(missing_ok=True)
    elif best_state is not None:
        model.load_state_dict(best_state)
    hist["selection_criterion"] = "val spearman + val macro-F1 - 0.001*RMSE - collapse penalty"
    if ckpt_dir:
        p = Path(ckpt_dir); p.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), p / "model.pt")
        (p / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8"); tok.save_pretrained(p); model.encoder.config.save_pretrained(p)
        (p / "meta.json").write_text(json.dumps({"architecture": "LMPCrossEncoder", "backbone": rc["backbone"], "dropout": rc["dropout"], "max_length": rc["max_length"], "seed": seed,
                                                "language_embedding": rc.get("language_embedding", False), "pretrained_weights_used": rc.get("pretrained", True)}), encoding="utf-8")
    return model, hist


def predict(model, tok, df, max_length=128, batch_size=16):
    """Returns (scores 1-10 np.ndarray, class_probs np.ndarray [n,3], categories list)."""
    dev = next(model.parameters()).device
    zs, lg = predict_raw(model, DataLoader(PairDS(df, tok, max_length), batch_size=batch_size), dev)
    probs = torch.softmax(torch.tensor(lg), -1).numpy()
    return np.clip(from_z(zs), 1, 10), probs, [class_names()[i] for i in probs.argmax(1)]


def load_checkpoint(ckpt_dir, device="cpu"):
    from transformers import AutoTokenizer, AutoConfig
    p = Path(ckpt_dir); meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    if meta.get("architecture") != "LMPCrossEncoder":
        raise ValueError(f"{p} is a legacy checkpoint ({meta.get('architecture', 'LMPRegressor')}); it is incompatible - retrain with scripts/train_crossenc.py")
    tok = AutoTokenizer.from_pretrained(p)
    model = LMPCrossEncoder(meta["backbone"], meta["dropout"], config=AutoConfig.from_pretrained(p), use_language_embedding=meta.get("language_embedding", False))
    model.load_state_dict(torch.load(p / "model.pt", map_location=device)); return model.to(device).eval(), tok, meta
