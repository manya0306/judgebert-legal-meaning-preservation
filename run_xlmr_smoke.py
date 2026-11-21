# DEPRECATED legacy smoke script (trains the collapse-prone LMPRegressor design). Use scripts/train_crossenc.py instead.
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lmp.config import load_config
from lmp.data.io import load_pairs
from lmp.data.splits import document_split
from lmp.data.augment import augment_with_sanity_pairs
from lmp.models import trainer

torch.set_num_threads(4)

cfg = load_config("configs/default.json")

cfg["regressor"]["batch_size"] = 2
cfg["regressor"]["epochs"] = 1
cfg["regressor"]["patience"] = 1
cfg["regressor"]["device"] = "cpu"

df = load_pairs("data/synthetic/synthetic_pairs.csv")
train, val, test = document_split(df, seed=cfg["seed"])

train = augment_with_sanity_pairs(train, cfg["seed"])

print("Training pairs:", len(train))
print("Validation pairs:", len(val))
print("Device: CPU")
print("Epochs: 1")
print("Batch size: 2")
print("Loading tokenizer...")

tok = AutoTokenizer.from_pretrained(cfg["regressor"]["backbone"])

print("Starting XLM-R training...")

checkpoint = Path("checkpoints/judgebert_smoke")

model, history = trainer.fit(
    train, val, cfg, cfg["seed"], tok,
    ckpt_dir=checkpoint
)

print("\nTRAINING COMPLETED")
print("History:", history)
print("Checkpoint directory:", checkpoint.resolve())
print("Checkpoint files:", [str(p) for p in checkpoint.rglob("*") if p.is_file()])
