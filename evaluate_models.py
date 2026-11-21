import json
import torch
import pandas as pd

from lmp.config import load_config
from lmp.data.io import load_pairs
from lmp.data.splits import document_split
from lmp.models import trainer
from lmp.models.st_baseline import SentenceTransformerBaseline
from lmp.evaluation.metrics import regression_metrics, category_metrics, calibrate_linear

torch.set_num_threads(4)

cfg = load_config("configs/default.json")
df = load_pairs("data/synthetic/synthetic_pairs.csv")
train, val, test = document_split(df, seed=cfg["seed"])

print("Reloading saved XLM-R checkpoint...")
model, tok, meta = trainer.load_checkpoint(
    "checkpoints/judgebert_smoke", device="cpu"
)

print("Generating XLM-R predictions...")
xlmr_train = trainer.predict(
    model, tok, train.original.tolist(), train.simplified.tolist(),
    max_length=meta["max_length"], batch_size=2, device="cpu"
)
xlmr_test = trainer.predict(
    model, tok, test.original.tolist(), test.simplified.tolist(),
    max_length=meta["max_length"], batch_size=2, device="cpu"
)

xlmr_cal = calibrate_linear(xlmr_train, train.lmp_score.values)
xlmr_cal_test = xlmr_cal(xlmr_test)

print("Loading Sentence-Transformer baseline...")
st = SentenceTransformerBaseline(
    cfg["sentence_transformer_model"], device="cpu"
)
st_train = st.predict(train.original.tolist(), train.simplified.tolist())
st_test = st.predict(test.original.tolist(), test.simplified.tolist())
st_cal = calibrate_linear(st_train, train.lmp_score.values)
st_cal_test = st_cal(st_test)

results = {}
for name, predictions in [
    ("Sentence-Transformer raw", st_test),
    ("Sentence-Transformer calibrated", st_cal_test),
    ("XLM-R smoke model raw", xlmr_test),
    ("XLM-R smoke model calibrated", xlmr_cal_test)
]:
    results[name] = {
        **regression_metrics(test.lmp_score, predictions),
        **category_metrics(test.lmp_score, predictions, cfg["category_thresholds"])
    }

print("\nHELD-OUT TEST COMPARISON")
print(json.dumps(results, indent=2))

output = test[["pair_id", "language", "legal_domain", "lmp_score"]].copy()
output["st_raw"] = st_test
output["st_calibrated"] = st_cal_test
output["xlmr_raw"] = xlmr_test
output["xlmr_calibrated"] = xlmr_cal_test

output.to_csv("outputs/model_comparison_test.csv", index=False)
print("\nPredictions saved to outputs/model_comparison_test.csv")
