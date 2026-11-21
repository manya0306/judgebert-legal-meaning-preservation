import json
import pandas as pd
from lmp.data.splits import document_split
from lmp.models.st_baseline import SentenceTransformerBaseline
from lmp.evaluation.metrics import regression_metrics, category_metrics, calibrate_linear
from lmp.config import load_config

cfg = load_config("configs/default.json")
df = pd.read_csv("data/synthetic/synthetic_pairs.csv")

train, val, test = document_split(df, seed=cfg["seed"])

print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
print(f"Train documents: {train.source_doc_id.nunique()}")
print(f"Test documents: {test.source_doc_id.nunique()}")

model = SentenceTransformerBaseline(
    cfg["sentence_transformer_model"], device="cpu"
)

raw_train = model.predict(train.original.tolist(), train.simplified.tolist())
raw_test = model.predict(test.original.tolist(), test.simplified.tolist())

calibrate = calibrate_linear(raw_train, train.lmp_score.values)
calibrated_test = calibrate(raw_test)

results = {
    "Sentence-Transformer cosine (raw)": {
        **regression_metrics(test.lmp_score, raw_test),
        **category_metrics(test.lmp_score, raw_test, cfg["category_thresholds"])
    },
    "Sentence-Transformer cosine (calibrated)": {
        **regression_metrics(test.lmp_score, calibrated_test),
        **category_metrics(test.lmp_score, calibrated_test, cfg["category_thresholds"])
    }
}

print("\nTEST RESULTS")
print(json.dumps(results, indent=2))
