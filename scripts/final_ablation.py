from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmp.models.sk_models import SKScorer
from lmp.models.st_baseline import SentenceTransformerBaseline
from lmp.evaluation.metrics import full_metrics, calibrate_linear

torch.set_num_threads(4)

OUT = Path("outputs/final_ablation")
OUT.mkdir(parents=True, exist_ok=True)

data = {
    s: pd.read_csv(f"data/splits/{s}.csv", keep_default_na=False)
    for s in ["train", "val", "test"]
}

train = data["train"]
y_train = train.lmp_score.to_numpy(float)

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not (
        set(data[a].group_id) & set(data[b].group_id)
    ), f"Group leakage: {a}, {b}"

print("Split sizes:", {s: len(d) for s, d in data.items()})
print("Group leakage check: PASS")

results = []
predictions = []
diagnostics = []

def evaluate(name, split, pred):
    df = data[split]
    pred = np.asarray(pred, float)

    subsets = {
        "all": np.ones(len(df), dtype=bool),
        "en": df.language.to_numpy() == "en",
        "hi": df.language.to_numpy() == "hi",
    }

    for subset, mask in subsets.items():
        if not mask.any():
            continue

        m = full_metrics(
            df.lmp_score.to_numpy(float)[mask],
            pred[mask]
        )

        results.append({
            "model": name,
            "split": split,
            "subset": subset,
            **{
                k: v for k, v in m.items()
                if not isinstance(v, (dict, list))
            }
        })

    p = df[["pair_id", "group_id", "language", "lmp_score"]].copy()
    p["model"] = name
    p["split"] = split
    p["prediction"] = pred
    predictions.append(p)

def save():
    pd.DataFrame(results).to_csv(
        OUT / "results.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / "predictions.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        OUT / "calibration_diagnostics.csv", index=False
    )

def evaluate_similarity(name, raw):
    x = raw["train"]

    slope, intercept = (
        np.polyfit(x, y_train, 1)
        if np.std(x) > 0
        else (0.0, y_train.mean())
    )

    diagnostics.append({
        "model": name,
        "slope": slope,
        "intercept": intercept,
        "train_spearman": full_metrics(
            y_train, x
        )["spearman"],
        "nonnegative_clamped": bool(slope < 0)
    })

    constrained = calibrate_linear(
        x, y_train, allow_negative=False
    )
    signed = calibrate_linear(
        x, y_train, allow_negative=True
    )

    for split in data:
        evaluate(name + "_raw", split, raw[split])
        evaluate(name + "_nonnegative", split,
                 constrained(raw[split]))
        evaluate(name + "_signed", split,
                 signed(raw[split]))

    save()

# 1. Constant baseline
print("\nRunning constant baseline...", flush=True)

for split, df in data.items():
    evaluate(
        "constant_mean", split,
        np.full(len(df), y_train.mean())
    )

# 2. Controlled feature ablations
kinds = [
    "lex_ridge",
    "rule_ridge",
    "lexrule_ridge",
    "lex_gbm",
    "rule_gbm",
    "lexrule_gbm",
]

for kind in kinds:
    print(f"Running {kind}...", flush=True)
    model = SKScorer(kind).fit(train)

    for split, df in data.items():
        evaluate(kind, split, model.predict(df))

    save()

# 3. TF-IDF calibration investigation
print("Running TF-IDF...", flush=True)

tfidf = SKScorer("tfidf_raw").fit(train)

evaluate_similarity(
    "tfidf",
    {s: tfidf.predict(df) for s, df in data.items()}
)

# 4. Sentence-Transformer calibration investigation
try:
    print("Running Sentence-Transformer...", flush=True)

    st = SentenceTransformerBaseline(device="cpu")

    raw_st = {
        s: st.predict(
            list(df.original),
            list(df.simplified)
        )
        for s, df in data.items()
    }

    evaluate_similarity("st", raw_st)

except Exception as exc:
    print("Sentence-Transformer NOT RUN:", repr(exc))
    (OUT / "st_error.txt").write_text(
        repr(exc), encoding="utf-8"
    )

# 5. Final summary
save()

r = pd.DataFrame(results)

cols = [
    "model", "spearman", "mae", "rmse",
    "accuracy", "f1_macro", "collapse_index"
]

for split in ["val", "test"]:
    print(f"\n===== {split.upper()} RESULTS =====")

    x = r[
        (r.split == split) &
        (r.subset == "all")
    ][cols]

    print(
        x.sort_values(
            "spearman", ascending=False
        ).round(4).to_string(index=False)
    )

print("\n===== CALIBRATION DIAGNOSTICS =====")
print(pd.DataFrame(diagnostics).to_string(index=False))

print("\nOutputs saved to:", OUT)
