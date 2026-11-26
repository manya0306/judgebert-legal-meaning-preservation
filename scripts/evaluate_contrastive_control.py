from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmp.models.sk_models import SKScorer
from lmp.evaluation.metrics import full_metrics
from lmp.preprocess import tokenize

OUT = Path("outputs/contrastive_control")
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    return pd.read_csv(path, keep_default_na=False)

training = {
    "original": load("data/splits/train.csv"),
    "targeted": load(
        "data/contrastive_pilot/train_targeted.csv"
    ),
}

training["resampled"] = load("data/contrastive_pilot/train_resampled.csv")

evaluation = {
    "val": load("data/splits/val.csv"),
    "test": load("data/splits/test.csv"),
    "challenge": load(
        "data/contrastive_pilot/challenge_new.csv"
    ),
}

for name, tr in training.items():
    for split, ev in evaluation.items():
        for col in ["group_id", "split_group", "pair_id"]:
            assert not (
                set(tr[col]) & set(ev[col])
            ), f"Leakage: {name}/{split}/{col}"

        assert not (
            set(tr.original) & set(ev.original)
        ), f"Original overlap: {name}/{split}"

print("Isolation checks: PASS")

def overlap(a, b):
    a, b = set(tokenize(a)), set(tokenize(b))
    return len(a & b) / max(1, len(a | b))

def contrastive_metrics(df):
    rows = []

    for _, g in df.groupby(["language", "original"]):
        good = g[g.category == "Preserved"]
        bad = g[g.category == "Not Preserved"]

        for a in good.itertuples():
            for b in bad.itertuples():
                margin = a.prediction - b.prediction

                rows.append({
                    "language": a.language,
                    "correct": float(margin > 0),
                    "tie": float(margin == 0),
                    "margin": margin,
                    "hard_case": b.overlap > a.overlap,
                })

    r = pd.DataFrame(rows)

    if r.empty:
        return {}

    hard = r[r.hard_case]

    return {
        "comparisons": len(r),
        "ranking_accuracy": r.correct.mean(),
        "tie_rate": r.tie.mean(),
        "mean_margin": r.margin.mean(),
        "hard_comparisons": len(hard),
        "hard_ranking_accuracy": (
            hard.correct.mean() if len(hard) else np.nan
        ),
    }

models = [
    "tfidf_raw",
    "tfidf_cal",
    "lex_ridge",
    "rule_ridge",
    "lexrule_ridge",
    "lex_gbm",
    "rule_gbm",
    "lexrule_gbm",
]

results = []
ranking = []
predictions = []

for train_name, tr in training.items():
    for kind in models:
        print(
            f"Training {train_name}: {kind}",
            flush=True
        )

        model = SKScorer(kind).fit(tr)

        for split, ev in evaluation.items():
            pred = model.predict(ev)

            p = ev.copy()
            p["prediction"] = pred
            p["training"] = train_name
            p["model"] = kind
            p["split"] = split

            p["overlap"] = [
                overlap(a, b)
                for a, b in zip(p.original, p.simplified)
            ]

            predictions.append(p)

            subsets = {
                "all": p,
                "en": p[p.language == "en"],
                "hi": p[p.language == "hi"],
            }

            for subset, d in subsets.items():
                if d.empty:
                    continue

                m = full_metrics(
                    d.lmp_score, d.prediction
                )

                results.append({
                    "training": train_name,
                    "model": kind,
                    "split": split,
                    "subset": subset,
                    **{
                        k: v for k, v in m.items()
                        if not isinstance(v, (dict, list))
                    }
                })

                ranking.append({
                    "training": train_name,
                    "model": kind,
                    "split": split,
                    "subset": subset,
                    **contrastive_metrics(d)
                })

r = pd.DataFrame(results)
c = pd.DataFrame(ranking)

r.to_csv(OUT / "results.csv", index=False)
c.to_csv(OUT / "contrastive.csv", index=False)

pd.concat(predictions, ignore_index=True).to_csv(
    OUT / "predictions.csv", index=False
)

for split in ["val", "test", "challenge"]:
    print(f"\n===== {split.upper()} =====")

    print(r[
        (r.split == split) &
        (r.subset == "all")
    ][[
        "training", "model", "spearman",
        "mae", "accuracy", "f1_macro"
    ]].round(4).to_string(index=False))

print("\n===== CHALLENGE CONTRASTIVE RANKING =====")

print(c[
    (c.split == "challenge") &
    (c.subset == "all")
][[
    "training", "model", "comparisons",
    "ranking_accuracy", "mean_margin",
    "hard_comparisons", "hard_ranking_accuracy"
]].round(4).to_string(index=False))

print("\nSaved to:", OUT)

