from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from lmp.models.sk_models import SKScorer
from lmp.evaluation.metrics import full_metrics

torch.set_num_threads(4)

OUT = Path("outputs/neural_hybrid")
OUT.mkdir(parents=True, exist_ok=True)

def load(path):
    return pd.read_csv(path, keep_default_na=False)

training = {
    "original": load("data/splits/train.csv"),
    "targeted": load(
        "data/contrastive_pilot/train_targeted.csv"
    ),
    "resampled": load(
        "data/contrastive_pilot/train_resampled.csv"
    ),
}

evaluation = {
    "val": load("data/splits/val.csv"),
    "test": load("data/splits/test.csv"),
    "challenge": load(
        "data/contrastive_pilot/challenge_new.csv"
    ),
}

for name, tr in training.items():
    for split, ev in evaluation.items():
        for col in ["pair_id", "group_id", "split_group"]:
            assert not set(tr[col]) & set(ev[col])
        assert not set(tr.original) & set(ev.original)

print("Isolation checks: PASS", flush=True)

print("Loading multilingual encoder...", flush=True)

encoder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu",
    local_files_only=True
)

frames = list(training.values()) + list(evaluation.values())

texts = sorted({
    text
    for df in frames
    for col in ["original", "simplified"]
    for text in df[col]
})

print("Unique sentences:", len(texts), flush=True)

vectors = encoder.encode(
    texts,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True
)

cache = dict(zip(texts, vectors))

def semantic_features(df):
    a = np.stack([cache[t] for t in df.original])
    b = np.stack([cache[t] for t in df.simplified])

    cosine = np.sum(a * b, axis=1, keepdims=True)

    return np.hstack([
        a,
        b,
        np.abs(a - b),
        a * b,
        cosine
    ])

results = []
predictions = []

for train_name, tr in training.items():

    print("\nTraining configuration:", train_name, flush=True)

    legal = SKScorer("rule_ridge").fit(tr)
    combined = SKScorer("lexrule_ridge").fit(tr)

    raw_train = semantic_features(tr)

    pca = PCA(
        n_components=min(64, len(tr)-1),
        random_state=42
    )

    pca.fit(raw_train)

    def features(df, mode):
        semantic = pca.transform(
            semantic_features(df)
        )

        cosine = np.array([
            np.dot(cache[a], cache[b])
            for a, b in zip(
                df.original, df.simplified
            )
        ]).reshape(-1, 1)

        semantic = np.hstack([semantic, cosine])

        if mode == "semantic":
            return semantic

        if mode == "legal":
            return legal._X(df)

        if mode == "lexrule":
            return combined._X(df)

        return np.hstack([
            semantic,
            combined._X(df)
        ])

    for mode in [
        "semantic",
        "legal",
        "lexrule",
        "hybrid"
    ]:
        X = features(tr, mode)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for estimator in ["ridge", "gbm"]:

            name = f"{mode}_{estimator}"
            print("Fitting:", name, flush=True)

            if estimator == "ridge":
                model = Ridge(alpha=3.0)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=0
                )

            model.fit(X, tr.lmp_score)

            for split, ev in evaluation.items():

                Xev = scaler.transform(
                    features(ev, mode)
                )

                pred = np.clip(
                    model.predict(Xev), 1, 10
                )

                p = ev.copy()
                p["prediction"] = pred
                p["training"] = train_name
                p["model"] = name
                p["split"] = split

                predictions.append(p)

                for subset, d in {
                    "all": p,
                    "en": p[p.language == "en"],
                    "hi": p[p.language == "hi"]
                }.items():

                    m = full_metrics(
                        d.lmp_score,
                        d.prediction
                    )

                    results.append({
                        "training": train_name,
                        "model": name,
                        "split": split,
                        "subset": subset,
                        **{
                            k: v for k, v in m.items()
                            if not isinstance(v, (dict, list))
                        }
                    })

            pd.DataFrame(results).to_csv(
                OUT / "results.csv",
                index=False
            )

            pd.concat(
                predictions, ignore_index=True
            ).to_csv(
                OUT / "predictions.csv",
                index=False
            )

r = pd.DataFrame(results)

for split in ["val", "test", "challenge"]:
    print(f"\n===== {split.upper()} =====")

    print(r[
        (r.split == split) &
        (r.subset == "all")
    ][[
        "training", "model",
        "spearman", "mae",
        "accuracy", "f1_macro"
    ]].round(4).to_string(index=False))

print("\nSaved to:", OUT)
