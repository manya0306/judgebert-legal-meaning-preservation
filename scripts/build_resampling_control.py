from pathlib import Path
import pandas as pd

base = pd.read_csv(
    "data/splits/train.csv", keep_default_na=False
)

new = pd.read_csv(
    "data/contrastive_pilot/train_new.csv",
    keep_default_na=False
)

rows = []

for (lang, cat), group in new.groupby(
    ["language", "category"]
):
    pool = base[
        (base.language == lang) &
        (base.category == cat)
    ]

    sampled = pool.sample(
        n=len(group),
        replace=True,
        random_state=42
    ).copy()

    sampled["parent_pair_id"] = sampled["pair_id"]
    rows.append(sampled)

extra = pd.concat(rows, ignore_index=True)

extra["pair_id"] = [
    f"RESAMPLE-{i:04d}"
    for i in range(len(extra))
]

extra["generation_strategy"] = "random_oversampling"
extra["annotation_status"] = (
    "Resampled existing training example; no new annotation"
)

control = pd.concat(
    [base, extra], ignore_index=True
)

assert len(extra) == len(new)
assert control.pair_id.is_unique

assert extra.groupby(
    ["language", "category"]
).size().equals(
    new.groupby(["language", "category"]).size()
)

out = Path("data/contrastive_pilot")
control.to_csv(
    out / "train_resampled.csv", index=False
)

print("Original training:", len(base))
print("Targeted training:", len(base) + len(new))
print("Resampled training:", len(control))
print("Language/category matching: PASS")
print(extra.groupby(["language", "category"]).size())
