"""Dataset inspection (from project root): python scripts/inspect_dataset.py [--data data/expanded/pairs_consolidated.csv]
Prints counts, distributions, QA flag summary, shortcut-risk statistics and writes outputs/dataset_stats.json + figures/dataset_*.png"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lmp.data.io import load_pairs
from lmp.data.qa import run_qa, distribution_report, lexical_cue_reliance
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/expanded/pairs_consolidated.csv"); a = ap.parse_args()
df = load_pairs(ROOT / a.data); rep = distribution_report(df); rep["lexical_cue_reliance"] = lexical_cue_reliance(df); fl = run_qa(df)
rep["qa_flags"] = fl.groupby(["severity", "check"]).size().to_dict().__repr__(); (ROOT / "outputs/figures").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/dataset_stats.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); df.groupby(["language", "category"]).size().unstack().plot.bar(ax=ax[0]); ax[0].set_title("language x category")
df.lmp_score.hist(bins=19, ax=ax[1]); ax[1].set_title("score distribution"); df.legal_domain.value_counts().plot.barh(ax=ax[2]); ax[2].set_title("domain")
plt.tight_layout(); plt.savefig(ROOT / "outputs/figures/dataset_overview.png", dpi=120)
print(json.dumps({k: rep[k] for k in ("n", "by_language", "by_category", "by_tier", "n_groups", "lexical_cue_reliance")}, indent=1, ensure_ascii=False))
