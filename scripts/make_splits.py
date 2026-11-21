"""Write a reproducible grouped train/val/test split and its report: python scripts/make_splits.py [--seed 42]
Outputs data/splits/{train,val,test}.csv and outputs/split_report.json. Test rows must never be used for tuning."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lmp.config import load_config
from lmp.data.io import load_pairs, save_pairs
from lmp.data.splits import grouped_split, split_report
ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args(); cfg = load_config()
df = load_pairs(ROOT / (a.data or cfg["data_path"])); tr, va, te = grouped_split(df, seed=a.seed); out = ROOT / "data/splits"; out.mkdir(exist_ok=True)
for n, d in (("train", tr), ("val", va), ("test", te)): d.to_csv(out / f"{n}.csv", index=False, encoding="utf-8-sig")
rep = split_report(tr, va, te); (ROOT / "outputs").mkdir(exist_ok=True); (ROOT / "outputs/split_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps(rep, indent=1, ensure_ascii=False)[:1500])
