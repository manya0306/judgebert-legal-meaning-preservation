#!/usr/bin/env python
"""Human-validation analysis: ingest completed workbooks, validate, consolidate, analyse, plot, report.

    python scripts/analyze_human_validation.py                        # statistics only (no models)
    python scripts/analyze_human_validation.py --run-models lexical_similarity,lexrule_gbm --xlmr-checkpoint checkpoints/crossenc
    python scripts/analyze_human_validation.py --predictions my_predictions.csv

Never modifies input workbooks, retrains models, downloads weights or uses human labels for tuning."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lmp.human_validation import figures, ingest, model_eval, report, stats  # noqa: E402


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "openpyxl", "torch", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main(argv=None, spec: ingest.DesignSpec | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "human_validation_v2" / "completed")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "human_validation_analysis")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="project root containing data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output directory")
    ap.add_argument("--run-models", default="", help="comma list: " + ",".join(model_eval.CANONICAL))
    ap.add_argument("--predictions", action="append", default=[], help="CSV with pair_id, model_name, score[, checkpoint_id, training_splits]")
    ap.add_argument("--xlmr-checkpoint", type=Path)
    ap.add_argument("--xlmr-provenance", type=Path, help='JSON {"training_splits": ["train"], "checkpoint_id": "..."}')
    a = ap.parse_args(argv)
    spec = spec or ingest.DesignSpec()
    out = a.output_dir
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        print(f"ERROR: {out} is not empty. Choose another --output-dir or pass --overwrite (previous runs are never overwritten by default).", file=sys.stderr)
        return 3
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    L = ingest.load_and_validate(a.input_dir, spec, a.data_root)
    rep = L.report
    (out / "validation_report.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (out / "validation_report.md").write_text(rep.to_markdown(), encoding="utf-8")
    if L.corrections is not None and len(L.corrections):
        L.corrections.to_csv(out / "correction_log.csv", index=False, encoding="utf-8-sig")
    meta = {"status": "STOPPED: critical validation errors" if not L.ok else "COMPLETED", "started_utc": started, "seed": a.seed, "n_boot": a.n_boot,
            "input_dir": str(a.input_dir), "input_sha256": L.hashes, "args": {k: str(v) for k, v in vars(a).items()}, "versions": versions(), "git_commit": git_commit(),
            "design_spec": {"annotators": dict(spec.annotators), "n_en": spec.n_en, "n_hi": spec.n_hi}}
    if not L.ok:
        for c in rep.critical_failures:
            print("CRITICAL:", c.id, "-", c.message, file=sys.stderr)
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Validation FAILED ({len(rep.critical_failures)} critical). Dependent analyses were not run. See {out / 'validation_report.md'}", file=sys.stderr)
        return 2
    cons, ind = L.consensus, L.individual
    ind.to_csv(out / "individual_annotations.csv", index=False, encoding="utf-8-sig")
    cons.to_csv(out / "pair_consensus.csv", index=False, encoding="utf-8-sig")
    m = out / "metrics"
    t = {}
    t["student_agreement"] = stats.student_agreement(cons, n_boot=a.n_boot, seed=a.seed)
    t["expert"], conf_e = stats.expert_comparison(cons, a.n_boot, a.seed)
    t["reference"], conf_r = stats.reference_comparison(cons, ind, a.n_boot, a.seed)
    t["subgroup"] = stats.subgroup_analysis(cons)
    t["semantic"] = stats.semantic_analysis(cons, ind)
    t["qualitative"] = stats.qualitative_tables(cons, ind)
    for k, fn in (("student_agreement", "student_agreement"), ("expert", "expert_comparison"), ("reference", "reference_label_comparison"), ("subgroup", "subgroup_analysis"),
                  ("semantic", "semantic_error_analysis"), ("qualitative", "qualitative_disagreements")):
        t[k].to_csv(m / f"{fn}.csv", index=False, encoding="utf-8-sig")
    pd.concat([conf_e, conf_r], ignore_index=True).to_csv(m / "confusion_matrices.csv", index=False, encoding="utf-8-sig")
    # optional model evaluation (opt-in; the statistics above never depend on it)
    requested = [x.strip() for x in a.run_models.split(",") if x.strip()]
    preds, not_run, issues = [], {}, []
    for p in a.predictions:
        pp, iss = model_eval.import_predictions(p, cons)
        preds += pp
        issues += iss
    if requested:
        lp, nr = model_eval.run_live(requested, cons, a.data_root, a.xlmr_checkpoint, a.xlmr_provenance)
        preds += lp
        not_run.update(nr)
    if not requested and not a.predictions:
        not_run = model_eval.not_requested()
    else:
        for k in model_eval.CANONICAL:
            if k not in not_run and not any(p.name == k for p in preds):
                not_run[k] = "NOT RUN: not requested in this run" if k not in requested else "NOT RUN: unavailable"
    t["model"] = model_eval.evaluate_models(preds, not_run, cons, a.data_root)
    t["model"].to_csv(m / "model_comparison.csv", index=False, encoding="utf-8-sig")
    if issues:
        (m / "model_import_issues.txt").write_text("\n".join(issues), encoding="utf-8")
    fig_status = figures.make_figures(cons, ind, t, out / "figures", t["model"])
    (out / "human_validation_report.md").write_text(report.build_report(cons, ind, t, rep, fig_status, meta, spec), encoding="utf-8")
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    meta["figures"] = fig_status
    meta["input_sha256_after_run"] = {p.name: ingest.sha256(p) for p in (a.input_dir / n for n in (ingest.STUDENT_FILE, ingest.LAWYER_FILE, ingest.KEY_FILE))}
    meta["inputs_unchanged"] = meta["input_sha256_after_run"] == L.hashes
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done. Outputs in {out}\n  individual judgments: {len(ind)} | pairs: {len(cons)} | validation: {rep.to_dict()['overall']} | inputs unchanged: {meta['inputs_unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
