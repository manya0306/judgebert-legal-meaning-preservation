import json
from lmp.config import load_config, ROOT
from lmp.evaluation.experiments import run_all

def test_end_to_end_experiments_without_transformers(tmp_path):
    cfg = load_config(); cfg["data_path"] = "data/synthetic/synthetic_pairs.csv"      # the legacy pipeline targets the original 48-pair set
    R = run_all(cfg, tmp_path, use_transformers=False)
    for f in ("results.json", "results.md", "exp5_error_analysis.md", "figures/exp1_scatter_test.png"): assert (tmp_path / f).exists()
    assert "NOT RUN" in R["meta"]["model_status"]["JUDGEBERT-inspired regressor"]      # honest reporting of what was not run
    assert R["meta"]["synthetic_only"] is True
    assert R["exp4_sanity"]["models"]["TF-IDF cosine (raw)"]["identical_mean"] > 9.99
    md = (tmp_path / "exp5_error_analysis.md").read_text(encoding="utf-8")
    assert md.count("\n## ") >= 15                                                        # >= 15 qualitative cases
    assert "SYNTHETIC" in (tmp_path / "results.md").read_text(encoding="utf-8")
