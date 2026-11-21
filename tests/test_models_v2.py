import json, numpy as np, pandas as pd, pytest, torch
from lmp.config import load_config
from lmp.data.io import load_pairs
from lmp.data.splits import grouped_split
from lmp.models.tiny import build_tokenizer, tiny_config
from lmp.models.crossenc import LMPCrossEncoder, to_z, from_z
from lmp.models import train_crossenc as T
from lmp.models.sk_models import SKScorer, SK_KINDS
from lmp.evaluation.metrics import full_metrics

@pytest.fixture(scope="module")
def split():
    df = load_pairs("data/expanded/pairs_consolidated.csv"); return grouped_split(df, seed=0)

@pytest.fixture(scope="module")
def tiny(split):
    tr = split[0]; tok = build_tokenizer(list(tr.original) + list(tr.simplified)); return tok, tiny_config(len(tok))

def test_tokenizer_handles_hindi_pairs_and_token_types(tiny, split):
    tok, _ = tiny; enc = tok(["किरायेदार किराया देगा।"], ["किरायेदार किराया नहीं देगा।"], return_tensors="pt")
    assert "token_type_ids" in enc and enc["token_type_ids"].max().item() == 1 and enc["token_type_ids"][0, 0].item() == 0

def test_forward_shapes_and_unbounded_linear_head(tiny, split):
    tok, cfg = tiny; m = LMPCrossEncoder(config=cfg); ds = T.PairDS(split[2].head(5), tok, 64)
    b = {k: torch.stack([ds[i][k] for i in range(5)]) for k in ("input_ids", "attention_mask", "token_type_ids", "lang")}
    z, lg = m(b["input_ids"], b["attention_mask"], b["token_type_ids"], b["lang"]); assert z.shape == (5,) and lg.shape == (5, 3) and torch.isfinite(z).all()
    m.init_regression_bias(0.3); assert abs(m.reg_head[-1].bias.item() - 0.3) < 1e-6

def test_score_scaling_roundtrip(): assert from_z(to_z(7.25)) == pytest.approx(7.25) and to_z(1) == pytest.approx(-1) and to_z(10) == pytest.approx(1)

def test_fit_predict_checkpoint_roundtrip(tmp_path, tiny, split):
    tok, cfg = tiny; c = load_config(); c["crossenc"].update(epochs=2, batch_size=16, grad_accum=1, max_length=64, patience=2, device="cpu", encoder_lr=1e-3, head_lr=2e-3, freeze_layers=0, pretrained=False)
    tr, va, te = split; model, hist = T.fit(tr.head(64), va.head(32), c, 0, tok, model=LMPCrossEncoder(config=cfg), ckpt_dir=tmp_path)
    assert len(hist["train_loss"]) >= 1 and (tmp_path / "model.pt").exists() and (tmp_path / "history.json").exists() and "collapse_index" in hist
    s, p, cats = T.predict(model, tok, te.head(10), 64); assert s.min() >= 1 and s.max() <= 10 and np.allclose(p.sum(1), 1, atol=1e-5) and set(cats) <= {"Not Preserved", "Partially Preserved", "Preserved"}
    m2, tok2, meta = T.load_checkpoint(tmp_path); s2, _, _ = T.predict(m2, tok2, te.head(10), 64); assert np.allclose(s, s2, atol=1e-4) and meta["architecture"] == "LMPCrossEncoder"

def test_legacy_checkpoint_is_refused(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"architecture": "LMPRegressor"}))
    with pytest.raises(ValueError, match="legacy"): T.load_checkpoint(tmp_path)

def test_collapse_index_detects_constant_output():
    y = np.arange(1, 11, dtype=float); assert T.collapse_index(np.full(10, 5.0), y) == 0 and T.collapse_index(y, y) == pytest.approx(1)

@pytest.mark.parametrize("kind", SK_KINDS)
def test_sk_scorers_fit_and_range(kind, split):
    tr, _, te = split; p = SKScorer(kind).fit(tr).predict(te.head(20)); assert len(p) == 20 and np.all((p >= 1 - 1e-9) & (p <= 10 + 1e-9)) and np.isfinite(p).all()

def test_full_metrics_perfect_and_constant():
    y = np.array([1, 3, 5, 8, 9.0]); m = full_metrics(y, y); assert m["rmse"] == 0 and m["accuracy"] == 1 and m["f1_macro"] == 1
    c = full_metrics(y, np.full(5, y.mean())); assert c["collapse_index"] == 0 and np.isnan(c["spearman"])

def test_missing_pretrained_weights_script_fails_loudly():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/train_crossenc.py", "--local-model-path", "/no/such/model"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "No model was trained" in (r.stderr + r.stdout)


def test_sentence_transformer_plumbing_with_stub_and_honest_failure(split, monkeypatch):
    from lmp.evaluation import framework as F
    tr, _, te = split
    msg = F.register_sentence_transformer(tr.head(4), "/definitely/not/a/model")
    assert msg.startswith("NOT RUN") and not F.ST_SIM                       # unreachable weights are reported, never faked
    F.ST_SIM.update({i: 0.5 for i in list(tr.pair_id) + list(te.pair_id)})   # STUB values: tests only the calibration plumbing, not any real model
    try:
        p = F._fit_predict("st_cal", tr, te); assert len(p) == len(te) and np.isfinite(p).all()
    finally: F.ST_SIM.clear()
