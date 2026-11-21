"""Exercises the real training code path with a TINY RANDOMLY-INITIALISED BERT and a character-level vocabulary, fully offline.
This verifies the code (shapes, loss, early stopping, checkpoints, prediction range). It says NOTHING about model quality."""
import json
import pytest
torch = pytest.importorskip("torch"); pytest.importorskip("transformers")
from transformers import BertConfig, BertTokenizerFast
from lmp.config import load_config, ROOT
from lmp.data.io import load_pairs
from lmp.data.splits import document_split
from lmp.models.judgebert_regressor import LMPRegressor
from lmp.models import trainer

def _tiny(tmp_path, df):
    chars = sorted({c for t in list(df.original) + list(df.simplified) for c in t})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", " "] + [c for c in chars if c != " "] + ["##" + c for c in chars if c != " "]
    (tmp_path / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)), encoding="utf-8")
    tok = BertTokenizerFast(str(tmp_path / "vocab.txt"), do_lower_case=False)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, max_position_embeddings=96)
    return tok, LMPRegressor(config=cfg)

def test_fit_predict_checkpoint_roundtrip(tmp_path):
    cfg = load_config(); cfg["regressor"].update(epochs=2, batch_size=8, max_length=64, patience=2, device="cpu", learning_rate=1e-3)
    df = load_pairs(ROOT / "data/synthetic/synthetic_pairs.csv"); tr, va, te = document_split(df, seed=0)
    tok, model = _tiny(tmp_path, df)
    model, hist = trainer.fit(tr, va, cfg, 0, tok, model=model, ckpt_dir=tmp_path / "ck")
    assert len(hist["train_loss"]) >= 1 and (tmp_path / "ck/model.pt").exists() and json.loads((tmp_path / "ck/history.json").read_text())["seed"] == 0
    p = trainer.predict(model, tok, list(te.original), list(te.simplified), 64)
    assert p.shape == (len(te),) and ((p >= 1) & (p <= 10)).all()

def test_training_is_reproducible_given_seed(tmp_path):
    cfg = load_config(); cfg["regressor"].update(epochs=1, batch_size=8, max_length=64, device="cpu", learning_rate=1e-3)
    df = load_pairs(ROOT / "data/synthetic/synthetic_pairs.csv"); tr, va, _ = document_split(df, seed=0)
    out = []
    for _ in range(2):
        torch.manual_seed(0); tok, m = _tiny(tmp_path, df)
        _, h = trainer.fit(tr, va, cfg, 7, tok, model=m); out.append(h["train_loss"][0])
    assert out[0] == pytest.approx(out[1], rel=1e-5)
