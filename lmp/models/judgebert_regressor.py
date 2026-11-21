"""JUDGEBERT-inspired sentence-pair regressor.

Design (deviations from the original JUDGEBERT are listed in docs/METHODOLOGY.md):
  [CLS] original [SEP] simplified [SEP] -> multilingual transformer encoder (cross-encoding, so the two
  sentences attend to each other) -> mean-pooled hidden states -> dropout -> linear -> sigmoid -> 1 + 9*sigmoid.
Targets in [1,10] are rescaled to [0,1] for the loss. Default backbone: xlm-roberta-base.
"""
from __future__ import annotations
import torch
from torch import nn

class LMPRegressor(nn.Module):
    def __init__(self, backbone: str = "xlm-roberta-base", dropout: float = 0.1, config=None):
        super().__init__()
        from transformers import AutoModel
        if config is not None:            # offline / tiny random-init model for tests
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(backbone)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.encoder.config, "type_vocab_size", 1) > 1:
            kw["token_type_ids"] = token_type_ids
        h = self.encoder(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.sigmoid(self.head(self.drop(pooled))).squeeze(-1)   # in (0,1)

def to_unit(score):  # [1,10] -> [0,1]
    return (score - 1.0) / 9.0

def from_unit(x):    # [0,1] -> [1,10]
    return 1.0 + 9.0 * x
