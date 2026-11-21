"""Multi-task multilingual cross-encoder for legal meaning preservation (JUDGEBERT-INSPIRED, not a reproduction of JUDGEBERT).

Input   : [CLS] original [SEP] simplified [SEP]  (one joint sequence -> every layer lets the two sentences attend to each other)
Pooling : concat([CLS], masked-mean over tokens, masked-mean over segment B - segment A)  -> interaction-aware representation
Heads   : regression (linear, 1-10 scale via unit rescale; NO sigmoid squashing; clipped only at inference)
          + 3-way classification (Not / Partially / Preserved) trained with class-weighted cross-entropy
Loss    : Huber(regression) + lambda * CE(classification)  (categories are derived from scores, so the heads are consistent by construction)
Reason for changes vs the legacy LMPRegressor (see docs/PROJECT_AUDIT.md section 5): sigmoid head + mean-pool only + delta-0.2 Huber +
          one tiny LR for head and encoder + checkpoint selection by validation RMSE (which a constant predictor can win).
"""
from __future__ import annotations
import torch
from torch import nn
from ..data.schema import CATEGORIES

SCORE_MID, SCORE_HALF = 5.5, 4.5     # regression target z = (score - 5.5) / 4.5  in [-1, 1]


def to_z(score):
    return (score - SCORE_MID) / SCORE_HALF


def from_z(z):
    return SCORE_MID + SCORE_HALF * z


class LMPCrossEncoder(nn.Module):
    def __init__(self, backbone: str = "xlm-roberta-base", dropout: float = 0.1, config=None, num_classes: int = 3,
                 use_language_embedding: bool = False, freeze_layers: int = 0):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_config(config) if config is not None else AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.lang_emb = nn.Embedding(2, h) if use_language_embedding else None
        self.reg_head = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h, 1))
        self.cls_head = nn.Sequential(nn.Linear(3 * h, h), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h, num_classes))
        if freeze_layers:
            self.freeze_bottom(freeze_layers)

    def freeze_bottom(self, n: int) -> None:
        """CPU-friendly option: freeze embeddings and the lowest n transformer layers."""
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        for layer in self.encoder.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def pooled(self, input_ids, attention_mask, token_type_ids=None, lang_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.encoder.config, "type_vocab_size", 1) > 1:
            kw["token_type_ids"] = token_type_ids
        h = self.encoder(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        if token_type_ids is not None and token_type_ids.max() > 0:      # segment-wise means; for XLM-R (no token types) fall back to the global mean
            mb = (token_type_ids.unsqueeze(-1).to(h.dtype) * m)
            ma = ((1 - token_type_ids).unsqueeze(-1).to(h.dtype) * m)
            diff = (h * mb).sum(1) / mb.sum(1).clamp(min=1) - (h * ma).sum(1) / ma.sum(1).clamp(min=1)
        else:
            diff = torch.zeros_like(mean)
        z = torch.cat([h[:, 0], mean, diff], dim=-1)
        if self.lang_emb is not None and lang_ids is not None:
            z = z + torch.cat([self.lang_emb(lang_ids)] * 3, dim=-1)
        return self.drop(z)

    def forward(self, input_ids, attention_mask, token_type_ids=None, lang_ids=None):
        z = self.pooled(input_ids, attention_mask, token_type_ids, lang_ids)
        return self.reg_head(z).squeeze(-1), self.cls_head(z)          # (z-score regression, class logits)

    @torch.no_grad()
    def init_regression_bias(self, mean_z: float) -> None:
        self.reg_head[-1].bias.fill_(float(mean_z))


def class_names():
    return list(CATEGORIES)
