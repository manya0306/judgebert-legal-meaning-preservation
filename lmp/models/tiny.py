"""Tiny RANDOMLY-INITIALISED BERT + word-level tokenizer built from given texts. For tests and design diagnostics only - it has NO pretrained
knowledge, so results with it say nothing about XLM-R or any pretrained encoder."""
from __future__ import annotations
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, PreTrainedTokenizerFast


def build_tokenizer(texts, max_vocab: int = 6000):
    cnt = Counter(t for s in texts for t in s.lower().replace("।", " । ").split())
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {w: i for i, w in enumerate(specials + [w for w, _ in cnt.most_common(max_vocab)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    from tokenizers import normalizers
    tk.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                  model_input_names=["input_ids", "token_type_ids", "attention_mask"])


def tiny_config(vocab_size: int, hidden: int = 64, layers: int = 2, heads: int = 2, max_pos: int = 256):
    return BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                      intermediate_size=hidden * 2, max_position_embeddings=max_pos, type_vocab_size=2)
