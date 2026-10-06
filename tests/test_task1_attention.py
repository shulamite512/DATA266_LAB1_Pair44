"""Correctness tests for the hand-written attention in task1_llm/parth/src/model.py.

They check output shapes, that the causal mask zeroes every future position, that
editing a future token cannot change earlier logits, that attention rows sum to one,
and that no prebuilt attention or Transformer module is used anywhere in the source.
"""
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "task1_llm" / "parth" / "src"
sys.path.insert(0, str(SRC))

from model import (  # noqa: E402
    GPTConfig, GPTLanguageModel, MultiHeadSelfAttention, scaled_dot_product_attention,
)

B, T, C, H, V = 2, 10, 32, 4, 20


def tiny_config(**overrides) -> GPTConfig:
    base = dict(vocab_size=V, context_length=16, d_model=C, n_heads=H, n_layers=2, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def test_sdpa_output_shapes():
    torch.manual_seed(0)
    D = C // H
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
    out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
    assert out.shape == (B, H, T, D)
    assert weights.shape == (B, H, T, T)


def test_mhsa_and_model_output_shapes():
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = MultiHeadSelfAttention(cfg).eval()
    x = torch.randn(B, T, C)
    out, weights = attn(x, return_attn=True)
    assert out.shape == (B, T, C)
    assert weights.shape == (B, H, T, T)

    model = GPTLanguageModel(cfg).eval()
    idx = torch.randint(0, V, (B, T))
    logits, loss = model(idx, idx)
    assert logits.shape == (B, T, V)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_causal_mask_zeroes_future_positions():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(tiny_config()).eval()
    _, weights = attn(torch.randn(B, T, C), return_attn=True)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # s > t
    assert torch.all(weights[..., future] == 0), "attention leaked onto future positions"
    # The first query can only see itself, so its full weight sits on position 0.
    assert torch.allclose(weights[..., 0, 0], torch.ones(B, H))


def test_future_token_change_does_not_affect_past_logits():
    torch.manual_seed(0)
    model = GPTLanguageModel(tiny_config()).eval()
    idx = torch.randint(0, V, (1, T))
    for t in (3, T - 1):
        changed = idx.clone()
        changed[0, t] = (changed[0, t] + 1) % V
        with torch.no_grad():
            a, _ = model(idx)
            b, _ = model(changed)
        assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6), f"positions < {t} changed"
        assert not torch.allclose(a[:, t], b[:, t]), f"position {t} should react to its own token"


def test_attention_rows_sum_to_one_over_valid_positions():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(tiny_config()).eval()
    _, weights = attn(torch.randn(B, T, C), return_attn=True)
    valid = torch.tril(torch.ones(T, T, dtype=torch.bool))
    row_sums = (weights * valid).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(weights >= 0)


def test_no_prebuilt_attention_modules_in_model():
    model = GPTLanguageModel(tiny_config())
    banned_types = tuple(
        getattr(nn, name)
        for name in ("MultiheadAttention", "Transformer", "TransformerEncoder", "TransformerDecoder",
                     "TransformerEncoderLayer", "TransformerDecoderLayer")
        if hasattr(nn, name)
    )
    for name, module in model.named_modules():
        assert not isinstance(module, banned_types), f"prebuilt module found: {name} ({type(module).__name__})"


def test_forward_never_calls_torch_sdpa(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("torch's built-in scaled_dot_product_attention was called")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
    model = GPTLanguageModel(tiny_config()).eval()
    idx = torch.randint(0, V, (B, T))
    logits, _ = model(idx)
    assert logits.shape == (B, T, V)


BANNED_SOURCE_PATTERNS = [
    r"nn\.MultiheadAttention",
    r"nn\.Transformer\b",
    r"TransformerEncoder",
    r"TransformerDecoder",
    r"F\.scaled_dot_product_attention",
    r"functional\.scaled_dot_product_attention",
    r"from\s+torch\.nn\.functional\s+import[^\n]*scaled_dot_product_attention",
    r"^\s*(from|import)\s+transformers\b",
    r"AutoTokenizer",
    r"^\s*(from|import)\s+tiktoken\b",
    r"^\s*(from|import)\s+sentencepiece\b",
    r"^\s*(from|import)\s+tokenizers\b",
]


@pytest.mark.parametrize("source_file", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_source_has_no_prohibited_modules(source_file):
    text = source_file.read_text(encoding="utf-8")
    for pattern in BANNED_SOURCE_PATTERNS:
        match = re.search(pattern, text, flags=re.MULTILINE)
        assert match is None, f"{source_file.name} contains prohibited usage: {match.group(0)!r}"
