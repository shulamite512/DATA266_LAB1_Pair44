"""Decoder-only character-level GPT built from basic tensor operations.

Attention, causal masking, and head splitting are written by hand with matmul,
masked_fill, and softmax, so every shape change is visible. Blocks use the pre-norm
layout and a final LayerNorm feeds the language modelling head.

Shape legend used in comments: B = batch, T = sequence length, C = d_model,
H = n_heads, D = head_dim = C / H, V = vocab_size.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

# Descriptive keys in model.yaml must match what this file actually implements, so the
# config can never claim an architecture the code does not build.
_IMPLEMENTED = {
    "model_type": "decoder-only-gpt",
    "tokenization": "character-level",
    "normalization": "pre-norm",
    "positional_embedding": "learned",
    "activation": "gelu",
}


def _normalize(value) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    ffn_hidden_multiplier: int = 4
    bias: bool = True
    tie_weights: bool = False

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, cfg: dict, vocab_size: int | None = None) -> "GPTConfig":
        for key, expected in _IMPLEMENTED.items():
            if key in cfg and _normalize(cfg[key]) != expected:
                raise ValueError(f"model.yaml {key}={cfg[key]!r} but model.py implements {expected!r}")
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0, training: bool = False):
    """Manual attention: softmax(Q K^T / sqrt(D)) V with an optional boolean mask.

    q, k, v have shape (B, H, T, D). mask is boolean and broadcastable to (B, H, T, T),
    with True meaning "query may attend to this key". Returns the per-head output of
    shape (B, H, T, D) and the pre-dropout attention weights of shape (B, H, T, T).
    """
    d = q.size(-1)
    # (B, H, T, D) @ (B, H, D, T) -> scores (B, H, T, T); row t = query t, column s = key s
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        # Blocked positions get -inf, so exp(-inf) = 0 after softmax: zero weight on the future.
        scores = scores.masked_fill(~mask, float("-inf"))
    # Why: softmax runs in float32 even under fp16/bf16 autocast, because exponentiating
    # low-precision logits is where attention overflow and NaNs usually start.
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # (B, H, T, T), rows sum to 1
    attn = F.dropout(weights, p=dropout_p, training=training) if dropout_p > 0 else weights
    # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D): each query's weighted mix of value vectors
    out = torch.matmul(attn, v)
    return out, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.attn_dropout_p = cfg.dropout
        # Why: one fused (C -> 3C) projection is mathematically identical to three separate
        # Q, K, V layers but runs as a single matmul.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular causal mask, (1, 1, T_max, T_max). Entry [t, s] is True when s <= t.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_length, cfg.context_length),
                             persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape  # x: (B, T, C)
        H, D = self.n_heads, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=-1)  # (B, T, 3C) -> three tensors of (B, T, C)

        # Head split: (B, T, C) -> (B, T, H, D) -> (B, H, T, D) so each head attends independently
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        mask = self.causal_mask[:, :, :T, :T]  # (1, 1, T, T), broadcast over B and H
        out, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.attn_dropout_p, training=self.training
        )  # out: (B, H, T, D), weights: (B, H, T, T)

        # Merge heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))  # (B, T, C)
        return (out, weights) if return_attn else out


class FeedForward(nn.Module):
    """Position-wise MLP: (B, T, C) -> (B, T, mult * C) -> GELU -> (B, T, C)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_hidden_multiplier * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Why: pre-norm keeps an unnormalized residual path from input to output, which
        # gives cleaner gradient flow and more stable early training than post-norm.
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        x = x + self.ffn(self.ln2(x))  # (B, T, C)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Why: each block adds two residual branches, so the output projections are scaled
        # by 1/sqrt(2 * n_layers) to keep the residual stream variance from growing with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("ffn.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape  # idx: (B, T) integer token ids
        if T > self.config.context_length:
            raise ValueError(f"Sequence length {T} exceeds context_length {self.config.context_length}")
        pos = torch.arange(T, device=idx.device)  # (T,)
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(pos)  # (T, C), broadcast over the batch
        x = self.drop(tok_emb + pos_emb)  # (B, T, C)
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            # Flatten to (B*T, V) logits vs (B*T,) targets; every position is a next-char prediction.
            loss = F.cross_entropy(logits.reshape(B * T, -1).float(), targets.reshape(B * T))
        return logits, loss

    def count_parameters(self, trainable_only: bool = True) -> int:
        # parameters() yields a tied weight once, so tying is not double counted.
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def configure_optimizer(self, learning_rate: float, weight_decay: float,
                            betas=(0.9, 0.95)) -> torch.optim.AdamW:
        # Why: decay applies only to matrices (linear and embedding weights). Biases and
        # LayerNorm gains are 1-D and shrinking them toward zero hurts without regularizing.
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=tuple(betas))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = False) -> torch.Tensor:
        """Autoregressively append max_new_tokens ids to idx of shape (B, T0).

        Greedy takes the argmax; otherwise logits are divided by temperature, optionally
        restricted to the top_k candidates, and sampled. The context is cropped to the
        last context_length ids each step, and no KV cache is used.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]  # (B, <=T)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()  # (B, V): distribution for the next character
            if greedy or temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(int(top_k), logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, 1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx
