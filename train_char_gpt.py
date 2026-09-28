from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = WORKSPACE_ROOT / "TinyStories"
TRAIN_PATH = DATA_DIR / "TinyStories-train.txt"
VAL_PATH = DATA_DIR / "TinyStories-valid.txt"
OUTPUT_DIR = PROJECT_ROOT / "member_1" / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SAMPLE_PATH = OUTPUT_DIR / "samples.txt"
MAX_TRAIN_CHARS = 500000
MAX_VAL_CHARS = 50000
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        if RAW_DATA_DIR.exists():
            for filename in ["TinyStories-train.txt", "TinyStories-valid.txt"]:
                src = RAW_DATA_DIR / filename
                dst = DATA_DIR / filename
                if src.exists() and not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            raise FileNotFoundError(
                "TinyStories dataset not found. Place the files in CLab1/TinyStories/ or task1_llm/data/."
            )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class CharSequenceDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128, char_to_idx: Dict[str, int] | None = None):
        self.seq_len = seq_len
        if char_to_idx is None:
            chars = sorted(set(text))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        else:
            self.char_to_idx = char_to_idx
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        unknown_chars = sorted(set(text) - set(self.char_to_idx))
        if unknown_chars:
            raise ValueError(f"Validation text contains characters absent from training vocabulary: {unknown_chars}")
        data = [self.char_to_idx[ch] for ch in text]
        if len(data) < seq_len + 1:
            pad = np.zeros(seq_len + 1 - len(data), dtype=int)
            data = np.concatenate([data, pad])
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.seq_len + 1
        seq = self.data[start:end]
        x = seq[:-1]
        y = seq[1:]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalSelfAttention(emb_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, num_heads: int = 4, num_layers: int = 3, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(seq_len, emb_dim)
        self.blocks = nn.ModuleList(
            [GPTBlock(emb_dim, num_heads, emb_dim * 2) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(x) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def create_loader(text: str, seq_len: int, batch_size: int, shuffle: bool):
    dataset = CharSequenceDataset(text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(train_text: str, val_text: str, epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE, seq_len: int = 128, lr: float = 3e-5, early_stopping_patience: int = EARLY_STOPPING_PATIENCE):
    shared_chars = sorted(set(train_text) | set(val_text))
    shared_char_to_idx = {ch: i for i, ch in enumerate(shared_chars)}
    train_dataset = CharSequenceDataset(train_text, seq_len, char_to_idx=shared_char_to_idx)
    val_dataset = CharSequenceDataset(val_text, seq_len, char_to_idx=train_dataset.char_to_idx)

    vocab_size = len(train_dataset.char_to_idx)
    model = CharGPT(vocab_size=vocab_size, emb_dim=96, num_heads=4, num_layers=3, seq_len=seq_len)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, len(train_dataset) // batch_size // 4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(1, warmup_steps), 1.0),
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [], "train_accuracy": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_tokens = 0
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()
            train_tokens += yb.numel()

        train_loss = running_loss / len(train_dataset)
        train_accuracy = train_correct / train_tokens
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_tokens += yb.numel()
        val_loss /= len(val_dataset)
        val_accuracy = val_correct / val_tokens

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(math.exp(train_loss))
        history["val_ppl"].append(math.exp(val_loss))
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={math.exp(val_loss):.2f} | train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs; best val_loss={best_val_loss:.4f}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_dataset, history


def greedy_generate(model: nn.Module, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str], seq_len: int, seed_text: str = "The ", max_new_tokens: int = 80):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([char_to_idx[ch] for ch in seed_text], dtype=torch.long, device=device).unsqueeze(0)
    generated = seed_text
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > seq_len:
                input_ids = input_ids[:, -seq_len:]
            logits = model(input_ids)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            next_char = idx_to_char[int(next_token.item())]
            generated += next_char
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_char == "\n" and len(generated) > 200:
                break
    return generated


def main() -> None:
    print("[Smoke test] Starting TinyStories char-level GPT training...")
    print(f"[Smoke test] Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[Smoke test] GPU: {torch.cuda.get_device_name(DEVICE)}")
    ensure_data_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_text = read_text(TRAIN_PATH)[:MAX_TRAIN_CHARS]
    val_text = read_text(VAL_PATH)[:MAX_VAL_CHARS]

    print(f"[Smoke test] Loaded {len(train_text)} training chars and {len(val_text)} validation chars.")
    print("[Smoke test] Using up to 20 epochs with 3e-5 learning rate, more data, and early stopping to reduce overfitting.")
    model, train_dataset, history = train_model(train_text, val_text, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, seq_len=128, lr=3e-5)

    sample = greedy_generate(
        model,
        train_dataset.char_to_idx,
        train_dataset.idx_to_char,
        seq_len=128,
        seed_text="Once upon a time",
        max_new_tokens=80,
    )

    SAMPLE_PATH.write_text(sample + "\n", encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nSample generation:")
    print(sample)
    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Sample saved to: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
