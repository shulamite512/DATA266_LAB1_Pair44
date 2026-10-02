"""Task 1: character-level GPT trained from scratch on TinyStories.

Expected location: task1_llm/<member>/src/train_char_gpt.py (config file next to it).

    python train_char_gpt.py            # full run using char_gpt_config.json
    python train_char_gpt.py --smoke    # tiny end-to-end run, written to <member>/smoke_test/
    python train_char_gpt.py --config path/to/config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SCRIPT_DIR.parent
TASK_DIR = MEMBER_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "char_gpt_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STORY_SEPARATOR = "<|endoftext|>"
UNK_CHAR = "�"


# ---------------------------------------------------------------- run bookkeeping

class Tee:
    """Mirror stdout into the raw log file so every run leaves an unedited log behind."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def relative(path: Path) -> str:
    """Paths written to logs/manifests are relative to the task folder (no personal paths)."""
    try:
        return path.resolve().relative_to(TASK_DIR.parent).as_posix()
    except ValueError:
        return path.name


def cpu_name() -> str:
    name = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return name or platform.machine()


def hardware_info() -> dict:
    info = {"device": str(DEVICE), "cpu": cpu_name(), "platform": platform.platform()}
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        info.update({"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 2), "cuda_version": torch.version.cuda})
    return info


def peak_memory_mb() -> dict:
    memory = {"gpu_peak_allocated_mb": None, "cpu_peak_rss_mb": None}
    if DEVICE.type == "cuda":
        memory["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory["cpu_peak_rss_mb"] = peak / 1024 ** 2 if sys.platform == "darwin" else peak / 1024
    except ImportError:
        try:
            import psutil
            memory["cpu_peak_rss_mb"] = psutil.Process().memory_info().peak_wset / 1024 ** 2
        except (ImportError, AttributeError):
            pass
    return memory


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, run_id: str, config: dict, artifacts: dict) -> None:
    import importlib.metadata as metadata

    packages = {}
    for name in ("torch", "numpy", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "run_id": run_id,
        "script": relative(Path(__file__)),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "packages": packages,
        "hardware": hardware_info(),
        "config": config,
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data

def ensure_source_file(data_cfg: dict) -> Path:
    data_dir = TASK_DIR / data_cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / data_cfg["source_file"]
    if not path.exists():
        url = f"{data_cfg['download_base_url']}/{data_cfg['source_file']}"
        print(f"Downloading {url} ...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def load_stories(path: Path, limit: int) -> List[str]:
    """Stream stories (separated by <|endoftext|>) so the 1.9 GB file is never fully loaded."""
    stories: List[str] = []
    current: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if STORY_SEPARATOR in line:
                head, _, tail = line.partition(STORY_SEPARATOR)
                current.append(head)
                story = "".join(current).strip()
                if story:
                    stories.append(story)
                    if len(stories) >= limit:
                        break
                current = [tail] if tail.strip() else []
            else:
                current.append(line)
    if len(stories) < limit and "".join(current).strip():
        stories.append("".join(current).strip())
    return stories


def make_split(data_cfg: dict, seed: int, output_dir: Path) -> Tuple[List[str], List[str]]:
    source = ensure_source_file(data_cfg)
    n_train, n_val = data_cfg["train_stories"], data_cfg["val_stories"]
    pool = load_stories(source, max(data_cfg["pool_stories"], n_train + n_val))
    if len(pool) < n_train + n_val:
        raise ValueError(f"Only {len(pool)} stories available; need {n_train + n_val}.")
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    train_ids, val_ids = order[:n_train], order[n_train:n_train + n_val]
    (output_dir / "split_indices.json").write_text(
        json.dumps({"source_file": data_cfg["source_file"], "pool_size": len(pool), "seed": seed, "train": train_ids, "val": val_ids}),
        encoding="utf-8",
    )
    return [pool[i] for i in train_ids], [pool[i] for i in val_ids]


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text) | {UNK_CHAR})
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text: str, char_to_idx: Dict[str, int]) -> torch.Tensor:
    """Vectorised char -> id mapping; characters unseen in training map to UNK."""
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    unique, inverse = np.unique(codepoints, return_inverse=True)
    unk = char_to_idx[UNK_CHAR]
    lookup = np.array([char_to_idx.get(chr(c), unk) for c in unique], dtype=np.int32)
    return torch.from_numpy(lookup[inverse.reshape(-1)])


class CharSequenceDataset(Dataset):
    """Fixed-length (input, target) windows where target is the input shifted by one character."""

    def __init__(self, data: torch.Tensor, seq_len: int, stride: int):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride
        self.count = max(0, (len(data) - seq_len - 1) // stride + 1)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        seq = self.data[start:start + self.seq_len + 1].long()
        return seq[:-1], seq[1:]


# ---------------------------------------------------------------- model (from scratch, no nn.MultiheadAttention)

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
    def __init__(self, vocab_size: int, emb_dim: int = 128, num_heads: int = 4, num_layers: int = 3, seq_len: int = 128, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(seq_len, emb_dim)
        self.blocks = nn.ModuleList(
            [GPTBlock(emb_dim, num_heads, emb_dim * ff_mult, dropout) for _ in range(num_layers)]
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


def build_model(vocab_size: int, model_cfg: dict) -> CharGPT:
    return CharGPT(
        vocab_size=vocab_size,
        emb_dim=model_cfg["emb_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        seq_len=model_cfg["seq_len"],
        ff_mult=model_cfg["ff_mult"],
        dropout=model_cfg["dropout"],
    )


# ---------------------------------------------------------------- training

def warmup_cosine(warmup_steps: int, total_steps: int, min_ratio: float):
    """Linear warm-up to the peak LR, then cosine decay down to min_ratio * peak."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, vocab_size: int) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, tokens = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        total_loss += loss_fn(logits.view(-1, vocab_size), yb.view(-1)).item()
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        tokens += yb.numel()
    return total_loss / tokens, correct / tokens


def save_checkpoint(path: Path, model: nn.Module, config: dict, char_to_idx: Dict[str, int], epoch: int, val_loss: float) -> None:
    torch.save({"model_state": model.state_dict(), "config": config, "char_to_idx": char_to_idx, "epoch": epoch, "val_loss": val_loss}, path)


def train_model(config: dict, train_data: torch.Tensor, val_data: torch.Tensor, vocab_size: int, char_to_idx: Dict[str, int], dirs: dict):
    model_cfg, train_cfg = config["model"], config["training"]
    seq_len = model_cfg["seq_len"]
    train_dataset = CharSequenceDataset(train_data, seq_len, train_cfg["window_stride"])
    val_dataset = CharSequenceDataset(val_data, seq_len, seq_len)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"] * 4, shuffle=False, pin_memory=pin)
    print(f"Train windows: {len(train_dataset):,} | val windows: {len(val_dataset):,} | steps/epoch: {len(train_loader):,}")

    model = build_model(vocab_size, model_cfg).to(DEVICE)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = max(1, int(steps_per_epoch * train_cfg["warmup_fraction_of_epoch"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup_steps, total_steps, train_cfg["min_lr_ratio"]))
    loss_fn = nn.CrossEntropyLoss()
    print(f"LR schedule: linear warm-up for {warmup_steps} steps, cosine decay to {train_cfg['min_lr_ratio']} x peak over {total_steps} steps")

    epoch_history: List[dict] = []
    step_history: List[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    loss_ema = None
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    run_start = time.perf_counter()

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss, train_correct, train_tokens = 0.0, 0, 0
        grad_norms: List[float] = []
        spikes, nan_steps = 0, 0
        sync()
        epoch_start = time.perf_counter()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            if not torch.isfinite(loss):
                nan_steps += 1
                scheduler.step()
                global_step += 1
                continue
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip"]))
            if not math.isfinite(grad_norm):
                nan_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                continue
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_value = loss.item()
            if loss_ema is not None and loss_value > train_cfg["loss_spike_factor"] * loss_ema:
                spikes += 1
            loss_ema = loss_value if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value
            grad_norms.append(grad_norm)
            running_loss += loss_value * yb.numel()
            train_correct += (logits.argmax(dim=-1) == yb).sum().item()
            train_tokens += yb.numel()
            if global_step % train_cfg["log_every_steps"] == 0:
                step_history.append({"step": global_step, "epoch": epoch, "loss": loss_value, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0]})
        sync()
        epoch_seconds = time.perf_counter() - epoch_start

        train_loss = running_loss / max(1, train_tokens)
        train_accuracy = train_correct / max(1, train_tokens)
        val_loss, val_accuracy = evaluate(model, val_loader, vocab_size)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": math.exp(train_loss),
            "val_ppl": math.exp(val_loss),
            "train_bpc": train_loss / math.log(2),
            "val_bpc": val_loss / math.log(2),
            "generalization_gap": val_loss - train_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
            "loss_spikes": spikes,
            "nan_steps": nan_steps,
            "lr_end": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
            "train_tokens_per_sec": train_tokens / epoch_seconds,
        }
        epoch_history.append(record)
        print(
            f"Epoch {epoch}/{train_cfg['epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={record['val_ppl']:.2f} "
            f"| val_bpc={record['val_bpc']:.3f} | train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f} | grad_norm(mean/max)={record['grad_norm_mean']:.3f}/{record['grad_norm_max']:.3f} "
            f"| spikes={spikes} | nan={nan_steps} | {record['train_tokens_per_sec']:,.0f} tok/s | {epoch_seconds:.1f}s"
        )

        save_checkpoint(dirs["checkpoints"] / "last.pt", model, config, char_to_idx, epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(dirs["checkpoints"] / "best.pt", model, config, char_to_idx, epoch, val_loss)
            print(f"  new best val_loss -> saved checkpoints/best.pt (epoch {epoch})")
        else:
            epochs_without_improvement += 1
        if epoch >= train_cfg["min_epochs"] and epochs_without_improvement >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping after {epoch} epochs; best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")
            break

    total_training_seconds = time.perf_counter() - run_start
    checkpoint = torch.load(dirs["checkpoints"] / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    summary = {
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_run": len(epoch_history),
        "total_training_seconds": total_training_seconds,
        **peak_memory_mb(),
    }
    return model, epoch_history, step_history, summary


# ---------------------------------------------------------------- generation + diversity metrics

@torch.no_grad()
def generate(model: CharGPT, prompt: str, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str], max_new_tokens: int, temperature: float, top_k: int | None) -> str:
    """temperature <= 0 means greedy decoding."""
    model.eval()
    unk = char_to_idx[UNK_CHAR]
    ids = torch.tensor([[char_to_idx.get(ch, unk) for ch in prompt]], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.seq_len:])[:, -1, :]
        if temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return "".join(idx_to_char[int(i)] for i in ids[0, len(prompt):].tolist())


def word_ngrams(text: str, n: int) -> List[tuple]:
    words = text.split()
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def distinct_n(texts: List[str], n: int) -> float:
    grams = [gram for text in texts for gram in word_ngrams(text, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    grams = word_ngrams(text, n)
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def run_generation(model: CharGPT, gen_cfg: dict, char_to_idx, idx_to_char, seed: int) -> Tuple[List[dict], dict]:
    samples: List[dict] = []
    summary: dict = {}
    for decoding in gen_cfg["decoding"]:
        torch.manual_seed(seed)
        texts = []
        generated_tokens = 0
        sync()
        start = time.perf_counter()
        for prompt in gen_cfg["prompts"]:
            text = generate(model, prompt, char_to_idx, idx_to_char, gen_cfg["max_new_tokens"], decoding["temperature"], decoding["top_k"])
            generated_tokens += gen_cfg["max_new_tokens"]
            texts.append(text)
            samples.append({"decoding": decoding["name"], "prompt": prompt, "generated": text, "repeated_4gram_rate": repeated_ngram_rate(text)})
        sync()
        seconds = time.perf_counter() - start
        name = decoding["name"]
        summary[name] = {
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
            "distinct_3": distinct_n(texts, 3),
            "repeated_4gram_rate": float(np.mean([repeated_ngram_rate(t) for t in texts])),
            "generation_tokens_per_sec": generated_tokens / seconds,
        }
    return samples, summary


# ---------------------------------------------------------------- reporting

def plot_curves(epoch_history: List[dict], step_history: List[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in epoch_history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [r["train_loss"] for r in epoch_history], marker="o", label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in epoch_history], marker="o", label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy (nats/char)", title="Training vs validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [r["train_accuracy"] for r in epoch_history], marker="o", label="train")
    axes[1].plot(epochs, [r["val_accuracy"] for r in epoch_history], marker="o", label="validation")
    axes[1].set(xlabel="epoch", ylabel="top-1 next-char accuracy", title="Next-character accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    if step_history:
        steps = [r["step"] for r in step_history]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(steps, [r["loss"] for r in step_history])
        axes[0].set(ylabel="train loss (step)")
        axes[1].plot(steps, [r["grad_norm"] for r in step_history], color="tab:orange")
        axes[1].set(ylabel="grad norm (pre-clip)")
        axes[2].plot(steps, [r["lr"] for r in step_history], color="tab:green")
        axes[2].set(ylabel="learning rate", xlabel="step")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_dynamics.png", dpi=150)
        plt.close(fig)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_report(path: Path, epoch_history: List[dict], summary: dict, generation: dict, hardware: dict) -> None:
    best = next(r for r in epoch_history if r["epoch"] == summary["best_epoch"])
    rows = [
        ("best_epoch", best["epoch"], "checkpoint: checkpoints/best.pt"),
        ("epochs_run", summary["epochs_run"], ""),
        ("train_cross_entropy", best["train_loss"], "nats/char, running mean over the epoch (dropout on)"),
        ("val_cross_entropy", best["val_loss"], "nats/char"),
        ("val_perplexity", best["val_ppl"], "exp(val CE)"),
        ("train_perplexity", best["train_ppl"], "exp(train CE)"),
        ("val_bits_per_char", best["val_bpc"], "val CE / ln 2"),
        ("train_bits_per_char", best["train_bpc"], "train CE / ln 2"),
        ("generalization_gap", best["generalization_gap"], "val CE - train CE"),
        ("val_top1_accuracy", best["val_accuracy"], "next-character"),
        ("train_top1_accuracy", best["train_accuracy"], "next-character"),
        ("grad_norm_mean_all_epochs", float(np.mean([r["grad_norm_mean"] for r in epoch_history])), "pre-clip L2 norm"),
        ("grad_norm_max_all_epochs", float(np.max([r["grad_norm_max"] for r in epoch_history])), "pre-clip L2 norm"),
        ("loss_spikes_total", sum(r["loss_spikes"] for r in epoch_history), "step loss > spike_factor x EMA"),
        ("nan_steps_total", sum(r["nan_steps"] for r in epoch_history), "non-finite loss or grad (step skipped)"),
        ("parameter_count", summary["parameter_count"], ""),
        ("train_tokens_per_sec_mean", float(np.mean([r["train_tokens_per_sec"] for r in epoch_history])), ""),
        ("total_training_seconds", summary["total_training_seconds"], "includes validation passes"),
        ("gpu_peak_allocated_mb", summary["gpu_peak_allocated_mb"], ""),
        ("cpu_peak_rss_mb", summary["cpu_peak_rss_mb"], ""),
    ]
    for name, values in generation.items():
        rows += [
            (f"{name}_distinct_1", values["distinct_1"], "word-level, pooled over prompts"),
            (f"{name}_distinct_2", values["distinct_2"], "word-level, pooled over prompts"),
            (f"{name}_distinct_3", values["distinct_3"], "word-level, pooled over prompts"),
            (f"{name}_repeated_4gram_rate", values["repeated_4gram_rate"], "word-level, mean over samples"),
            (f"{name}_generation_tokens_per_sec", values["generation_tokens_per_sec"], "batch size 1"),
        ]
    rows.append(("hardware", hardware.get("gpu") or hardware["cpu"], hardware["device"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "notes"])
        writer.writerows(rows)


def write_failure_candidates(path: Path, samples: List[dict]) -> None:
    ranked = sorted(samples, key=lambda s: s["repeated_4gram_rate"], reverse=True)
    lines = [
        "# Generated samples (failure-analysis candidates)",
        "",
        "Sorted by repeated 4-gram rate. Pick three failures and write your own analysis in ../failure_analysis.md.",
        "",
    ]
    for i, sample in enumerate(ranked, 1):
        lines += [
            f"## {i}. `{sample['decoding']}` | repeated 4-gram rate {sample['repeated_4gram_rate']:.3f}",
            f"**Prompt:** {sample['prompt']}",
            "",
            "```text",
            sample["prompt"] + sample["generated"],
            "```",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


FAILURE_TEMPLATE = """# Task 1: Sequence model failure analysis

Samples to choose from: outputs/failure_candidates.md (checkpoint: checkpoints/best.pt)

## Case 1
- Decoding / prompt:
- Snippet:
- Failure type (repetition / broken grammar / loss of coherence / hallucination / ...):
- Observation:

## Case 2
- Decoding / prompt:
- Snippet:
- Failure type:
- Observation:

## Case 3
- Decoding / prompt:
- Snippet:
- Failure type:
- Observation:
"""


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline end to end")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    smoke_overrides = config.pop("smoke_test_overrides", {})
    if args.smoke:
        deep_update(config, smoke_overrides)
    base_dir = MEMBER_DIR / "smoke_test" if args.smoke else MEMBER_DIR
    dirs = {name: base_dir / name for name in ("outputs", "checkpoints", "logs", "manifests")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dirs["logs"] / f"train_{run_id}.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.__stdout__.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = Tee(sys.__stdout__, log_file)

    try:
        set_seed(config["seed"])
        hardware = hardware_info()
        print(f"Run {run_id}{' (smoke test)' if args.smoke else ''}")
        print(f"Hardware: {json.dumps(hardware)}")
        print(f"Config: {json.dumps(config)}")

        train_stories, val_stories = make_split(config["data"], config["seed"], dirs["outputs"])
        joiner = config["data"]["story_joiner"]
        train_text, val_text = joiner.join(train_stories), joiner.join(val_stories)
        char_to_idx, idx_to_char = build_vocab(train_text)
        (dirs["outputs"] / "vocab.json").write_text(json.dumps(char_to_idx, ensure_ascii=False, indent=1), encoding="utf-8")
        train_data, val_data = encode(train_text, char_to_idx), encode(val_text, char_to_idx)
        unk_in_val = int((val_data == char_to_idx[UNK_CHAR]).sum())
        print(f"Stories: train={len(train_stories):,} val={len(val_stories):,} | chars: train={len(train_data):,} val={len(val_data):,} | vocab={len(char_to_idx)} | unseen val chars mapped to UNK: {unk_in_val}")

        model, epoch_history, step_history, summary = train_model(config, train_data, val_data, len(char_to_idx), char_to_idx, dirs)

        samples, generation = run_generation(model, config["generation"], char_to_idx, idx_to_char, config["seed"])
        for sample in samples:
            if sample["decoding"] == config["generation"]["decoding"][0]["name"]:
                print(f"\n[{sample['decoding']}] {sample['prompt']}{sample['generated']}")

        outputs = dirs["outputs"]
        write_csv(outputs / "epoch_history.csv", epoch_history)
        write_csv(outputs / "step_history.csv", step_history)
        (outputs / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        write_failure_candidates(outputs / "failure_candidates.md", samples)
        plot_curves(epoch_history, step_history, outputs)
        metrics = {"run_id": run_id, "hardware": hardware, "summary": summary, "generation": generation, "epoch_history": epoch_history}
        (outputs / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_metrics_report(base_dir / "metrics_report.csv", epoch_history, summary, generation, hardware)
        failure_path = base_dir / "failure_analysis.md"
        if not failure_path.exists():
            failure_path.write_text(FAILURE_TEMPLATE, encoding="utf-8")

        write_manifest(
            dirs["manifests"] / f"manifest_{run_id}.json",
            run_id,
            config,
            {
                "raw_log": relative(log_path),
                "best_checkpoint": relative(dirs["checkpoints"] / "best.pt"),
                "best_epoch": summary["best_epoch"],
                "metrics_report": relative(base_dir / "metrics_report.csv"),
                "metrics_json": relative(outputs / "metrics.json"),
                "loss_curves": relative(outputs / "loss_curves.png"),
                "samples": relative(outputs / "samples.json"),
                "split_indices": relative(outputs / "split_indices.json"),
            },
        )
        print(f"\nDone. Metrics: {relative(base_dir / 'metrics_report.csv')} | raw log: {relative(log_path)}")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
