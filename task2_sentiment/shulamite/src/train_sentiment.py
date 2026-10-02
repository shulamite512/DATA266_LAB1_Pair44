"""Task 2: binary sentiment classification with embeddings learned from scratch.

Expected location: task2_sentiment/<member>/src/train_sentiment.py (config file next to it).

    python train_sentiment.py            # full run using sentiment_config.json
    python train_sentiment.py --smoke    # tiny end-to-end run, written to <member>/smoke_test/
    python train_sentiment.py --config path/to/config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
MEMBER_DIR = SCRIPT_DIR.parent
TASK_DIR = MEMBER_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "sentiment_config.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = {
    "yelp_polarity": {
        "url": "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz",
        "archive": "yelp_review_polarity_csv.tgz",
        "folder": "yelp_review_polarity_csv",
    },
    "imdb": {
        "url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "archive": "aclImdb_v1.tar.gz",
        "folder": "aclImdb",
    },
}


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
    """Paths written to logs/manifests are relative to the repo root (no personal paths)."""
    try:
        return path.resolve().relative_to(TASK_DIR.parent).as_posix()
    except ValueError:
        return path.name


def cpu_name() -> str:
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
    return platform.processor() or platform.machine()


def hardware_info() -> dict:
    info = {"device": str(DEVICE), "cpu": cpu_name(), "platform": platform.platform()}
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        info.update({"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 1024 ** 3, 2), "cuda_version": torch.version.cuda})
    return info


def cpu_peak_rss_mb() -> float | None:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 ** 2 if sys.platform == "darwin" else peak / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1024 ** 2
        except (ImportError, AttributeError):
            return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, run_id: str, config: dict, artifacts: dict) -> None:
    import importlib.metadata as metadata

    packages = {}
    for name in ("torch", "numpy", "scikit-learn", "scipy", "nltk", "matplotlib"):
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


# ---------------------------------------------------------------- data loading

def ensure_dataset(data_cfg: dict) -> Path:
    spec = DATASETS[data_cfg["dataset"]]
    data_dir = TASK_DIR / data_cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    folder = data_dir / spec["folder"]
    if folder.exists():
        return folder
    archive = data_dir / spec["archive"]
    if not archive.exists():
        print(f"Downloading {spec['url']} ...")
        partial = archive.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], partial)
        partial.rename(archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(data_dir, filter="data")
    return folder


def read_split(folder: Path, dataset: str, split: str) -> Tuple[List[str], List[int], dict]:
    """Return raw texts, labels (0=neg, 1=pos) and counts of dropped malformed rows."""
    texts: List[str] = []
    labels: List[int] = []
    dropped = {"malformed_rows": 0, "missing_or_empty_text": 0, "invalid_label": 0}
    if dataset == "yelp_polarity":
        csv.field_size_limit(10 ** 8)
        with (folder / f"{split}.csv").open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 2:
                    dropped["malformed_rows"] += 1
                    continue
                label, text = row
                if label.strip() not in {"1", "2"}:
                    dropped["invalid_label"] += 1
                    continue
                text = text.replace("\\n", " ").replace('\\"', '"').strip()
                if not text:
                    dropped["missing_or_empty_text"] += 1
                    continue
                texts.append(text)
                labels.append(int(label) - 1)
    else:
        for label_name, label in (("neg", 0), ("pos", 1)):
            for path in sorted((folder / split / label_name).glob("*.txt")):
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    dropped["missing_or_empty_text"] += 1
                    continue
                texts.append(text)
                labels.append(label)
    return texts, labels, dropped


def stratified_sample(labels: Sequence[int], limit: int, rng: np.random.Generator) -> np.ndarray:
    """Pick up to limit//2 indices per class so the sample is balanced."""
    labels = np.asarray(labels)
    chosen = []
    for label in (0, 1):
        candidates = np.where(labels == label)[0]
        chosen.append(rng.choice(candidates, size=min(len(candidates), limit // 2), replace=False))
    return rng.permutation(np.concatenate(chosen))


# ---------------------------------------------------------------- preprocessing

TAG_RE = re.compile(r"<[^>]+>")
NEGATION_RE = re.compile(r"n['’]t\b")
TOKEN_RE = re.compile(r"[a-z]+")


def make_preprocessor(cfg: dict) -> Callable[[str], List[str]]:
    stopwords = set(cfg["stopwords"])
    if cfg["stemming"]:
        from nltk.stem import PorterStemmer
        stem = lru_cache(maxsize=None)(PorterStemmer().stem)
    else:
        def stem(word: str) -> str:
            return word

    def preprocess(text: str) -> List[str]:
        text = TAG_RE.sub(" ", text.lower())
        if cfg["expand_negations"]:
            text = NEGATION_RE.sub(" not", text)
        return [stem(token) for token in TOKEN_RE.findall(text) if token not in stopwords]

    return preprocess


def build_vocab(texts: Sequence[Sequence[str]], min_frequency: int) -> Dict[str, int]:
    counts = Counter(token for text in texts for token in text)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_frequency:
            vocab[token] = len(vocab)
    return vocab


class ReviewDataset(Dataset):
    def __init__(self, texts: Sequence[Sequence[str]], labels: Sequence[int], vocab: Dict[str, int], max_length: int):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.inputs = torch.zeros((len(texts), max_length), dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [vocab.get(token, vocab["<unk>"]) for token in text[:max_length]]
            if ids:
                self.inputs[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]


def run_eda(raw: Dict[str, Tuple[List[str], List[int]]], tokens: Dict[str, List[List[str]]], dropped: dict, max_length: int, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eda = {"dropped_rows": dropped}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, (texts, labels) in raw.items():
        labels_arr = np.array(labels)
        raw_lengths = np.array([len(t.split()) for t in texts])
        processed_lengths = np.array([len(t) for t in tokens[split]])
        eda[split] = {
            "count": len(texts),
            "class_counts": {"negative": int((labels_arr == 0).sum()), "positive": int((labels_arr == 1).sum())},
            "positive_fraction": float(labels_arr.mean()),
            "duplicate_texts": len(texts) - len(set(texts)),
            "raw_word_length": {
                "mean": float(raw_lengths.mean()), "median": float(np.median(raw_lengths)),
                "p90": float(np.percentile(raw_lengths, 90)), "max": int(raw_lengths.max()),
                "mean_by_class": {"negative": float(raw_lengths[labels_arr == 0].mean()), "positive": float(raw_lengths[labels_arr == 1].mean())},
            },
            "processed_token_length": {"mean": float(processed_lengths.mean()), "median": float(np.median(processed_lengths))},
            "fraction_truncated_at_max_length": float((processed_lengths > max_length).mean()),
            "empty_after_preprocessing": int((processed_lengths == 0).sum()),
        }
        if split == "train":
            clip = np.percentile(raw_lengths, 99)
            for label, name in ((0, "negative"), (1, "positive")):
                axes[0].hist(np.minimum(raw_lengths[labels_arr == label], clip), bins=60, alpha=0.6, label=name)
            axes[0].set(xlabel="words per review (clipped at p99)", ylabel="reviews", title="Train review length by class")
            axes[0].legend()
    splits = list(raw.keys())
    negatives = [eda[s]["class_counts"]["negative"] for s in splits]
    positives = [eda[s]["class_counts"]["positive"] for s in splits]
    x = np.arange(len(splits))
    axes[1].bar(x - 0.2, negatives, 0.4, label="negative")
    axes[1].bar(x + 0.2, positives, 0.4, label="positive")
    axes[1].set_xticks(x, splits)
    axes[1].set(title="Class balance per split", ylabel="reviews")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eda.png", dpi=150)
    plt.close(fig)
    (output_dir / "eda.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    return eda


# ---------------------------------------------------------------- models

class MeanClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 96, channels: int = 96, kernel_sizes: Sequence[int] = (3, 4, 5)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, channels, width) for width in kernel_sizes)
        self.classifier = nn.Linear(channels * len(kernel_sizes), 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        return self.classifier(torch.cat(pooled, dim=1)).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.embedding(tokens))
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(representation).squeeze(-1)


def build_model(vocab_size: int, spec: dict) -> nn.Module:
    params = {key: value for key, value in spec.items() if key != "type"}
    classes = {"mean": MeanClassifier, "cnn": CNNClassifier, "gru": GRUClassifier}
    return classes[spec["type"]](vocab_size, **params)


# ---------------------------------------------------------------- training

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, train_cfg: dict, checkpoint_path: Path) -> Tuple[nn.Module, dict]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    start = time.perf_counter()
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    examples_seen = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(tokens), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            examples_seen += len(labels)
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                logits = model(tokens)
                valid_loss += loss_fn(logits, labels).item() * len(labels)
                valid_correct += ((logits >= 0).float() == labels).sum().item()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader.dataset),
            "val_loss": valid_loss / len(val_loader.dataset),
            "val_accuracy": valid_correct / len(val_loader.dataset),
        }
        history.append(record)
        print(f"Epoch {epoch}/{train_cfg['epochs']} | train_loss={record['train_loss']:.4f} | val_loss={record['val_loss']:.4f} | val_acc={record['val_accuracy']:.4f}")
        if record["val_loss"] < best_val_loss:
            best_val_loss = record["val_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping; restoring epoch {best_epoch} with val_loss={best_val_loss:.4f}.")
                break
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    stats = {
        "history": history,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": elapsed,
        "train_examples_per_second": examples_seen / elapsed,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 2 if DEVICE.type == "cuda" else None,
        "cpu_peak_rss_mb_process": cpu_peak_rss_mb(),
    }
    return model, stats


def predict(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for tokens, batch_labels in loader:
            probabilities.append(torch.sigmoid(model(tokens.to(DEVICE))).cpu().numpy())
            labels.append(batch_labels.numpy())
    elapsed = time.perf_counter() - start
    return np.concatenate(labels).astype(int), np.concatenate(probabilities), len(loader.dataset) / elapsed


# ---------------------------------------------------------------- metrics

def ece_score(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if selected.any():
            score += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(score)


def confusion_stats(labels: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Vectorised over a leading bootstrap axis: returns (accuracy, macro_f1, mcc)."""
    tp = ((predictions == 1) & (labels == 1)).sum(axis=-1).astype(float)
    tn = ((predictions == 0) & (labels == 0)).sum(axis=-1).astype(float)
    fp = ((predictions == 1) & (labels == 0)).sum(axis=-1).astype(float)
    fn = ((predictions == 0) & (labels == 1)).sum(axis=-1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fn + fp))
        mcc = np.nan_to_num((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return accuracy, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(labels: np.ndarray, predictions: np.ndarray, repetitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = {"accuracy": [], "macro_f1": [], "mcc": []}
    for start in range(0, repetitions, 100):
        chunk = min(100, repetitions - start)
        idx = rng.integers(0, len(labels), size=(chunk, len(labels)))
        accuracy, macro_f1, mcc = confusion_stats(labels[idx], predictions[idx])
        results["accuracy"].append(accuracy)
        results["macro_f1"].append(macro_f1)
        results["mcc"].append(mcc)
    return {name: [float(np.percentile(np.concatenate(v), 2.5)), float(np.percentile(np.concatenate(v), 97.5))] for name, v in results.items()}


def slice_masks(token_lengths: np.ndarray, has_negation: np.ndarray, eval_cfg: dict) -> Dict[str, np.ndarray]:
    masks = {f"length_{name}": (token_lengths >= low) & (token_lengths <= high) for name, (low, high) in eval_cfg["length_slices"].items()}
    masks["with_negation"] = has_negation
    masks["without_negation"] = ~has_negation
    return masks


def evaluate(name: str, labels: np.ndarray, probabilities: np.ndarray, slices: Dict[str, np.ndarray], eval_cfg: dict, seed: int) -> dict:
    predictions = (probabilities >= eval_cfg["threshold"]).astype(int)
    metrics = {"model": name, "accuracy": float(accuracy_score(labels, predictions))}
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average, zero_division=0)
        metrics.update({f"precision_{average}": float(precision), f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    metrics.update({
        "precision_by_class": precision.tolist(),
        "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "ece": ece_score(labels, probabilities, eval_cfg["ece_bins"]),
        "bootstrap_ci_95": bootstrap_ci(labels, predictions, eval_cfg["bootstrap_repetitions"], seed),
    })
    per_slice = {}
    for slice_name, mask in slices.items():
        if mask.any():
            per_slice[slice_name] = {
                "count": int(mask.sum()),
                "macro_f1": float(precision_recall_fscore_support(labels[mask], predictions[mask], average="macro", zero_division=0)[2]),
                "error_rate": float((predictions[mask] != labels[mask]).mean()),
            }
    metrics["slice_metrics"] = per_slice
    return metrics


def mcnemar_test(baseline: np.ndarray, experimental: np.ndarray, labels: np.ndarray) -> dict:
    baseline_correct = baseline == labels
    experimental_correct = experimental == labels
    baseline_only = int(np.sum(baseline_correct & ~experimental_correct))
    experimental_only = int(np.sum(~baseline_correct & experimental_correct))
    discordant = baseline_only + experimental_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(baseline_only, experimental_only), discordant, 0.5).pvalue)
    return {"baseline_only_correct": baseline_only, "experimental_only_correct": experimental_only, "test": "exact binomial (two-sided)", "p_value": p_value}


# ---------------------------------------------------------------- error review + plots

def select_error_candidates(labels: np.ndarray, probabilities: np.ndarray, threshold: float, slices: Dict[str, np.ndarray], slice_metrics: dict, raw_texts: Sequence[str]) -> Tuple[List[dict], str]:
    """Pick the 20 errors the brief asks for. Error types and fixes are left for manual review."""
    predictions = (probabilities >= threshold).astype(int)
    errors = predictions != labels
    confidence = np.abs(probabilities - threshold)
    worst_slice = max(slice_metrics, key=lambda s: slice_metrics[s]["error_rate"])
    groups = [
        ("confident_false_positive", np.where(errors & (predictions == 1))[0], -confidence),
        ("confident_false_negative", np.where(errors & (predictions == 0))[0], -confidence),
        ("near_threshold", np.where(errors)[0], confidence),
        (f"slice_specific ({worst_slice})", np.where(errors & slices[worst_slice])[0], confidence),
    ]
    chosen: List[dict] = []
    used = set()
    for category, indices, order_key in groups:
        picked = 0
        for index in indices[np.argsort(order_key[indices])]:
            if picked == 5:
                break
            if index in used:
                continue
            used.add(index)
            picked += 1
            chosen.append({
                "test_index": int(index), "category": category, "label": int(labels[index]),
                "prediction": int(predictions[index]), "probability_positive": float(probabilities[index]),
                "text": raw_texts[index],
            })
    return chosen, worst_slice


def write_error_review(chosen: List[dict], model_name: str, output_dir: Path, failure_path: Path) -> None:
    (output_dir / "error_review_candidates.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    if failure_path.exists():
        return
    lines = [
        "# Task 2: Error review (20 errors)",
        "",
        f"Model reviewed: `{model_name}` (checkpoint: checkpoints/{model_name}.pt). Full texts: outputs/error_review_candidates.json",
        "",
        "| # | test idx | category | label | pred | p(pos) | error type (manual) |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(chosen, 1):
        lines.append(f"| {i} | {row['test_index']} | {row['category']} | {row['label']} | {row['prediction']} | {row['probability_positive']:.3f} | |")
    lines += ["", "## Notes per error", ""]
    for i, row in enumerate(chosen, 1):
        snippet = row["text"][:400].replace("\n", " ")
        lines += [f"### {i}. {row['category']} (idx {row['test_index']})", f"> {snippet}{'...' if len(row['text']) > 400 else ''}", "", "- Error type:", "- Why the model got it wrong:", ""]
    lines += ["## Proposed testable fix", "", "- Fix:", "- How to test it:", ""]
    failure_path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(results: Dict[str, dict], curves: Dict[str, Tuple[np.ndarray, np.ndarray]], histories: Dict[str, List[dict]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    for ax, name in zip(np.atleast_1d(axes), names):
        matrix = np.array(results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        ax.set(title=name, xlabel="predicted", ylabel="true", xticks=[0, 1], yticks=[0, 1], xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (labels, probabilities) in curves.items():
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        axes[1].plot(recall, precision, label=f"{name} (AP {results[name]['pr_auc']:.3f})")
        edges = np.linspace(0, 1, 11)
        bins = np.clip(np.digitize(probabilities, edges) - 1, 0, 9)
        centers, observed = [], []
        for b in range(10):
            if (bins == b).any():
                centers.append(probabilities[bins == b].mean())
                observed.append(labels[bins == b].mean())
        axes[2].plot(centers, observed, marker="o", label=f"{name} (ECE {results[name]['ece']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC", xlabel="false positive rate", ylabel="true positive rate")
    axes[1].set(title="Precision-recall", xlabel="recall", ylabel="precision")
    axes[2].set(title="Reliability diagram", xlabel="mean predicted p(pos)", ylabel="observed positive rate")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_pr_calibration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, history in histories.items():
        epochs = [r["epoch"] for r in history]
        line, = ax.plot(epochs, [r["train_loss"] for r in history], marker="o", label=f"{name} train")
        ax.plot(epochs, [r["val_loss"] for r in history], marker="s", linestyle="--", color=line.get_color(), label=f"{name} val")
    ax.set(xlabel="epoch", ylabel="BCE loss", title="Training curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def write_metrics_report(path: Path, results: Dict[str, dict]) -> None:
    rows = []
    for metrics in results.values():
        row = {}
        for key, value in metrics.items():
            if key in {"history", "slice_metrics", "bootstrap_ci_95", "confusion_matrix", "hardware"} or isinstance(value, list):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    row[f"{key}_{sub_key}"] = sub_value
            else:
                row[key] = value
        tn, fp, fn, tp = np.array(metrics["confusion_matrix"]).ravel()
        row.update({"cm_tn": tn, "cm_fp": fp, "cm_fn": fn, "cm_tp": tp})
        for name, (low, high) in metrics["bootstrap_ci_95"].items():
            row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = low, high
        for slice_name, values in metrics["slice_metrics"].items():
            row[f"slice_{slice_name}_n"] = values["count"]
            row[f"slice_{slice_name}_macro_f1"] = values["macro_f1"]
            row[f"slice_{slice_name}_error_rate"] = values["error_rate"]
        row["hardware"] = metrics["hardware"].get("gpu") or metrics["hardware"]["cpu"]
        rows.append(row)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        seed = config["seed"]
        set_seed(seed)
        rng = np.random.default_rng(seed)
        data_cfg, pre_cfg, train_cfg, eval_cfg = config["data"], config["preprocessing"], config["training"], config["evaluation"]
        hardware = hardware_info()
        print(f"Run {run_id}{' (smoke test)' if args.smoke else ''}")
        print(f"Hardware: {json.dumps(hardware)}")
        print(f"Config: {json.dumps(config)}")

        folder = ensure_dataset(data_cfg)
        all_train_texts, all_train_labels, dropped_train = read_split(folder, data_cfg["dataset"], "train")
        all_test_texts, all_test_labels, dropped_test = read_split(folder, data_cfg["dataset"], "test")
        print(f"Raw rows kept: train={len(all_train_texts):,} test={len(all_test_texts):,} | dropped train={dropped_train} test={dropped_test}")

        train_pool = stratified_sample(all_train_labels, data_cfg["train_limit"], rng)
        n_val = int(len(train_pool) * data_cfg["val_fraction"])
        val_idx, train_idx = train_pool[:n_val], train_pool[n_val:]
        test_idx = stratified_sample(all_test_labels, data_cfg["test_limit"], rng)
        raw = {
            "train": ([all_train_texts[i] for i in train_idx], [all_train_labels[i] for i in train_idx]),
            "val": ([all_train_texts[i] for i in val_idx], [all_train_labels[i] for i in val_idx]),
            "test": ([all_test_texts[i] for i in test_idx], [all_test_labels[i] for i in test_idx]),
        }
        (dirs["outputs"] / "split_indices.json").write_text(json.dumps({"dataset": data_cfg["dataset"], "seed": seed, "train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}), encoding="utf-8")
        del all_train_texts, all_test_texts

        preprocess = make_preprocessor(pre_cfg)
        start = time.perf_counter()
        tokens = {split: [preprocess(text) for text in texts] for split, (texts, _) in raw.items()}
        print(f"Preprocessed {sum(len(v) for v in tokens.values()):,} reviews in {time.perf_counter() - start:.1f}s")
        eda = run_eda(raw, tokens, {"train": dropped_train, "test": dropped_test}, pre_cfg["max_length"], dirs["outputs"])
        for split in raw:
            print(f"{split}: {eda[split]['count']:,} reviews | classes {eda[split]['class_counts']} | median words {eda[split]['raw_word_length']['median']:.0f} | truncated {eda[split]['fraction_truncated_at_max_length']:.1%}")

        vocab = build_vocab(tokens["train"], pre_cfg["min_frequency"])
        (dirs["checkpoints"] / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        print(f"Vocabulary size: {len(vocab):,} (min frequency {pre_cfg['min_frequency']})")
        loaders = {
            split: DataLoader(ReviewDataset(tokens[split], raw[split][1], vocab, pre_cfg["max_length"]), batch_size=train_cfg["batch_size"], shuffle=(split == "train"))
            for split in raw
        }

        negation_words = set(eval_cfg["negation_words"])
        test_texts = raw["test"][0]
        has_negation = np.array([bool(negation_words & set(TOKEN_RE.findall(NEGATION_RE.sub(" not", text.lower())))) for text in test_texts])
        token_lengths = np.array([len(t) for t in tokens["test"]])
        slices = slice_masks(token_lengths, has_negation, eval_cfg)

        results: Dict[str, dict] = {}
        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        histories: Dict[str, List[dict]] = {}
        predictions: Dict[str, np.ndarray] = {}
        test_labels = None
        for name, spec in config["models"].items():
            set_seed(seed)
            model = build_model(len(vocab), spec)
            parameter_count = sum(p.numel() for p in model.parameters())
            print(f"\nTraining {name} ({spec}) with {parameter_count:,} parameters on {hardware.get('gpu') or hardware['cpu']}...")
            model, stats = train_model(model, loaders["train"], loaders["val"], train_cfg, dirs["checkpoints"] / f"{name}.pt")
            test_labels, probabilities, inference_eps = predict(model, loaders["test"])
            metrics = evaluate(name, test_labels, probabilities, slices, eval_cfg, seed)
            metrics.update({
                "parameter_count": parameter_count,
                "best_epoch": stats["best_epoch"],
                "epochs_run": stats["epochs_run"],
                "training_seconds": stats["training_seconds"],
                "train_examples_per_second": stats["train_examples_per_second"],
                "inference_examples_per_second": inference_eps,
                "gpu_peak_allocated_mb": stats["gpu_peak_allocated_mb"],
                "cpu_peak_rss_mb_process": stats["cpu_peak_rss_mb_process"],
                "hardware": hardware,
                "checkpoint": relative(dirs["checkpoints"] / f"{name}.pt"),
            })
            results[name] = metrics
            curves[name] = (test_labels, probabilities)
            histories[name] = stats["history"]
            predictions[name] = (probabilities >= eval_cfg["threshold"]).astype(int)
            with (dirs["outputs"] / f"predictions_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["test_index", "label", "probability_positive", "prediction"])
                writer.writerows(zip(range(len(test_labels)), test_labels.tolist(), probabilities.round(6).tolist(), predictions[name].tolist()))
            print(f"{name}: test acc={metrics['accuracy']:.4f} | macro-F1={metrics['f1_macro']:.4f} | MCC={metrics['mcc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | ECE={metrics['ece']:.4f}")

        names = list(results)
        baseline = names[0]
        for name in names[1:]:
            results[name]["mcnemar_vs_baseline"] = mcnemar_test(predictions[baseline], predictions[name], test_labels)
            print(f"McNemar {baseline} vs {name}: p={results[name]['mcnemar_vs_baseline']['p_value']:.4g}")

        review_model = max(results, key=lambda n: results[n]["f1_macro"]) if eval_cfg["error_review_model"] == "best" else eval_cfg["error_review_model"]
        chosen, worst_slice = select_error_candidates(test_labels, curves[review_model][1], eval_cfg["threshold"], slices, results[review_model]["slice_metrics"], test_texts)
        write_error_review(chosen, review_model, dirs["outputs"], base_dir / "failure_analysis.md")
        print(f"Error review: {len(chosen)} errors from {review_model} (worst slice: {worst_slice})")

        plot_results(results, curves, histories, dirs["outputs"])
        for name in results:
            results[name]["history"] = histories[name]
        (dirs["outputs"] / "metrics.json").write_text(json.dumps({"run_id": run_id, "eda": eda, "results": results}, indent=2), encoding="utf-8")
        write_metrics_report(base_dir / "metrics_report.csv", results)

        write_manifest(
            dirs["manifests"] / f"manifest_{run_id}.json",
            run_id,
            config,
            {
                "raw_log": relative(log_path),
                "checkpoints": {name: results[name]["checkpoint"] for name in results},
                "vocab": relative(dirs["checkpoints"] / "vocab.json"),
                "metrics_report": relative(base_dir / "metrics_report.csv"),
                "metrics_json": relative(dirs["outputs"] / "metrics.json"),
                "split_indices": relative(dirs["outputs"] / "split_indices.json"),
                "error_review_model": review_model,
            },
        )
        print(f"\nDone. Metrics: {relative(base_dir / 'metrics_report.csv')} | raw log: {relative(log_path)}")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
