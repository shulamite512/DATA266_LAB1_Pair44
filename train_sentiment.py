from __future__ import annotations

import json
import random
import re
import tarfile
import time
import urllib.request
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import binomtest
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "member_1" / "outputs"
ARCHIVE_PATH = DATA_DIR / "aclImdb_v1.tar.gz"
IMDB_DIR = DATA_DIR / "aclImdb_v1" / "aclImdb"
TRAIN_DIR = IMDB_DIR / "train"
VALID_DIR = IMDB_DIR / "test"
DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
TRAIN_LIMIT = 100000
VALID_LIMIT = 10000
MAX_LENGTH = 120
MIN_FREQUENCY = 2
BATCH_SIZE = 128
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
SEED = 266
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "this", "to", "was", "we", "were", "with", "you", "your",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_DIR.exists() and VALID_DIR.exists():
        return
    print("Downloading raw IMDB dataset...")
    urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(DATA_DIR)


def preprocess(text: str) -> List[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def read_imdb_dataset(directory: Path, limit: int) -> Tuple[List[List[str]], List[int]]:
    texts: List[List[str]] = []
    labels: List[int] = []
    per_class_limit = max(1, limit // 2)
    for label_name, label in (("neg", 0), ("pos", 1)):
        class_paths = sorted((directory / label_name).glob("*.txt"))[:per_class_limit]
        for path in class_paths:
            tokens = preprocess(path.read_text(encoding="utf-8", errors="replace"))
            if tokens:
                texts.append(tokens[:MAX_LENGTH])
                labels.append(label)
    return texts, labels


def build_vocab(texts: Sequence[Sequence[str]]) -> Dict[str, int]:
    counts = Counter(token for text in texts for token in text)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= MIN_FREQUENCY:
            vocab[token] = len(vocab)
    return vocab


class ReviewDataset(Dataset):
    def __init__(self, texts: Sequence[Sequence[str]], labels: Sequence[int], vocab: Dict[str, int]):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.inputs = torch.zeros((len(texts), MAX_LENGTH), dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [vocab.get(token, vocab["<unk>"]) for token in text[:MAX_LENGTH]]
            self.inputs[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index]


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
    def __init__(self, vocab_size: int, embedding_dim: int = 96, channels: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, channels, width) for width in (3, 4, 5))
        self.classifier = nn.Linear(channels * 3, 1)

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


def ece_score(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (probabilities <= upper if upper == 1 else probabilities < upper)
        if selected.any():
            score += selected.mean() * abs(labels[selected].mean() - probabilities[selected].mean())
    return float(score)


def bootstrap_interval(values: np.ndarray, statistic, seed: int = SEED, repetitions: int = 500) -> List[float]:
    rng = np.random.default_rng(seed)
    results = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(repetitions)]
    return [float(np.percentile(results, 2.5)), float(np.percentile(results, 97.5))]


def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader) -> Tuple[nn.Module, float, float, int]:
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    start = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(tokens), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for tokens, labels in valid_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                valid_loss += loss_fn(model(tokens), labels).item() * len(labels)
        average_val_loss = valid_loss / len(valid_loader.dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} | train_loss={total_loss / len(train_loader.dataset):.4f} | val_loss={average_val_loss:.4f}")
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping; restoring epoch {best_epoch} with val_loss={best_val_loss:.4f}.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    return model, elapsed, len(train_loader.dataset) * (epoch + 1) / elapsed, best_epoch


def predict(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for tokens, batch_labels in loader:
            probabilities.append(torch.sigmoid(model(tokens.to(DEVICE))).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probabilities)


def evaluate(name: str, model: nn.Module, loader: DataLoader, texts: Sequence[Sequence[str]], train_time: float, examples_per_second: float, best_epoch: int) -> dict:
    labels, probabilities = predict(model, loader)
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    per_slice = {}
    lengths = np.array([len(text) for text in texts])
    for slice_name, selected in {"short": lengths <= 20, "medium": (lengths > 20) & (lengths <= 60), "long": lengths > 60}.items():
        if selected.any():
            per_slice[slice_name] = {"count": int(selected.sum()), "f1": float(precision_recall_fscore_support(labels[selected], predictions[selected], average="macro", zero_division=0)[2])}
    correct = predictions == labels
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)[1]),
        "f1_macro": float(precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)[2]),
        "f1_micro": float(precision_recall_fscore_support(labels, predictions, average="micro", zero_division=0)[2]),
        "f1_weighted": float(precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)[2]),
        "precision_by_class": precision.tolist(),
        "recall_by_class": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "ece": ece_score(labels, probabilities),
        "bootstrap_ci": {
            "accuracy": bootstrap_interval(correct.astype(float), np.mean),
            "macro_f1": bootstrap_interval(np.column_stack([labels, predictions]), lambda values: precision_recall_fscore_support(values[:, 0], values[:, 1], average="macro", zero_division=0)[2]),
            "mcc": bootstrap_interval(np.column_stack([labels, predictions]), lambda values: matthews_corrcoef(values[:, 0], values[:, 1])),
        },
        "slice_metrics": per_slice,
        "training_seconds": train_time,
        "examples_per_second": examples_per_second,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }
    return metrics


def write_error_review(metrics: dict, texts: Sequence[Sequence[str]], labels: Sequence[int]) -> None:
    predictions = np.array(metrics["predictions"])
    probabilities = np.array(metrics["probabilities"])
    errors = predictions != np.array(labels)
    error_indices = set(np.where(errors)[0].tolist())
    selected = {}

    def add_group(indices, error_type, proposed_fix):
        for index in indices:
            if index in error_indices and index not in selected and len([row for row in selected.values() if row["error_type"] == error_type]) < 5:
                selected[index] = {"index": int(index), "error_type": error_type, "proposed_fix": proposed_fix, "label": int(labels[index]), "prediction": int(predictions[index]), "probability": float(probabilities[index]), "text": " ".join(texts[index])}

    confidence = np.abs(probabilities - 0.5)
    add_group(np.where((predictions == 1) & (labels == 0))[0][np.argsort(-confidence[(predictions == 1) & (labels == 0)])], "confident_false_positive", "Add negation and sarcasm-aware features.")
    add_group(np.where((predictions == 0) & (labels == 1))[0][np.argsort(-confidence[(predictions == 0) & (labels == 1)])], "confident_false_negative", "Increase context length and improve positive sentiment coverage.")
    add_group(np.argsort(confidence), "near_threshold", "Calibrate the decision threshold on a validation split.")
    lengths = np.array([len(text) for text in texts])
    slice_indices = np.where(errors & ((lengths <= 20) | (lengths > 60)))[0]
    add_group(slice_indices[np.argsort(confidence[slice_indices])], "slice_specific", "Train and report separate short- and long-review slices.")
    for index in np.where(errors)[0]:
        if len(selected) >= 20:
            break
        if index not in selected:
            selected[index] = {"index": int(index), "error_type": "additional_error", "proposed_fix": "Review this example and assign a specific failure category.", "label": int(labels[index]), "prediction": int(predictions[index]), "probability": float(probabilities[index]), "text": " ".join(texts[index])}
    rows = list(selected.values())[:20]
    (OUTPUT_DIR / "error_review_20.json").write_text(json.dumps(rows[:20], indent=2), encoding="utf-8")


def mcnemar_test(baseline: np.ndarray, experimental: np.ndarray, labels: np.ndarray) -> dict:
    baseline_correct = baseline == labels
    experimental_correct = experimental == labels
    baseline_only = int(np.sum(baseline_correct & ~experimental_correct))
    experimental_only = int(np.sum(~baseline_correct & experimental_correct))
    discordant = baseline_only + experimental_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(baseline_only, experimental_only), discordant, 0.5).pvalue)
    return {"baseline_only_correct": baseline_only, "experimental_only_correct": experimental_only, "p_value": p_value}


def main() -> None:
    set_seed()
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    ensure_dataset()
    train_texts, train_labels = read_imdb_dataset(TRAIN_DIR, TRAIN_LIMIT)
    valid_texts, valid_labels = read_imdb_dataset(VALID_DIR, VALID_LIMIT)
    print(f"Loaded {len(train_texts)} training reviews and {len(valid_texts)} validation reviews.")
    print(f"Training class counts: negative={train_labels.count(0)}, positive={train_labels.count(1)}")
    print(f"Validation class counts: negative={valid_labels.count(0)}, positive={valid_labels.count(1)}")
    vocab = build_vocab(train_texts)
    train_loader = DataLoader(ReviewDataset(train_texts, train_labels, vocab), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(ReviewDataset(valid_texts, valid_labels, vocab), batch_size=BATCH_SIZE)
    models = {
        "baseline_mean": MeanClassifier(len(vocab)),
        "experimental_cnn": CNNClassifier(len(vocab)),
        "experimental_gru": GRUClassifier(len(vocab)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name, model in models.items():
        print(f"\nTraining {name} with {sum(parameter.numel() for parameter in model.parameters())} parameters...")
        trained, train_time, examples_per_second, best_epoch = train_model(model, train_loader, valid_loader)
        metrics = evaluate(name, trained, valid_loader, valid_texts, train_time, examples_per_second, best_epoch)
        all_metrics.append(metrics)
        if name == "baseline_mean":
            write_error_review(metrics, valid_texts, valid_labels)
    baseline_predictions = np.array(all_metrics[0]["predictions"])
    for metrics in all_metrics[1:]:
        metrics["mcnemar_vs_baseline"] = mcnemar_test(baseline_predictions, np.array(metrics["predictions"]), np.array(valid_labels))
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved Task 2 metrics to {OUTPUT_DIR / 'metrics.json'}")
    print(f"Saved baseline error review to {OUTPUT_DIR / 'error_review_20.json'}")


if __name__ == "__main__":
    main()