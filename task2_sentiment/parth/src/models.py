"""Model definitions. The neural model is a from-scratch embedding bag
classifier (learned embeddings, masked mean pooling, MLP head). The baseline
is TF-IDF features feeding logistic regression. New neural architectures are
added by writing a class and registering it in NEURAL_MODELS."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tokenizer import PAD_ID


class MeanPoolMLP(nn.Module):
    """Embedding -> masked mean over tokens -> Linear -> ReLU -> Dropout -> Linear (2 logits)."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, num_classes=2):
        super().__init__()
        # Why: embeddings are randomly initialised and learned from Yelp only;
        # the lab forbids pretrained embeddings.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)                      # (B, T, E)
        mask = attention_mask.unsqueeze(-1).to(emb.dtype)    # (B, T, 1)
        # Why: dividing by the real token count, not T, stops padding from
        # shrinking the representation of short reviews.
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        h = self.dropout(self.act(self.fc1(self.dropout(pooled))))
        return self.fc2(h)


class TextCNN(nn.Module):
    """Embedding -> parallel Conv1D branches (one per kernel size) -> ReLU ->
    masked global max pool over time -> concat -> Dropout -> Linear (2 logits).
    Same forward(input_ids, attention_mask) signature as MeanPoolMLP."""

    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, num_classes=2):
        super().__init__()
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self.min_len = max(self.kernel_sizes)
        # Why: same from-scratch, randomly initialised embeddings as MeanPoolMLP.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        # Why: each branch detects a different n-gram width (3, 4, 5 tokens), so
        # word order inside short phrases matters, unlike mean pooling.
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in self.kernel_sizes])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask):
        # Why: a width-5 kernel needs at least 5 positions, so batches of very
        # short reviews are right-padded up to the widest kernel.
        if input_ids.size(1) < self.min_len:
            extra = self.min_len - input_ids.size(1)
            input_ids = nn.functional.pad(input_ids, (0, extra), value=PAD_ID)
            attention_mask = nn.functional.pad(attention_mask, (0, extra), value=False)
        lengths = attention_mask.sum(1)                          # (B,) real tokens, padding is on the right
        x = self.embedding(input_ids).transpose(1, 2)            # (B, E, T)
        pooled = []
        for k, conv in zip(self.kernel_sizes, self.convs):
            h = self.act(conv(x))                                # (B, F, T-k+1)
            # Why: windows made only of padding still output ReLU(bias) and could
            # win the max, which would make a review's prediction depend on the
            # other reviews in its batch. Keep starts 0..L-k (start 0 if L < k).
            last = (lengths - k).clamp(min=0)
            pos = torch.arange(h.size(2), device=h.device)
            valid = pos.unsqueeze(0) <= last.unsqueeze(1)        # (B, T-k+1)
            h = h.masked_fill(~valid.unsqueeze(1), float("-inf"))
            pooled.append(h.max(dim=2).values)                   # (B, F)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


# Register additional experimental architectures here: "arch_name": ClassName.
# Each class must accept (vocab_size, **arch kwargs from YAML) and implement
# forward(input_ids, attention_mask) -> logits of shape (B, 2).
NEURAL_MODELS = {"meanpool_mlp": MeanPoolMLP, "textcnn": TextCNN}

_NON_ARCH_KEYS = {"type", "arch", "batch_size", "eval_batch_size", "learning_rate", "weight_decay",
                  "epochs", "grad_clip", "num_workers", "select_by", "log_every", "description"}


def build_neural_model(model_cfg, vocab_size):
    arch = model_cfg.get("arch", "meanpool_mlp")
    if arch not in NEURAL_MODELS:
        raise KeyError(f"Unknown arch '{arch}'. Registered: {list(NEURAL_MODELS)}")
    kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_ARCH_KEYS}
    return NEURAL_MODELS[arch](vocab_size=vocab_size, **kwargs)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_logreg(model_cfg, seed):
    """TF-IDF over the already-preprocessed token string, then logistic regression."""
    t = model_cfg["tfidf"]
    vec = TfidfVectorizer(
        lowercase=False,               # text is already lowercased in preprocessing
        token_pattern=r"(?u)\b\w+\b",  # keep 1-character tokens the preprocessor produced
        ngram_range=tuple(t["ngram_range"]),
        max_features=t["max_features"],
        min_df=t["min_df"],
        sublinear_tf=t["sublinear_tf"],
        dtype=np.float64,
    )
    clf = LogisticRegression(C=model_cfg["C"], max_iter=model_cfg["max_iter"],
                             solver=model_cfg.get("solver", "lbfgs"), random_state=seed)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def logreg_param_count(pipeline):
    clf = pipeline.named_steps["clf"]
    return int(clf.coef_.size + clf.intercept_.size)
