# Task 2 — Sentiment Classification

This folder is reserved for the sentiment classification task.

## Expected structure

- `data/` — shared raw Yelp/IMDB dataset
- `member_1/` — one member's code, outputs, metrics, and failure analysis
- `member_2/` — second member's implementation
- `report/` — combined report data

## Current status

The member 1 pipeline is implemented in `member_1/src/train_sentiment.py`. It downloads and caches the raw Stanford IMDB dataset (`aclImdb_v1.tar.gz`), preprocesses reviews, uses balanced positive and negative samples, learns embeddings from scratch, trains a mean-pooling baseline plus CNN and GRU experiments, restores the best validation checkpoint with early stopping, and saves metrics, McNemar comparisons, and a structured 20-error review file.

Run it from the project root with:

```powershell
& ".\.venv\Scripts\python.exe" task2_sentiment\member_1\src\train_sentiment.py
```

Outputs are written to `task2_sentiment/member_1/outputs/`. The PDF requires the team to review and refine the automatically generated error file so it contains five confident false positives, five confident false negatives, five near-threshold errors, and five slice-specific failures with proposed fixes.
