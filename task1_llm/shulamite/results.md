# Task 1: Character-level GPT on TinyStories

## Data

- Source: TinyStories-train.txt; first 150,000 stories shuffled with seed 266 -> train 100,000 / val 10,000 stories (indices in outputs/split_indices.json)
- Character-level vocabulary built from train only: 109 symbols (own char_to_idx / idx_to_char; unseen val chars -> UNK)
- Input/target: fixed windows of 128 chars, target = input shifted by one; train stride 128

## Model (from scratch: no nn.Transformer / nn.MultiheadAttention)

| Setting | Value |
|---|---|
| Layers / heads / embedding dim | 3 / 4 / 96 |
| FFN hidden | 192 (2x, GELU) |
| Context length | 128 |
| Dropout | 0.1 |
| Block | pre-LayerNorm, causal multi-head self-attention + FFN, residual connections |
| Embeddings | learned token + learned positional |
| Head | LayerNorm -> Linear to vocabulary |
| Parameters | 257,869 |

**Why this architecture and size:**

_(your explanation)_

## Training

| Setting | Value |
|---|---|
| Loss / optimizer | cross-entropy / AdamW (weight decay 0.01) |
| Peak LR | 3e-05 |
| Schedule | linear warm-up (0.25 epoch), cosine decay to 0.1x peak |
| Batch size / grad clip | 32 / 1.0 |
| Epochs | 20 run (min 10, max 20, early-stop patience 3) |
| Checkpoint | checkpoints/best.pt (epoch 20) |

**Why these hyperparameters:**

_(your explanation)_

Plots: outputs/loss_curves.png, outputs/training_dynamics.png. Per-epoch numbers: outputs/epoch_history.csv

## Results (best checkpoint)

| Metric | Value |
|---|---|
| Train cross-entropy (dropout on) | 1.1844 |
| Validation cross-entropy | 1.0939 |
| Validation perplexity | 2.986 |
| Validation bits-per-character | 1.578 |
| Generalization gap (val - train) | -0.0906 |
| Top-1 next-char accuracy (val) | 65.82% |
| Grad norm mean / max (pre-clip) | 0.503 / 1.410 |
| Loss spikes / NaN steps | 0 / 0 |
| Parameter count | 257,869 |
| Training tokens/sec (excl. epoch 7) | 251,664 |
| Total training time (est., without the sleep) | 1.97 h (epoch 7 replaced by the average epoch time; train loop only) |
| Peak GPU memory | 147 MB |
| Peak CPU RSS | 3986 MB |

> Epoch 7 includes a ~75 min laptop sleep (wall-clock timing). Its raw value is kept in the log; the throughput/time above are computed without it.

### Generation

| Decoding | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate | Gen tokens/sec |
|---|---|---|---|---|---|
| greedy | 0.088 | 0.118 | 0.137 | 0.703 | 257 |
| temp_0.8 | 0.557 | 0.939 | 0.994 | 0.000 | 237 |
| temp_1.0_top20 | 0.604 | 0.955 | 1.000 | 0.000 | 239 |

Distinct-n and repeated 4-gram rate are word-level over the generated continuation (prompt excluded). Samples: outputs/samples.json

Hardware: NVIDIA GeForce RTX 3080 Ti Laptop GPU (16.0 GB), CPU 12th Gen Intel(R) Core(TM) i9-12900HK, CUDA 12.8

## Observations

_(your interpretation: loss curves, gap, stability, greedy vs sampling)_

## Limitations and next steps

_(yours)_

## Failure analysis

See failure_analysis.md.
