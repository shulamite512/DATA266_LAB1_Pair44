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

The three-block, four-head design is large enough to represent short-range syntax and character interactions while remaining practical to train on a laptop GPU. Four heads allow different attention patterns to be learned at the same time; the 128-character context covers several short-story clauses. Pre-LayerNorm and residual connections make optimization more stable, while the 2x GELU feed-forward layer supplies nonlinear capacity. All attention, masking, normalization, embeddings, and the language-model head are implemented directly in the source code; no prebuilt Transformer or attention module is used.

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

AdamW was selected for stable updates with mild weight decay. The warm-up protects the randomly initialized model from large early updates, and cosine decay reduces the learning rate as the model approaches convergence. Gradient clipping at 1.0 limits occasional unstable updates without hiding the raw gradient statistics. Twenty epochs were run so the model exceeded the ten-epoch requirement; the best checkpoint was retained using validation performance.

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

The validation cross-entropy is 1.0939 and the training value is 1.1844 with dropout enabled, giving a small negative reported gap (-0.0906). This is not evidence that validation is intrinsically easier: training includes dropout noise, while validation is evaluated without dropout. There are zero recorded loss spikes and NaN steps, indicating stable optimization; the pre-clip gradient norm was 0.503 on average and 1.410 at maximum. Greedy decoding is highly repetitive, while temperature sampling increases diversity and removes repeated 4-grams but produces more grammar and coherence errors. The model has learned common TinyStories character patterns, yet its limited capacity and character-level objective do not give it dependable long-range story planning.

## Limitations and next steps

The run uses a 128-character context and a relatively small 257,869-parameter model, so long dependencies are truncated and rare words are difficult to represent. The character vocabulary also makes spelling errors expensive because a whole word must be constructed one character at a time. The next experiment should compare repetition-penalty or constrained decoding, a longer context, and a modestly larger model while holding the split and evaluation prompts fixed. Epoch 7 timing should also be rerun without the recorded laptop-sleep interruption before making throughput claims.

## Failure analysis

See failure_analysis.md.
