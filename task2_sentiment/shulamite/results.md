# Task 2: Yelp Polarity sentiment classification

## Data and preprocessing

- Dataset: yelp_polarity; train 90,000 / val 10,000 (held out from train) / test 10,000; seed 266
- Class balance (train): {'negative': 44992, 'positive': 45008}; dropped rows: {'train': {'malformed_rows': 0, 'missing_or_empty_text': 0, 'invalid_label': 0}, 'test': {'malformed_rows': 0, 'missing_or_empty_text': 0, 'invalid_label': 0}}
- Review length (train, words): median 98, p90 284, max 1023; 26.2% truncated at 120 tokens
- Steps: lowercase, strip HTML, expand n't -> not, keep [a-z]+ tokens, remove 28 stopwords, Porter stemming, vocab min frequency 2
- Plots: outputs/eda.png

**Why these preprocessing choices:**

_(your explanation)_

## Models

| | baseline_mean | experimental_cnn | experimental_gru |
|---|---|---|---|
| Config | {'type': 'mean', 'embedding_dim': 64, 'hidden_dim': 64} | {'type': 'cnn', 'embedding_dim': 96, 'channels': 96, 'kernel_sizes': [3, 4, 5]} | {'type': 'gru', 'embedding_dim': 64, 'hidden_dim': 64} |
| Parameters | 2,058,241 | 3,192,193 | 2,104,065 |
| Best epoch / epochs run | 2 / 4 | 2 / 4 | 3 / 5 |
| Checkpoint | checkpoints/baseline_mean.pt | checkpoints/experimental_cnn.pt | checkpoints/experimental_gru.pt |

Shared training settings: {'batch_size': 128, 'epochs': 5, 'early_stopping_patience': 2, 'lr': 0.002}

Embeddings: learned from scratch (nn.Embedding, random init), no pretrained vectors.

### baseline_mean

**Architecture and embedding choice, and why:**

_(your explanation)_

### experimental_cnn

**Architecture and embedding choice, and why:**

_(your explanation)_

### experimental_gru

**Architecture and embedding choice, and why:**

_(your explanation)_

## Results (test set)

| Metric | baseline_mean | experimental_cnn | experimental_gru |
|---|---|---|---|
| accuracy | 0.9119 | 0.9143 | 0.9258 |
| precision_macro | 0.9119 | 0.9145 | 0.9262 |
| recall_macro | 0.9119 | 0.9143 | 0.9258 |
| f1_macro | 0.9119 | 0.9143 | 0.9258 |
| precision_micro | 0.9119 | 0.9143 | 0.9258 |
| recall_micro | 0.9119 | 0.9143 | 0.9258 |
| f1_micro | 0.9119 | 0.9143 | 0.9258 |
| precision_weighted | 0.9119 | 0.9145 | 0.9262 |
| recall_weighted | 0.9119 | 0.9143 | 0.9258 |
| f1_weighted | 0.9119 | 0.9143 | 0.9258 |
| roc_auc | 0.9705 | 0.9755 | 0.9785 |
| pr_auc | 0.9703 | 0.9755 | 0.9791 |
| mcc | 0.8238 | 0.8288 | 0.8520 |
| brier_score | 0.0646 | 0.0621 | 0.0553 |
| ece | 0.0069 | 0.0271 | 0.0165 |
| 95% CI accuracy | [0.9065, 0.9173] | [0.9089, 0.9197] | [0.9206, 0.9312] |
| 95% CI macro_f1 | [0.9065, 0.9173] | [0.9089, 0.9197] | [0.9206, 0.9312] |
| 95% CI mcc | [0.8131, 0.8346] | [0.8180, 0.8397] | [0.8417, 0.8629] |
| Confusion matrix [[TN,FP],[FN,TP]] | [[4583, 417], [464, 4536]] | [[4516, 484], [373, 4627]] | [[4710, 290], [452, 4548]] |
| McNemar vs baseline (p) | - | 0.393 | 4.69e-08 |
| slice length_short: macro-F1 / error | 0.908 / 0.085 | 0.909 / 0.083 | 0.921 / 0.074 |
| slice length_medium: macro-F1 / error | 0.926 / 0.073 | 0.927 / 0.073 | 0.942 / 0.057 |
| slice length_long: macro-F1 / error | 0.901 / 0.097 | 0.905 / 0.094 | 0.914 / 0.084 |
| slice with_negation: macro-F1 / error | 0.898 / 0.098 | 0.903 / 0.094 | 0.916 / 0.080 |
| slice without_negation: macro-F1 / error | 0.915 / 0.060 | 0.911 / 0.063 | 0.922 / 0.057 |
| Training time (s) | 9.2 | 15.5 | 15.4 |
| Train examples/sec | 39,013 | 23,296 | 29,164 |
| Peak GPU memory (MB) | 58 | 102 | 237 |

Hardware (all models): NVIDIA GeForce RTX 3080 Ti Laptop GPU (16.0 GB), CPU Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, CUDA 12.8

Plots: outputs/confusion_matrices.png, outputs/roc_pr_calibration.png, outputs/training_curves.png

## Comparison and observations

_(your interpretation: which model is better and why, what the CIs / McNemar / slices / calibration show)_

## Strengths, weaknesses, limitations

_(yours)_

## Possible improvements

_(yours)_

## Error analysis

See failure_analysis.md.
