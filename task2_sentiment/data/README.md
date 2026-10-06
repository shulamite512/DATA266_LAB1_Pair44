# Task 2 data: Yelp Review Polarity

The dataset is not committed. Either source works, and local files take priority:

1. Local CSVs in this folder: `train.csv` and `test.csv`. The original headerless format (labels 1 = negative, 2 = positive) and a headed `label,text` format (labels 0/1) are both handled.
2. Hugging Face: if the local files are missing, the code calls `datasets.load_dataset("fancyzhx/yelp_polarity")`, falling back to `yelp_polarity`. The reported runs used `fancyzhx/yelp_polarity`.

To download once and save the shared CSVs here (run from the repository root):

```bash
python task2_sentiment/parth/src/dataset.py --config task2_sentiment/parth/config/train.yaml --export_csv
```

Labels are mapped to 0 = negative, 1 = positive.
