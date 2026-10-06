# Task 2 data: Yelp Review Polarity

The dataset is not committed. The team's shared [datasets.zip on Google Drive](https://drive.google.com/file/d/1Elz-CSlEQwb5e6X7R8mPWsrb1negSjpQ/view?usp=drive_link) (read access, see the root `README.md`) unpacks the CSVs into a subfolder:

```text
task2_sentiment/data/yelp_review_polarity_csv/train.csv
task2_sentiment/data/yelp_review_polarity_csv/test.csv
```

Parth's code (`task2_sentiment/parth/`) looks for the CSVs directly in this folder, not in that subfolder. Either source works, and local files take priority:

1. Local CSVs in this folder: `task2_sentiment/data/train.csv` and `task2_sentiment/data/test.csv`. The original headerless format (labels 1 = negative, 2 = positive) and a headed `label,text` format (labels 0/1) are both handled. To use the zip copy, copy the two files from `yelp_review_polarity_csv/` up into this folder.
2. Hugging Face: if either local file is missing, the code calls `datasets.load_dataset("fancyzhx/yelp_polarity")`, falling back to `yelp_polarity`. The reported runs used `fancyzhx/yelp_polarity`.

To download once and save the shared CSVs here with Parth's code (run from the repository root):

```bash
python task2_sentiment/parth/src/dataset.py --config task2_sentiment/parth/config/train.yaml --export_csv
```

Labels are mapped to 0 = negative, 1 = positive.
