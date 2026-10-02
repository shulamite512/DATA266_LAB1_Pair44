# DATA266 Lab 1: LLM pretraining, sentiment classification, CycleGAN

Repository link: _(add GitHub URL)_

| Member | Folder name |
|---|---|
| _(name)_ | `shulamite` |
| _(name)_ | _(folder)_ |

## Datasets (Google Drive)

Datasets are too large for GitHub, so they are shared as one zip on Google Drive (read access enabled):

**Dataset zip:** _(paste Google Drive link here)_

Unzip it into the repo root so the folders land here:

```
task1_llm/data/TinyStories-train.txt, TinyStories-valid.txt
task2_sentiment/data/yelp_review_polarity_csv/train.csv, test.csv
task3_gan/data/monet_jpg/, photo_jpg/
```

If the Task 1 or Task 2 data is missing, the scripts download it from the original sources
(Hugging Face TinyStories, fast.ai Yelp Polarity mirror) on the first run.

## Setup

Python 3.11+ and an NVIDIA GPU (the scripts fall back to CPU, but full runs are slow there).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Smoke test (one command)

Trains a tiny Task 1 model end to end (about 1 minute; downloads TinyStories-valid if needed):

```bash
python task1_llm/shulamite/src/train_char_gpt.py --smoke
```

The output goes to `task1_llm/shulamite/smoke_test/` (metrics, checkpoint, plots, raw log, manifest).
`python task2_sentiment/shulamite/src/train_sentiment.py --smoke` does the same for Task 2.

## Reproducing a member's full run

Each run is config-driven. Hyperparameters live in the JSON next to the script, not in the code.

| Task | Command | Config | Notebook (results inline) |
|---|---|---|---|
| 1. Char-level GPT | `python task1_llm/<member>/src/train_char_gpt.py` | `char_gpt_config.json` | `task1_char_gpt.ipynb` |
| 2. Sentiment (Yelp Polarity) | `python task2_sentiment/<member>/src/train_sentiment.py` | `sentiment_config.json` | `task2_sentiment.ipynb` |
| 3. CycleGAN | `python task3_gan/<member>/src/train_cyclegan.py` | | |

Seeds are fixed (266) and every run saves its exact train/val/test split to `outputs/split_indices.json`.

## Where results live (per member, per task)

```
<task>/<member>/
├── src/                  code, config, notebook (with outputs)
├── checkpoints/          trained weights (Task 1: best.pt, Task 2: one .pt per model + vocab.json)
├── outputs/              plots, samples/predictions, metrics.json, split indices
├── logs/                 raw training logs, unedited (train_<run_id>.log)
├── manifests/            package versions, hardware, config, checkpoint -> result mapping
├── metrics_report.csv    every required metric for the task
├── failure_analysis.md   failure / error analysis
└── results.md            architecture + hyperparameter justification and results
```

The final report is in `report/DATA266_Lab1_Report_Team_[Team Number].pdf`.
