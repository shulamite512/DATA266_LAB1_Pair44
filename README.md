# DATA266 Lab 1 (Pair 44): LLM pretraining, sentiment classification, CycleGAN

Repository link: https://github.com/shulamite512/DATA266_LAB1_Pair44

- **Task 1:** character-level GPT on TinyStories, built from scratch.
- **Task 2:** Yelp polarity sentiment classification.
- **Task 3:** CycleGAN translation between Monet paintings and photos.

## Team

| Member | Folder name | Tasks currently in this repository |
|---|---|---|
| Chelsi Shulamite Elthuri | `shulamite` | Task 1, Task 2, Task 3 |
| Parth Patel | `parth` | Task 1, Task 2, Task 3 |

Each member works in a named folder inside each task folder. `task1_llm/member_1/` and `task2_sentiment/member_1/` hold an earlier Task 1 and Task 2 upload from the team's first commits (September 28) and are kept unchanged.

## Repository structure

```text
README.md  requirements.txt  .gitignore  smoke_test.py
task1_llm/
  data/README.md              how to get TinyStories (data not committed)
  member_1/                   earlier upload (src/train_char_gpt.py, outputs/)
  shulamite/                  src/ (task1_char_gpt.ipynb, train_char_gpt.py, char_gpt_config.json),
                              checkpoints/best.pt, outputs/, logs/, manifests/,
                              metrics_report.csv, results.md, failure_analysis.md
  parth/
    src/                      task1.ipynb and the scripts used for the reported run
    config/                   model.yaml, train.yaml
    checkpoints/best.pt       final model (committed)
    outputs/ figures/ logs/   samples, eval_results.json, loss curve, run history
    metrics_report.csv  results.md  failure_analysis.md
task2_sentiment/
  data/README.md              how to get Yelp polarity (data not committed)
  member_1/                   earlier upload (src/train_sentiment.py, outputs/)
  shulamite/                  src/ (task2_sentiment.ipynb, train_sentiment.py, sentiment_config.json),
                              checkpoints/ (baseline_mean.pt, experimental_cnn.pt, experimental_gru.pt, vocab.json),
                              outputs/, logs/, manifests/, metrics_report.csv, results.md, failure_analysis.md
  parth/
    src/                      task2.ipynb and the scripts used for the reported runs
    config/train.yaml
    checkpoints/              logreg_tfidf.joblib, best_neural.pt, best_neural_cnn.pt (committed)
    outputs/ figures/ logs/   metrics, predictions, error reviews, slices, McNemar, plots, logs
    metrics_report.csv  results.md  failure_analysis.md
task3_gan/
  data/README.md              how to get the Monet and photo images (data not committed)
  parth/
    src/Task3_Parth_CycleGAN_Final.ipynb   final notebook, with saved outputs
    configs/                  baseline.yaml, exp05_5090_baseline_control.yaml
    checkpoints/<EXP-05 run>/final_generators.pt   final model (committed)
    outputs/                  runs/, plots/, eval/, checkpoint_sweep/
    evaluation/campaign_reports/   internal-evaluator campaign reports
    exp05_control/            EXP-05 run report, checkpoint-sweep report, console log, run state
    submission.csv  full_metrics_report.csv  metrics_report.csv  results.md  failure_analysis.md
    experiments.csv  checkpoint_sweep_exp05.csv  EXP05_PROVENANCE.csv
  shulamite/
    src/                      train_cyclegan.py (all runs defined here), evaluate_cyclegan.py, advanced_metrics.py,
                              plot_training.py, diff_augment.py, Part3_Evaluation_Script.ipynb, requirements_freeze.txt
    evaluate_local.py         evaluation entry point
    manifest.json             environment, per-run config, checkpoint -> Kaggle result map
    submission.csv  metrics_report.csv  full_metrics_report.csv   run 1
    logs/ outputs/            run 1 (raw log, per-epoch samples, plots, evaluation)
    run2/                     final run: checkpoints/cyclegan_epoch_95.pt (Git LFS), logs/, outputs/ (pred_A2B/,
                              pred_B2A/, evaluation/, plots/), submission.csv, metrics_report.csv, full_metrics_report.csv
    run3/ run4/ run4_first_attempt_stopped_epoch35/   stopped runs: raw logs and per-epoch outputs only
reproducibility/              Parth's runs
  raw_logs/                   unedited training and evaluation logs (Task 2 in task2_parth/)
  manifests/                  environment, config, data split and checkpoint map per run, pip freezes
tests/                        CPU tests for Parth's Task 1 and Task 2 code (no dataset needed)
report/                       final team report (not yet added)
```

## Datasets (Google Drive)

Datasets are too large for GitHub, so they are shared as one zip on Google Drive (read access enabled):

**Dataset zip:** [datasets.zip on Google Drive](https://drive.google.com/file/d/1Elz-CSlEQwb5e6X7R8mPWsrb1negSjpQ/view?usp=drive_link) (718 MB)

Unzip it into the repo root so the folders land here:

```
task1_llm/data/TinyStories-train.txt, TinyStories-valid.txt
task2_sentiment/data/yelp_review_polarity_csv/train.csv, test.csv
task3_gan/data/monet_jpg/, photo_jpg/
```

If the Task 1 or Task 2 data is missing, Chelsi's scripts download it from the original sources
(Hugging Face TinyStories, fast.ai Yelp Polarity mirror) on the first run.

How Parth's code finds the data (details in each `task*/data/README.md`):

- **Task 1:** with `data_file: null`, the loader uses the first `.txt`/`.jsonl`/`.json` file directly in `task1_llm/data/` whose name contains "train" (and not "valid"), so with the zip layout it picks `TinyStories-train.txt`. Parth's reported run used `TinyStoriesV2-GPT4-train.txt`.
- **Task 2:** the loader reads only `task2_sentiment/data/train.csv` and `task2_sentiment/data/test.csv`, not the zip's `yelp_review_polarity_csv/` subfolder. If either file is missing it loads `fancyzhx/yelp_polarity` (then `yelp_polarity`) from Hugging Face, which is the source the reported runs used. To use the zip copy, copy the two CSVs from `yelp_review_polarity_csv/` up into `task2_sentiment/data/`.
- **Task 3:** the notebook expects `task3_gan/data/monet_jpg/` (300 paintings) and `task3_gan/data/photo_jpg/` (7,038 photos), the same layout as the zip.

## Setup

Python 3.11+ and an NVIDIA GPU (the scripts fall back to CPU, but full runs are slow there). An RTX 5090 needs a CUDA 12.8+ build of PyTorch.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Run every command from the repository root.

## Smoke tests

Each member's code has its own smoke test.

**Parth (`parth`), Tasks 1 and 2:** one command, CPU only, no dataset needed:

```bash
python smoke_test.py
```

`smoke_test.py` runs the tests in `tests/` against Parth's Task 1 and Task 2 code only, each task in its own Python process, because both tasks have modules named `dataset`, `tokenizer` and `utils`. It does not test Chelsi's code or Task 3.

**Chelsi (`shulamite`), Tasks 1 and 2:** trains a tiny Task 1 model end to end (about 1 minute; downloads TinyStories-valid if needed):

```bash
python task1_llm/shulamite/src/train_char_gpt.py --smoke
```

The output goes to `task1_llm/shulamite/smoke_test/` (metrics, checkpoint, plots, raw log, manifest).
`python task2_sentiment/shulamite/src/train_sentiment.py --smoke` does the same for Task 2.

## Chelsi Shulamite Elthuri (`shulamite`): Tasks 1 and 2

Each run is config-driven. Hyperparameters live in the JSON next to the script, not in the code.

| Task | Command | Config | Notebook (results inline) |
|---|---|---|---|
| 1. Char-level GPT | `python task1_llm/shulamite/src/train_char_gpt.py` | `char_gpt_config.json` | `task1_char_gpt.ipynb` |
| 2. Sentiment (Yelp Polarity) | `python task2_sentiment/shulamite/src/train_sentiment.py` | `sentiment_config.json` | `task2_sentiment.ipynb` |

Seeds are fixed (266) and every run saves its exact train/val/test split to `outputs/split_indices.json`.

Where results live:

```
<task>/shulamite/
├── src/                  code, config, notebook (with outputs)
├── checkpoints/          trained weights (Task 1: best.pt, Task 2: one .pt per model + vocab.json)
├── outputs/              plots, samples/predictions, metrics.json, split indices
├── logs/                 raw training logs, unedited (train_<run_id>.log)
├── manifests/            package versions, hardware, config, checkpoint -> result mapping
├── metrics_report.csv    every required metric for the task
├── failure_analysis.md   failure / error analysis
└── results.md            architecture + hyperparameter justification and results
```

## Chelsi Shulamite Elthuri (`shulamite`): Task 3, CycleGAN

All runs are defined in the `RUNS` dict in `task3_gan/shulamite/src/train_cyclegan.py` and selected with `--run`. Training needs the images in `task3_gan/data/` (see its `README.md`); evaluation runs automatically after training.

| Command | What it does |
|---|---|
| `python task3_gan/shulamite/src/train_cyclegan.py --run smoke` | smoke test: 2 epochs on 64 photos, writes to `task3_gan/shulamite/smoke/` (git-ignored) |
| `python task3_gan/shulamite/src/train_cyclegan.py --run run2` | reproduces the final run (overwrites `task3_gan/shulamite/run2/`) |
| `python task3_gan/shulamite/evaluate_local.py --run run2` | re-evaluates the committed checkpoint (overwrites `run2/outputs/`) |

- **Final model:** run2, checkpoint `task3_gan/shulamite/run2/checkpoints/cyclegan_epoch_95.pt`, stored with Git LFS (run `git lfs pull` after cloning if the file is a small pointer).
- **Kaggle-style result (class evaluation notebook):** run2 FID 96.0077, MiFID 0.40847 (`run2/submission.csv`); run1 FID 104.1418, MiFID 0.41564 (`submission.csv`).
- **Kaggle leaderboard (run2 submission, team PairProgramming_Team_44):** public score −48.2080, rank 17.
- **Human audit (run2, 30 fixed samples, 2 raters, 1–5 scale):** mean score 4.10, Cohen's kappa −0.194 (exact agreement 3.3%, within ±1 point 86.7%); ratings in `run2/outputs/evaluation/human_audit_manifest.json`, per-criterion statistics in `human_audit_summary.json`.
- **All metrics:** `run2/full_metrics_report.csv` (run1: `full_metrics_report.csv`); environment and checkpoint map in `manifest.json`.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB), Windows 11, Python 3.12.10, torch 2.6.0+cu124.
- Only the selected checkpoint is committed. run3 (stopped at epoch 50) and run4 (interrupted at epoch 15) keep their raw logs and per-epoch outputs.
- The folder was named `member_1/` during training, so the unedited logs, `manifest.json` and metrics files still refer to `member_1/...` paths; those paths now live under `task3_gan/shulamite/`.

## Parth Patel (`parth`): Task 1, character-level GPT

- **Implementation:** `task1_llm/parth/src/task1.ipynb` (self-contained), with the scripts used for the reported run in `task1_llm/parth/src/` (`model.py` holds the hand-written attention).
- **Best checkpoint (committed):** `task1_llm/parth/checkpoints/best.pt`, epoch 10, 3,233,792 parameters.
- **Results:** `task1_llm/parth/metrics_report.csv`, the loss curve `figures/task1_loss_curve.png`, the generated samples `outputs/generated_samples.txt`, plus `results.md` and `failure_analysis.md` in `task1_llm/parth/`.
- **Headline:** validation cross-entropy 0.739443 nats/char, perplexity 2.094769, 1.066791 bits/char, top-1 next-character accuracy 0.765526.

Demo, text generation from the committed checkpoint (no dataset needed):

```bash
python task1_llm/parth/src/generate.py \
  --checkpoint task1_llm/parth/checkpoints/best.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 300 \
  --output task1_llm/parth/outputs/demo/demo_samples.txt
```

Always pass `--output`: the default output path is the preserved `outputs/generated_samples.txt`, which the command would overwrite.

## Parth Patel (`parth`): Task 2, Yelp polarity sentiment

- **Models:** TF-IDF + logistic regression (`logreg`, baseline), embeddings + mean pooling + MLP (`neural`, MeanPoolMLP) and embeddings + TextCNN (`neural_cnn`). All embeddings are learned from scratch.
- **Implementation:** `task2_sentiment/parth/src/task2.ipynb`, with the scripts used for the reported runs in `task2_sentiment/parth/src/` and the config in `config/train.yaml`.
- **Checkpoints (committed):** `task2_sentiment/parth/checkpoints/logreg_tfidf.joblib`, `best_neural.pt`, `best_neural_cnn.pt`.
- **Results:** `task2_sentiment/parth/metrics_report.csv`, per-model detail in `outputs/`, figures (loss curves, confusion matrices, reliability plots, EDA) in `figures/`, plus `results.md` and `failure_analysis.md` in `task2_sentiment/parth/`.
- **Headline (test set, 38,000 reviews):** accuracy 0.9463 / 0.9313 / 0.9379, ROC-AUC 0.9876 / 0.9803 / 0.9848, ECE 0.0321 / 0.0048 / 0.0059 for `logreg` / `neural` / `neural_cnn`.

Demo, prediction for one review (prints the result and writes no file; no dataset needed):

```bash
python task2_sentiment/parth/src/predict.py \
  --config task2_sentiment/parth/config/train.yaml \
  --model neural_cnn \
  --text "The food was excellent but the service was slow."
```

`--model` also accepts `logreg` and `neural`. Do not use `train.py` or `evaluate.py` for a demo: they write into `task2_sentiment/parth/` and would replace the reported files.

## Parth Patel (`parth`): Task 3, CycleGAN, Monet and photos

- **Implementation:** `task3_gan/parth/src/Task3_Parth_CycleGAN_Final.ipynb`, executed, with saved outputs.
- **Final model:** EXP-05, run `task3_parth_exp05_5090_baseline_control_20260930-145628`, checkpoint `task3_gan/parth/checkpoints/task3_parth_exp05_5090_baseline_control_20260930-145628/final_generators.pt` (committed).
- **Course evaluator result:** submission FID 103.0564, MiFID 0.41867, combined 51.7375 (lower is better), in `task3_gan/parth/submission.csv`.
- **Results:** `task3_gan/parth/full_metrics_report.csv`, `metrics_report.csv`, training plots in `outputs/plots/`, the checkpoint sweep in `checkpoint_sweep_exp05.csv`, plus `results.md` and `failure_analysis.md` in `task3_gan/parth/`.

The internal (local) evaluator and the course evaluator use different pipelines, image counts and reference sets, so their FID values must not be compared directly. Later internal-only campaign runs (EXP-06 to EXP-11) appear in `evaluation/campaign_reports/`; they are not part of the reported results.

**Evidence kept outside this repository.** The executed course-evaluator notebooks and two run-control logs (`exp05_control/watcher.log`, `exp05_control/evaluation.log`) contain local machine paths, so they are not committed. They are preserved unedited in the original project folder. The course result in this repository is established by `submission.csv`, `exp05_control/EXP-05_RESULT.md`, `exp05_control/EXP-05_CHECKPOINT_SWEEP.md`, `outputs/checkpoint_sweep/.../course_metrics.json` and the SHA-256 records in `EXP05_PROVENANCE.csv`. Some documents still name these files as sources.

Demo, translation with the committed checkpoint (there is no standalone script):

1. Work in a copy of the notebook, so the executed original keeps its saved outputs.
2. Set `TASK3_RUN_MODE=none` before starting the kernel. The default, `smoke`, trains a small model.
3. Run the setup and definition cells through section 14 (checkpoint loading in section 13, `translate` in section 14).
4. Load the checkpoint with `load_generators_checkpoint(...)`.
5. Call `translate(...)` on a few input images.
6. Write the outputs to a scratch or demo folder.

The notebook needs the images in `task3_gan/data/` (see its `README.md`). Skip the cells that read `outputs/pred_A2B/`, `outputs/pred_B2A/`, `outputs/samples/` or the executed evaluator notebook (the prediction check in section 14, the course-evaluator comparison, and the figures in section 19); those files are not in this repository, and those cells already show their saved outputs.

## Notebook output status

| Member | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Chelsi (`shulamite`) | `task1_char_gpt.ipynb`, with outputs | `task2_sentiment.ipynb`, with outputs | scripts in `src/`; no notebook with outputs yet |
| Parth (`parth`) | `task1.ipynb`, with outputs | `task2.ipynb`, with outputs | `Task3_Parth_CycleGAN_Final.ipynb`, with outputs |

Parth's Task 1 and Task 2 notebooks carry the saved outputs of a full end-to-end run on the RTX 5090 (October 2), which retrained the models. The reported numbers above still come from the committed September 25 checkpoints; each notebook's "Re-evaluating the reported checkpoint(s)" section re-evaluates them and reproduces the reported metrics, and an "About the saved outputs" note in each notebook explains this.

## Parth's reproducibility notes

- **Environment:** root `requirements.txt`; exact package versions per run are in the manifests and pip-freeze files under `reproducibility/manifests/`.
- **Raw logs:** `reproducibility/raw_logs/` holds the unedited training and evaluation logs of every reported run; Task 2's logs are also kept under `task2_sentiment/parth/logs/`.
- **Checkpoints:** the final models listed above are committed, and each `results.md` records their SHA-256.
- **Seed:** all of Parth's runs use seed 8503 (Chelsi's runs use 266). GPU training uses cuDNN autotuning, so reruns match closely but not bit for bit.
- **Tests:** `python smoke_test.py` (Parth's Tasks 1 and 2).

The reported Task 1 and Task 2 runs were produced with these commands (they need the datasets and write into the member folders, replacing the reported files):

```bash
python task1_llm/parth/src/train.py --config task1_llm/parth/config/train.yaml
python task1_llm/parth/src/evaluate.py --checkpoint task1_llm/parth/checkpoints/best.pt --config task1_llm/parth/config/train.yaml
python task2_sentiment/parth/src/train.py --config task2_sentiment/parth/config/train.yaml --model logreg   # then neural, neural_cnn
python task2_sentiment/parth/src/evaluate.py --config task2_sentiment/parth/config/train.yaml --model logreg
```

The notebooks can also be run with *Restart & Run All*; their run settings come from environment variables (`TASK1_RUN_MODE`, `TASK2_RUN_MODE`: `smoke` or `full`, default `full`; `TASK3_RUN_MODE`: `none`, `smoke` or `full`, default `smoke`). Task 1 and Task 2 notebook runs write to `task*/parth/notebook_runs/<run tag>/` and leave the reported files untouched. The Task 3 training scripts (`src/*.py`) and the local evaluator are not in this repository; the notebook contains the full training implementation.

Hardware used for Parth's reported runs:

| Task | Hardware and software |
|---|---|
| 1 | NVIDIA RTX 5090, Linux under WSL2, Python 3.11.13, torch 2.7.1+cu128 |
| 2 | Same machine; `logreg` on the CPU (24 cores), neural models on the RTX 5090 |
| 3 | NVIDIA RTX 5090, Windows 11, Python 3.12.10, torch 2.11.0+cu130, torchvision 0.26.0+cu130 |

## Report

The final report will be added as `report/DATA266_Lab1_Report_Team_44.pdf`. It is not in the repository yet.

## Before final submission / demo

This section lists open items and will be updated or removed before final submission.

- Chelsi's Task 3: `results.md`, `failure_analysis.md` and a notebook with saved outputs.
- Team architecture and hyperparameter comparison, and team best-model comparison.
- Ownership statement.
- Final team report in `report/`.
- Parth's Task 3 prediction folders (`outputs/pred_A2B/`, `outputs/pred_B2A/`), local evaluator and reference statistics, restored if recoverable.
- Task 3 Kaggle upload of `submission.csv`, with the leaderboard score and rank recorded.
- Task 3 blinded human audit (30 samples, 2 raters, agreement).
- Final demo rehearsal.
