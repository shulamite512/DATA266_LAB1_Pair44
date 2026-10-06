"""Loads Yelp Polarity from local CSVs or Hugging Face, cleans malformed rows,
runs the EDA the lab asks for, preprocesses text, and builds train/val/test
splits with robustness-slice columns. Processed splits are cached in
data_processed/ so later runs skip the slow preprocessing step."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from tokenizer import PAD_ID, TextPreprocessor, normalize_raw
from utils import (Timer, apply_smoke_overrides, config_hash, get_paths, load_config,
                   load_json, make_synthetic_df, rel, save_json, setup_logger, timestamp)


# ---------------------------------------------------------------- raw loading
def read_yelp_csv(path):
    """Read a Yelp CSV with or without a header; returns columns text, label."""
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=True)
    first = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    if "text" in first and "label" in first:  # file has a header row
        df.columns = first
        df = df.iloc[1:]
    else:  # fast.ai format: label, text with no header
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
    return df[["text", "label"]].reset_index(drop=True)


def normalize_labels(series):
    """Map labels to 0=negative, 1=positive. Accepts {0,1} or {1,2}; invalid -> NaN."""
    y = pd.to_numeric(series, errors="coerce")
    valid = set(y.dropna().unique().tolist())
    if valid <= {0, 1}:
        return y
    if valid <= {1, 2}:
        return y - 1
    # Mixed or unexpected values: keep only 1/2 style if that is the majority
    if y.isin([1, 2]).mean() >= y.isin([0, 1]).mean():
        return (y - 1).where(y.isin([1, 2]))
    return y.where(y.isin([0, 1]))


def load_raw_splits(cfg, paths, logger):
    """Prefer local CSVs in task2_sentiment/data/, otherwise download from Hugging Face."""
    train_csv, test_csv = paths["data_dir"] / "train.csv", paths["data_dir"] / "test.csv"
    if train_csv.exists() and test_csv.exists():
        logger.info(f"Loading local CSVs from {rel(paths['data_dir'])}")
        return read_yelp_csv(train_csv), read_yelp_csv(test_csv), "local_csv"
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise FileNotFoundError(
            f"No CSVs at {rel(train_csv)} / {rel(test_csv)} and `datasets` is not installed."
        ) from e
    last_err = None
    for name in cfg["data"]["hf_names"]:
        try:
            logger.info(f"Loading Hugging Face dataset '{name}'")
            ds = load_dataset(name)
            return ds["train"].to_pandas(), ds["test"].to_pandas(), f"hf:{name}"
        except Exception as e:  # try the next alias
            last_err = e
            logger.warning(f"Could not load '{name}': {e}")
    raise RuntimeError(f"Could not load Yelp Polarity from any source: {last_err}")


def clean_split(df, split, drop_duplicates):
    """Drop missing/empty text and invalid labels; report counts for the EDA summary."""
    report = {"split": split, "rows_raw": int(len(df))}
    df = df.copy()
    df["label"] = normalize_labels(df["label"])
    report["missing_text"] = int(df["text"].isna().sum())
    report["invalid_label"] = int(df["label"].isna().sum())
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].map(normalize_raw)
    empty = df["text"].str.len() == 0
    report["empty_text"] = int(empty.sum())
    df = df[~empty]
    dup_mask = df.duplicated(subset=["text"], keep=False)
    conflicting = df[dup_mask].groupby("text")["label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    report["duplicate_texts"] = int(df.duplicated(subset=["text"]).sum())
    report["conflicting_label_texts"] = len(conflicting)
    if drop_duplicates:
        # Why: only the training split is de-duplicated. The official test set is
        # left untouched so results stay comparable with other members.
        df = df[~df["text"].isin(conflicting)]
        df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report


# ---------------------------------------------------------------- slices
def _word_regex(words):
    # Why: str() guards against YAML turning an unquoted `no` into False.
    return re.compile(r"\b(?:" + "|".join(re.escape(str(w)) for w in words) + r")\b")


def add_slice_columns(df, slice_cfg):
    """Robustness slices computed on the raw text (before stopword removal)."""
    lower = df["text"].str.lower()
    neg_re = _word_regex(slice_cfg["negation_words"])
    con_re = _word_regex(slice_cfg["contrast_words"])
    df["n_words_raw"] = df["text"].str.split().str.len()
    df["length_bucket"] = np.where(
        df["n_words_raw"] <= slice_cfg["short_max_words"], "short",
        np.where(df["n_words_raw"] >= slice_cfg["long_min_words"], "long", "medium"))
    contraction_negation = lower.str.contains("n\'t", regex=False, na=False) | lower.str.contains("n’t", regex=False, na=False)
    df["has_negation"] = lower.str.contains(neg_re, regex=True, na=False) | contraction_negation
    df["has_contrast"] = lower.str.contains(con_re)
    return df


def slice_masks(df):
    """Named boolean masks used by evaluation for per-slice metrics."""
    return {
        "all": np.ones(len(df), dtype=bool),
        "length_short": (df["length_bucket"] == "short").to_numpy(),
        "length_medium": (df["length_bucket"] == "medium").to_numpy(),
        "length_long": (df["length_bucket"] == "long").to_numpy(),
        "has_negation": df["has_negation"].to_numpy(bool),
        "no_negation": ~df["has_negation"].to_numpy(bool),
        "has_contrast": df["has_contrast"].to_numpy(bool),
        "no_contrast": ~df["has_contrast"].to_numpy(bool),
    }


# ---------------------------------------------------------------- EDA
def run_eda(splits, reports, cfg, paths, source):
    """Length distribution, class balance, missing/malformed counts, train-test overlap."""
    max_len = cfg["tokenizer"]["max_length"]
    summary = {"source": source, "cleaning": reports, "splits": {}}
    for name, df in splits.items():
        counts = df["label"].value_counts().sort_index()
        summary["splits"][name] = {
            "rows": int(len(df)),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
            "positive_fraction": float(df["label"].mean()),
            "raw_words": df["n_words_raw"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict(),
            "processed_tokens": df["n_tokens"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict(),
            "fraction_truncated_at_max_length": float((df["n_tokens"] > max_len).mean()),
            "empty_after_preprocessing": int((df["n_tokens"] == 0).sum()),
            "length_bucket_counts": df["length_bucket"].value_counts().to_dict(),
            "has_negation_fraction": float(df["has_negation"].mean()),
            "has_contrast_fraction": float(df["has_contrast"].mean()),
        }
    overlap = set(splits["train"]["text"]) & set(splits["test"]["text"])
    summary["train_test_exact_overlap"] = len(overlap)
    save_json(summary, paths["outputs"] / "eda_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, df in splits.items():
        axes[0].hist(df["n_words_raw"].clip(upper=1000), bins=cfg["eda"]["length_bins"],
                     alpha=0.5, label=name, density=True)
        axes[1].hist(df["n_tokens"].clip(upper=600), bins=cfg["eda"]["length_bins"],
                     alpha=0.5, label=name, density=True)
    axes[0].set_title("Raw review length (words, clipped at 1000)")
    axes[1].set_title("Processed length (tokens, clipped at 600)")
    axes[1].axvline(max_len, color="k", ls="--", label=f"max_length={max_len}")
    for ax in axes:
        ax.set_xlabel("length")
        ax.set_ylabel("density")
        ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "eda_review_length_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(splits)
    x = np.arange(len(names))
    for lbl, off in [(0, -0.2), (1, 0.2)]:
        vals = [int((splits[n]["label"] == lbl).sum()) for n in names]
        ax.bar(x + off, vals, width=0.4, label="negative (0)" if lbl == 0 else "positive (1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("reviews")
    ax.set_title("Class balance per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "eda_class_balance.png", dpi=120)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------- prepare
def _process(df, pre, slice_cfg):
    df = df.copy()
    df["tokens"] = df["text"].map(pre.to_string)
    df["n_tokens"] = df["tokens"].str.split().str.len()
    return add_slice_columns(df, slice_cfg)


def prepare_data(cfg, paths, logger, smoke=False):
    """Return {'train','val','test'} DataFrames with columns id, text, label, tokens, slices."""
    seed = cfg["seed"]
    pre = TextPreprocessor(cfg["preprocessing"])
    if smoke:
        s = cfg["smoke"]
        train_raw = make_synthetic_df(s["n_train"], seed)
        test_raw = make_synthetic_df(s["n_test"], seed + 1)
        source = "synthetic"
    else:
        key = config_hash({k: cfg[k] for k in ["seed", "data", "preprocessing", "slices"]})
        meta_path = paths["data_processed"] / "meta.json"
        if meta_path.exists() and load_json(meta_path).get("key") == key:
            logger.info(f"Using cached processed data in {rel(paths['data_processed'])} (key={key})")
            return {s: pd.read_pickle(paths["data_processed"] / f"{s}.pkl") for s in ["train", "val", "test"]}
        train_raw, test_raw, source = load_raw_splits(cfg, paths, logger)

    train_df, r_train = clean_split(train_raw, "train", drop_duplicates=not smoke)
    test_df, r_test = clean_split(test_raw, "test", drop_duplicates=False)
    logger.info(f"Cleaning report: {r_train} | {r_test}")

    for name, df, n in [("train", train_df, cfg["data"].get("train_subset")),
                        ("test", test_df, cfg["data"].get("test_subset"))]:
        if n and not smoke and n < len(df):
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            logger.info(f"Subsampled {name} to {n} rows")
        if name == "train":
            train_df = df
        else:
            test_df = df

    # Why: validation is carved from train (stratified) so the test set is only
    # touched once, at final evaluation.
    tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=cfg["data"]["val_fraction"],
                                      random_state=seed, stratify=train_df["label"])
    splits = {"train": train_df.iloc[tr_idx], "val": train_df.iloc[va_idx], "test": test_df}

    with Timer() as t:
        for name in splits:
            splits[name] = _process(splits[name], pre, cfg["slices"]).reset_index(drop=True)
            splits[name].insert(0, "id", [f"{name}_{i}" for i in range(len(splits[name]))])
    logger.info(f"Preprocessing took {t.seconds:.1f}s")

    # Reviews that become empty after preprocessing carry no signal for training.
    n_before = len(splits["train"])
    splits["train"] = splits["train"][splits["train"]["n_tokens"] >= cfg["preprocessing"]["min_tokens"]]
    splits["train"] = splits["train"].reset_index(drop=True)
    r_train["dropped_empty_after_preprocessing"] = n_before - len(splits["train"])

    run_eda(splits, [r_train, r_test], cfg, paths, source)
    if not smoke:
        for s, df in splits.items():
            df.to_pickle(paths["data_processed"] / f"{s}.pkl")
        save_json({"key": key, "source": source, "rows": {s: len(d) for s, d in splits.items()}},
                  paths["data_processed"] / "meta.json")
    logger.info("Split sizes: " + ", ".join(f"{s}={len(d)}" for s, d in splits.items()))
    return splits


# ---------------------------------------------------------------- torch dataset
class EncodedReviews(Dataset):
    """Stores all token ids in one flat int32 array plus offsets to keep memory low."""

    def __init__(self, token_strings, labels, tokenizer, max_length):
        seqs = [tokenizer.encode(s, max_length) for s in token_strings]
        self.lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.flat = np.fromiter((i for s in seqs for i in s), dtype=np.int32, count=int(self.offsets[-1]))
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        ids = self.flat[self.offsets[i]:self.offsets[i + 1]]
        return torch.from_numpy(ids.astype(np.int64)), int(self.labels[i])


def collate_batch(batch):
    """Pad each batch only to its own longest review, and build the padding mask."""
    ids, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    return padded, padded != PAD_ID, torch.tensor(labels, dtype=torch.long)


def make_loader(dataset, batch_size, shuffle, seed, num_workers=0, pin_memory=False):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                      num_workers=num_workers, pin_memory=pin_memory, generator=gen)


def export_hf_to_csv(cfg, paths, logger):
    """Save the Hugging Face copy as task2_sentiment/data/{train,test}.csv for the team."""
    train_raw, test_raw, source = load_raw_splits(cfg, paths, logger)
    for name, df in [("train", train_raw), ("test", test_raw)]:
        df[["label", "text"]].to_csv(paths["data_dir"] / f"{name}.csv", index=False)
    logger.info(f"Exported {source} to {rel(paths['data_dir'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare Yelp Polarity data and run EDA")
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export_csv", action="store_true", help="download from HF and save CSVs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    paths = get_paths(cfg, smoke=args.smoke)
    Path(paths["data_dir"]).mkdir(parents=True, exist_ok=True)
    log = setup_logger("data", paths["logs"] / f"data_{timestamp()}.log")
    if args.export_csv:
        export_hf_to_csv(cfg, paths, log)
    prepare_data(cfg, paths, log, smoke=args.smoke)
