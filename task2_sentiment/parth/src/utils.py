"""Shared helpers for Task 2: repo-relative paths, config loading, seeding,
device selection, logging, hardware and memory reporting, and run manifests.
Every other module in src/ imports from here so behaviour stays consistent
between training, evaluation, prediction, and the smoke tests."""

import copy
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Why: cuBLAS needs this set before CUDA initialises for deterministic matmuls.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")  # Why: lab machines and CI have no display.

import numpy as np
import pandas as pd
import torch
import yaml

# src/ -> parth/ -> task2_sentiment/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

LABEL_NAMES = {0: "negative", 1: "positive"}


# ---------------------------------------------------------------- config / paths
def resolve_path(p):
    """Return an absolute path; relative paths are anchored at the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path):
    """Load the YAML config from a path given relative to cwd or repo root."""
    p = Path(path)
    if not p.exists():
        p = resolve_path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(cfg):
    """Return a copy of cfg shrunk for smoke runs (tiny data, 1-2 epochs, no workers)."""
    cfg = copy.deepcopy(cfg)
    s = cfg.get("smoke", {})
    cfg["evaluation"]["bootstrap_samples"] = s.get("bootstrap_samples", 50)
    cfg["evaluation"]["min_slice_n"] = 5
    cfg["tokenizer"]["min_freq"] = 1
    for m in cfg["models"].values():
        if m["type"] == "neural":
            m["epochs"] = s.get("epochs", 2)
            m["batch_size"] = s.get("batch_size", 32)
            m["num_workers"] = 0
        elif m["type"] == "logreg":
            m["tfidf"]["min_df"] = 1
            m["max_iter"] = 200
    return cfg


def get_paths(cfg, smoke=False):
    """Build (and create) the output directories for a real or smoke run."""
    # Why: smoke runs write to a separate folder so they can never overwrite
    # real checkpoints, metrics, or logs.
    root = resolve_path(cfg["paths"]["smoke_dir"] if smoke else cfg["paths"]["member_dir"])
    paths = {
        "root": root,
        "data_dir": resolve_path(cfg["paths"]["data_dir"]),
        "data_processed": root / "data_processed",
        "checkpoints": root / "checkpoints",
        "outputs": root / "outputs",
        "figures": root / "figures",
        "logs": root / "logs",
        "metrics_report": root / "metrics_report.csv",
    }
    for k in ["data_processed", "checkpoints", "outputs", "figures", "logs"]:
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def get_model_cfg(cfg, model_name):
    if model_name not in cfg["models"]:
        raise KeyError(f"Model '{model_name}' not in config. Available: {list(cfg['models'])}")
    return cfg["models"][model_name]


def checkpoint_paths(paths, model_name, model_type):
    """File names follow the lab spec: logreg_tfidf.joblib, best_neural.pt, latest_neural.pt."""
    if model_type == "logreg":
        return {"model": paths["checkpoints"] / f"{model_name}_tfidf.joblib"}
    return {
        "best": paths["checkpoints"] / f"best_{model_name}.pt",
        "latest": paths["checkpoints"] / f"latest_{model_name}.pt",
    }


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Why: warn_only keeps training running if an op has no deterministic kernel,
    # but the warning still lands in the raw log.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_capability": None,
        "gpu_total_mem_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            gpu_capability=f"sm_{props.major}{props.minor}",
            gpu_total_mem_gb=round(props.total_memory / 1024**3, 2),
        )
    return info


def device_label(info, device):
    """Human-readable name of the device a stage actually ran on."""
    if device.type == "cuda" and info["gpu_name"]:
        return f"{info['gpu_name']} ({info['gpu_capability']}, {info['gpu_total_mem_gb']} GB)"
    return f"CPU ({info['cpu']}, {info['cpu_count']} cores)"


# ---------------------------------------------------------------- memory / timing
def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    """GPU peak allocated by PyTorch in this phase, plus process-lifetime peak RSS."""
    gpu = None
    if device.type == "cuda":
        gpu = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
    cpu = None
    try:
        import resource  # Linux/macOS only

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Why: Linux reports KB, macOS reports bytes.
        cpu = round(rss / 1024**2, 1) if sys.platform == "darwin" else round(rss / 1024, 1)
    except ImportError:
        pass
    return {"gpu_peak_mb": gpu, "cpu_peak_rss_mb": cpu}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


# ---------------------------------------------------------------- logging / io
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(name, log_path):
    """Log to console and to a new timestamped file that is never rewritten."""
    logger = logging.getLogger(f"{name}_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def to_builtin(obj):
    """Convert numpy scalars/arrays into plain Python types for JSON and torch.save."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rel(path):
    """Repo-relative string for logs and CSVs, so no personal paths get committed."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(obj):
    return hashlib.sha256(json.dumps(to_builtin(obj), sort_keys=True).encode()).hexdigest()[:12]


def upsert_csv_row(path, row, key_cols):
    """Insert or replace one row (matched on key_cols) so reruns don't duplicate models."""
    new = pd.DataFrame([to_builtin(row)])
    if Path(path).exists():
        old = pd.read_csv(path)
        if all(k in old.columns for k in key_cols):
            mask = np.ones(len(old), dtype=bool)
            for k in key_cols:
                mask &= old[k].astype(str) == str(row[k])
            old = old[~mask]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def package_versions():
    names = ["torch", "numpy", "pandas", "scikit-learn", "scipy", "nltk", "PyYAML",
             "matplotlib", "joblib", "datasets"]
    out = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return out
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def write_manifest(paths, model_name, stage, extra):
    """Record versions, config, hardware, and checkpoint hashes for one run stage."""
    manifest = {
        "model": model_name,
        "stage": stage,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "packages": package_versions(),
        "hardware": hardware_info(),
        **extra,
    }
    # Why: timestamped like the raw logs, so an earlier run's manifest is never overwritten.
    out = paths["logs"] / f"manifest_{stage}_{model_name}_{timestamp()}.json"
    save_json(manifest, out)
    return out


# ---------------------------------------------------------------- synthetic data
_POS = ["amazing", "great", "delicious", "friendly", "excellent", "wonderful", "fresh", "perfect"]
_NEG = ["terrible", "awful", "rude", "bland", "horrible", "dirty", "slow", "disgusting"]
_NOUNS = ["food", "service", "staff", "pizza", "coffee", "place", "waiter", "price"]
_FILLER = ["we", "came", "here", "on", "a", "friday", "night", "with", "family", "and", "ordered"]


def make_synthetic_df(n, seed):
    """Tiny labelled review set for smoke runs and tests; roughly balanced, varied lengths."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = i % 2
        words = _POS if label == 1 else _NEG
        parts = []
        for _ in range(rng.randint(1, 4)):
            parts.append(f"The {rng.choice(_NOUNS)} was {rng.choice(words)}.")
        if rng.random() < 0.3:  # some long reviews so every length slice is populated
            parts.append(" ".join(rng.choices(_FILLER, k=rng.randint(60, 320))) + ".")
        if rng.random() < 0.2:  # some negated opposite words
            opp = _NEG if label == 1 else _POS
            parts.append(f"It was not {rng.choice(opp)} at all, but the {rng.choice(_NOUNS)} was ok.")
        rows.append({"text": " ".join(parts), "label": label})
    return pd.DataFrame(rows)
