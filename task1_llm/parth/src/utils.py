"""Shared helpers for the Task 1 scripts: config loading, seeding, device and AMP
selection, environment capture, and repo-relative path handling. Every config path
is resolved against the repository root, so no machine-specific absolute path ever
needs to appear in a committed file.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Why: src -> parth -> task1_llm -> repo root. Deriving the root from this file's
# location lets every script run from any working directory without hardcoded paths.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are interpreted from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def repo_relative(path: str | Path) -> str:
    """Render a path relative to the repo root for logs, so no personal paths leak."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def load_yaml(path: str | Path) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base and return the result."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(config_path: str | Path, smoke: bool = False) -> dict:
    """Load train.yaml, pull in the model.yaml it points to, and optionally apply the
    smoke-test overrides. The returned dict carries the architecture under "model".
    """
    cfg = load_yaml(config_path)
    model_path = cfg.get("model_config")
    model_cfg = load_yaml(model_path) if model_path else {}
    cfg["model"] = deep_merge(model_cfg, cfg.get("model", {}) or {})
    smoke_cfg = cfg.pop("smoke", {}) or {}
    if smoke:
        cfg = deep_merge(cfg, smoke_cfg)
    cfg["is_smoke"] = bool(smoke)
    return cfg


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(device: torch.device, enabled: bool, dtype_pref: str = "auto"):
    """Pick the autocast dtype. Returns (amp_enabled, amp_dtype, use_grad_scaler)."""
    if device.type != "cuda" or not enabled:
        return False, torch.float32, False
    pref = str(dtype_pref).lower()
    if pref in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif pref in ("fp16", "float16", "half"):
        dtype = torch.float16
    else:
        # Why: bf16 is native only from compute capability 8.0 upward. An RTX 5090 gets bf16
        # with no loss scaling, while a T4 (sm_75) falls back to fp16 plus GradScaler
        # instead of slow emulated bf16.
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return True, dtype, dtype == torch.float16


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], timeout: int = 60) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": None if status is None else bool(status),
    }


def pip_freeze() -> str:
    return _run([sys.executable, "-m", "pip", "freeze"], timeout=120) or "pip freeze unavailable"


def collect_environment() -> dict:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import matplotlib

        env["matplotlib"] = matplotlib.__version__
    except ImportError:
        env["matplotlib"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            {
                "gpu_name": props.name,
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_gb": round(props.total_memory / 1024**3, 2),
                "gpu_count": torch.cuda.device_count(),
                "nvidia_driver": _run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                ),
            }
        )
    else:
        env["cpu"] = platform.processor() or platform.machine()
    return env


def write_json(path: str | Path, obj: Any) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return p


def read_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    return p
