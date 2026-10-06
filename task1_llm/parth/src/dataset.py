"""TinyStories loading, train/validation splitting, and fixed-length character windows.

Stories are streamed from task1_llm/data, shuffled with the run seed, and split at the
story level before the tokenizer is fit, so no validation story contributes characters
to the vocabulary or to any training window.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import CharTokenizer
from utils import read_json, repo_relative, resolve_path, write_json

STORY_DELIMITER = "<|endoftext|>"
TEXT_FIELDS = ("text", "story", "content")
SUPPORTED_SUFFIXES = (".txt", ".jsonl", ".json")

# split_unit sets what train_size / val_size count. Every unit assigns whole stories to
# one split before windows are cut; "sequences" (fixed-length windows) is the default and
# the unit of the reported run.
SUPPORTED_SPLIT_UNITS = ("sequences", "stories", "characters")


# ============================================================================ file loading
def find_data_file(data_dir: str | Path, data_file: str | None = None) -> Path:
    """Return the TinyStories file to use, preferring a file whose name contains 'train'."""
    if data_file:
        path = resolve_path(data_file)
        if not path.exists():
            raise FileNotFoundError(f"data_file not found: {repo_relative(path)}")
        return path
    directory = resolve_path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"data_dir not found: {repo_relative(directory)}")
    candidates = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in SUPPORTED_SUFFIXES)
    if not candidates:
        raise FileNotFoundError(
            f"No .txt/.jsonl/.json file in {repo_relative(directory)}. "
            "Download TinyStories first (see task1_llm/data/README.md)."
        )
    train_like = [p for p in candidates if "train" in p.name.lower() and "valid" not in p.name.lower()]
    return (train_like or candidates)[0]


def _iter_txt_stories(path: Path) -> Iterator[str]:
    """Stream stories from a .txt file without reading the whole file into memory."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = f.read(1 << 20)
        f.seek(0)
        use_delimiter = STORY_DELIMITER in head
        buffer: list[str] = []
        for line in f:
            if use_delimiter:
                if STORY_DELIMITER in line:
                    parts = line.split(STORY_DELIMITER)
                    buffer.append(parts[0])
                    yield "".join(buffer)
                    yield from parts[1:-1]
                    buffer = [parts[-1]]
                else:
                    buffer.append(line)
            elif line.strip() == "":
                # Why: without an explicit delimiter, a blank line is the story boundary.
                if buffer:
                    yield "".join(buffer)
                    buffer = []
            else:
                buffer.append(line)
        if buffer:
            yield "".join(buffer)


def _extract_text(record) -> str | None:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in TEXT_FIELDS:
            value = record.get(key)
            if isinstance(value, str):
                return value
    return None


def _iter_jsonl_stories(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = _extract_text(json.loads(line))
            except json.JSONDecodeError:
                continue
            if text is not None:
                yield text


def _iter_json_stories(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list)]
        records = list_values[0] if list_values else [data]
    elif isinstance(data, list):
        records = data
    else:
        records = []
    for record in records:
        text = _extract_text(record)
        if text is not None:
            yield text


def iter_stories(path: str | Path) -> Iterator[str]:
    path = Path(path)
    if path.suffix == ".txt":
        return _iter_txt_stories(path)
    if path.suffix == ".jsonl":
        return _iter_jsonl_stories(path)
    if path.suffix == ".json":
        return _iter_json_stories(path)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def load_stories(path: str | Path, max_stories: int | None = None) -> list[str]:
    """Load up to max_stories non-empty, whitespace-stripped stories in file order."""
    stories: list[str] = []
    for story in iter_stories(path):
        story = story.strip()
        if story:
            stories.append(story)
            if max_stories and len(stories) >= max_stories:
                break
    return stories


# ============================================================================ splitting
def chars_needed_for_windows(num_windows: int, context_length: int, stride: int) -> int:
    """Characters required to cut num_windows (x, y) pairs, where y is x shifted by one."""
    return (num_windows - 1) * stride + context_length + 1


def _take_until(stories: list[str], start: int, need_chars: int, separator: str):
    taken: list[str] = []
    total = 0
    i = start
    while total < need_chars:
        if i >= len(stories):
            raise ValueError(
                f"Ran out of stories: needed {need_chars:,} characters but the loaded pool is "
                "too small. Increase data.max_stories_to_load in train.yaml."
            )
        total += len(stories[i]) + (len(separator) if taken else 0)
        taken.append(stories[i])
        i += 1
    return taken, i


def split_stories(
    stories: list[str],
    split_unit: str,
    train_size: int,
    val_size: int,
    context_length: int,
    stride: int,
    separator: str,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Shuffle stories with the seed, then assign whole stories to val first, then train."""
    if split_unit not in SUPPORTED_SPLIT_UNITS:
        raise ValueError(f"split_unit must be one of {SUPPORTED_SPLIT_UNITS}, got {split_unit!r}")
    shuffled = list(stories)
    random.Random(seed).shuffle(shuffled)

    if split_unit == "stories":
        if len(shuffled) < train_size + val_size:
            raise ValueError(
                f"Need {train_size + val_size:,} stories but only {len(shuffled):,} were loaded. "
                "Increase data.max_stories_to_load."
            )
        return shuffled[val_size : val_size + train_size], shuffled[:val_size]

    if split_unit == "characters":
        val_need, train_need = val_size, train_size
    else:
        val_need = chars_needed_for_windows(val_size, context_length, stride)
        train_need = chars_needed_for_windows(train_size, context_length, stride)

    # Why: splitting whole stories, never mid-story, means no validation story shares a
    # single character window with the training stream.
    val, pos = _take_until(shuffled, 0, val_need, separator)
    train, _ = _take_until(shuffled, pos, train_need, separator)
    return train, val


# ============================================================================ torch dataset
class CharWindowDataset(Dataset):
    """Fixed-length autoregressive windows over a 1-D id stream.

    Window i starts at i * stride. x has shape (T,) and y has shape (T,), with y equal
    to x shifted left by one character, so y[t] is the target for the prefix x[:t+1].
    """

    def __init__(self, ids: torch.Tensor, context_length: int, stride: int | None = None,
                 max_windows: int | None = None):
        if ids.dim() != 1:
            raise ValueError("ids must be a 1-D tensor")
        self.ids = ids
        self.context_length = int(context_length)
        self.stride = int(stride or context_length)
        if len(ids) < self.context_length + 1:
            raise ValueError(
                f"Need at least {self.context_length + 1} tokens, got {len(ids)}"
            )
        n = (len(ids) - self.context_length - 1) // self.stride + 1
        self.num_windows = min(n, max_windows) if max_windows else n

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, i: int):
        if i < 0 or i >= self.num_windows:
            raise IndexError(i)
        start = i * self.stride
        chunk = self.ids[start : start + self.context_length + 1].long()  # (T + 1,)
        x = chunk[:-1]  # (T,)
        y = chunk[1:]  # (T,)
        return x, y


def encode_to_tensor(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    # Why: int32 halves memory versus int64 for the full stream; windows are cast to
    # long only when a batch is sliced out.
    return torch.tensor(tokenizer.encode(text), dtype=torch.int32)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int,
                    num_workers: int = 0, pin_memory: bool = False,
                    drop_last: bool = False) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


# ============================================================================ pipeline
def build_datasets(data_cfg: dict, context_length: int, seed: int,
                   tokenizer: CharTokenizer | None = None) -> dict:
    """Load, split, tokenize, and window TinyStories in a leakage-safe order.

    Order: load stories, shuffle and split by story, fit the tokenizer on training text
    only (skipped when a tokenizer is passed in, e.g. for evaluation), encode both splits,
    then cut fixed-length windows. Returns datasets, id tensors, tokenizer, and split metadata.
    """
    path = find_data_file(data_cfg.get("data_dir", "task1_llm/data"), data_cfg.get("data_file"))
    split_unit = data_cfg.get("split_unit", "sequences")
    train_size = int(data_cfg["train_size"])
    val_size = int(data_cfg["val_size"])
    stride = int(data_cfg.get("window_stride") or context_length)
    separator = data_cfg.get("story_separator", "\n\n")
    max_stories = data_cfg.get("max_stories_to_load")

    stories = load_stories(path, max_stories)
    train_stories, val_stories = split_stories(
        stories, split_unit, train_size, val_size, context_length, stride, separator, seed
    )
    train_text = separator.join(train_stories)
    val_text = separator.join(val_stories)
    if split_unit == "characters":
        train_text, val_text = train_text[:train_size], val_text[:val_size]

    fitted_here = tokenizer is None
    if fitted_here:
        tokenizer = CharTokenizer(min_freq=int(data_cfg.get("min_char_freq", 1))).fit(train_text)

    train_ids = encode_to_tensor(tokenizer, train_text)
    val_ids = encode_to_tensor(tokenizer, val_text)
    cap_train = train_size if split_unit == "sequences" else None
    cap_val = val_size if split_unit == "sequences" else None
    train_ds = CharWindowDataset(train_ids, context_length, stride, cap_train)
    val_ds = CharWindowDataset(val_ids, context_length, stride, cap_val)

    meta = {
        "data_file": repo_relative(path),
        "split_unit": split_unit,
        "split_unit_note": "NEEDS_CLARIFICATION: confirm unit of the 100K/10K split",
        "train_size_requested": train_size,
        "val_size_requested": val_size,
        "seed": seed,
        "stories_loaded": len(stories),
        "stories_loaded_note": "first max_stories_to_load stories in file order, then seeded shuffle",
        "train_stories": len(train_stories),
        "val_stories": len(val_stories),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "context_length": context_length,
        "window_stride": stride,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fitted_on": "train split only" if fitted_here else "loaded from checkpoint",
        "val_unknown_chars": tokenizer.count_unknown(val_text),
        "story_separator": separator,
    }
    return {
        "tokenizer": tokenizer,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "meta": meta,
    }


def save_processed(processed_dir: str | Path, train_ids: torch.Tensor, val_ids: torch.Tensor,
                   meta: dict) -> None:
    out = resolve_path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(train_ids, out / "train_ids.pt")
    torch.save(val_ids, out / "val_ids.pt")
    write_json(out / "split_meta.json", meta)


def load_processed(processed_dir: str | Path):
    """Return (train_ids, val_ids, meta) if a processed split exists, else None."""
    out = resolve_path(processed_dir)
    files = [out / "train_ids.pt", out / "val_ids.pt", out / "split_meta.json"]
    if not all(f.exists() for f in files):
        return None
    return torch.load(files[0]), torch.load(files[1]), read_json(files[2])
