"""Character-level tokenizer for the Task 1 TinyStories GPT.

The vocabulary is built from training text only and sorted by Unicode code point, so
the same training text always yields the same ids. Characters never seen during fit
map to a reserved unknown id instead of raising an error.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Why: U+FFFD is the standard Unicode replacement character, so an unknown id decodes to
# a visible marker instead of silently vanishing from generated text.
UNK_CHAR = "\ufffd"


class CharTokenizer:
    """Maps single characters to integer ids with its own char_to_idx / idx_to_char."""

    def __init__(self, min_freq: int = 1):
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        self.min_freq = min_freq
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}
        self.unk_idx = 0
        self.num_rare_chars_dropped = 0

    # ------------------------------------------------------------------ building
    def fit(self, text: str) -> "CharTokenizer":
        """Build the vocabulary from training text. Never call this on validation text."""
        if not text:
            raise ValueError("Cannot fit a tokenizer on empty text")
        counts = Counter(text)
        counts.pop(UNK_CHAR, None)
        kept = sorted(c for c, n in counts.items() if n >= self.min_freq)
        self.num_rare_chars_dropped = len(counts) - len(kept)
        # Why: id 0 is reserved for unknown characters so validation text containing a
        # character absent from the training split still encodes cleanly.
        self._set_vocab([UNK_CHAR] + kept)
        return self

    def _set_vocab(self, vocab: list[str]) -> None:
        if not vocab or vocab[0] != UNK_CHAR:
            raise ValueError("Vocabulary must start with the unknown character")
        if len(set(vocab)) != len(vocab):
            raise ValueError("Vocabulary contains duplicate characters")
        self.char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.unk_idx = 0

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)

    @property
    def vocab(self) -> list[str]:
        return [self.idx_to_char[i] for i in range(self.vocab_size)]

    def _check_fitted(self) -> None:
        if not self.char_to_idx:
            raise RuntimeError("Tokenizer is not fitted; call fit() or load() first")

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str) -> list[int]:
        self._check_fitted()
        lookup = self.char_to_idx
        unk = self.unk_idx
        return [lookup.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        self._check_fitted()
        table = self.idx_to_char
        return "".join(table.get(int(i), UNK_CHAR) for i in ids)

    def count_unknown(self, text: str) -> int:
        """Number of characters in text that would map to the unknown id."""
        self._check_fitted()
        return sum(1 for ch in text if ch not in self.char_to_idx)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "type": "char",
            "unk_char": UNK_CHAR,
            "unk_idx": self.unk_idx,
            "min_freq": self.min_freq,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if data.get("type") != "char":
            raise ValueError(f"Not a character tokenizer: type={data.get('type')}")
        tok = cls(min_freq=int(data.get("min_freq", 1)))
        tok._set_vocab(list(data["vocab"]))
        return tok

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size}, min_freq={self.min_freq})"
