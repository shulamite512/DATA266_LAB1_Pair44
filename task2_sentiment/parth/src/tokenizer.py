"""Text preprocessing and a word-level tokenizer built from scratch.
TextPreprocessor applies the lab's required cleaning steps (lowercase,
punctuation removal, stopword removal, stemming). WordTokenizer builds a
frequency-capped vocabulary from training text only and maps words to ids."""

import re
from collections import Counter
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

_ESCAPED_NEWLINE = re.compile(r"\\n")      # Yelp CSVs store newlines as a literal "\n"
_ESCAPED_QUOTE = re.compile(r'\\"')
_APOSTROPHE = re.compile(r"[\u2019']")
_NON_ALNUM_LOWER = re.compile(r"[^a-z0-9\s]")
_NON_ALNUM_ANY = re.compile(r"[^A-Za-z0-9\s]")


def normalize_raw(text):
    """Undo CSV escape artifacts only; used for display text and slice detection."""
    text = _ESCAPED_NEWLINE.sub(" ", str(text))
    return _ESCAPED_QUOTE.sub('"', text).strip()


class TextPreprocessor:
    """Turns a raw review into a list of cleaned tokens according to cfg['preprocessing']."""

    def __init__(self, pp_cfg):
        self.cfg = pp_cfg
        keep = set(str(w).lower() for w in pp_cfg.get("keep_stopwords", []))
        # Why: the stock English stopword list contains negations ("not", "no",
        # "never"). Deleting them flips "not good" into "good", which directly
        # hurts sentiment, so the config can protect them.
        self.stopwords = set(ENGLISH_STOP_WORDS) - keep
        self._stem = None
        if pp_cfg.get("stemming", False):
            from nltk.stem import PorterStemmer

            # Why: Yelp has far fewer unique words than tokens, so caching the
            # stemmer turns ~70M stem calls into a few hundred thousand.
            self._stem = lru_cache(maxsize=None)(PorterStemmer().stem)

    def __call__(self, text):
        t = normalize_raw(text)
        if self.cfg.get("lowercase", True):
            t = t.lower()
        if self.cfg.get("remove_punctuation", True):
            t = _APOSTROPHE.sub("", t)  # "don't" -> "dont" instead of "don t"
            t = (_NON_ALNUM_LOWER if self.cfg.get("lowercase", True) else _NON_ALNUM_ANY).sub(" ", t)
        tokens = t.split()
        if self.cfg.get("remove_stopwords", True):
            tokens = [w for w in tokens if w.lower() not in self.stopwords]
        if self._stem is not None:
            tokens = [self._stem(w) for w in tokens]
        return tokens

    def to_string(self, text):
        return " ".join(self(text))


class WordTokenizer:
    """Word-level vocabulary: id 0 is padding, id 1 is unknown, then words by frequency."""

    def __init__(self, itos=None):
        self.itos = list(itos) if itos else [PAD_TOKEN, UNK_TOKEN]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_strings, max_vocab, min_freq):
        """Count words in the training split only (never val/test) and keep the top ones."""
        counts = Counter()
        for s in token_strings:
            counts.update(s.split())
        words = [w for w, c in counts.most_common() if c >= min_freq]
        if max_vocab:
            words = words[: max_vocab - 2]
        return cls([PAD_TOKEN, UNK_TOKEN] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, token_string, max_length):
        ids = [self.stoi.get(w, UNK_ID) for w in token_string.split()[:max_length]]
        # Why: an empty review would give a zero-length mean pool; one <unk> keeps
        # the shape valid and is an honest "no information" input.
        return ids if ids else [UNK_ID]

    def decode(self, ids):
        return [self.itos[i] for i in ids if i != PAD_ID]
