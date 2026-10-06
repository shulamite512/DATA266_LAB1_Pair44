"""Fast CPU smoke tests for Task 1 that need no TinyStories download.

They cover the tokenizer, story splitting and windowing, a tiny forward pass, a few
training steps, text generation, and one full train -> generate -> evaluate run on a
synthetic corpus written to a temporary directory.
"""
import csv
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "task1_llm" / "parth" / "src"
sys.path.insert(0, str(SRC))

from dataset import CharWindowDataset, build_datasets, chars_needed_for_windows, load_stories  # noqa: E402
from model import GPTConfig, GPTLanguageModel  # noqa: E402
from tokenizer import UNK_CHAR, CharTokenizer  # noqa: E402

ANIMALS = ["cat", "dog", "bird", "fox", "bear", "frog"]
NAMES = ["Lily", "Tom", "Mia", "Ben", "Sue"]
PLACES = ["park", "garden", "forest", "house", "lake"]


def synthetic_stories(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    stories = []
    for _ in range(n):
        name, animal, place = rng.choice(NAMES), rng.choice(ANIMALS), rng.choice(PLACES)
        stories.append(
            f"Once upon a time, {name} found a little {animal} in the {place}. "
            f"The {animal} was sad. {name} gave it a hug and they became friends."
        )
    return stories


def tiny_model(vocab_size: int, T: int = 16) -> GPTLanguageModel:
    cfg = GPTConfig(vocab_size=vocab_size, context_length=T, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    return GPTLanguageModel(cfg)


# ---------------------------------------------------------------------------- tokenizer
def test_tokenizer_roundtrip_unknown_and_save_load(tmp_path):
    text = "hello world. once upon a time!"
    tok = CharTokenizer().fit(text)
    assert tok.vocab_size == len(set(text)) + 1  # +1 for the unknown id
    assert tok.idx_to_char[0] == UNK_CHAR
    assert list(tok.char_to_idx) == [UNK_CHAR] + sorted(set(text))  # deterministic ordering
    assert tok.decode(tok.encode(text)) == text

    ids = tok.encode("hello Z")  # 'Z' never seen in fit
    assert ids[-1] == tok.unk_idx
    assert tok.count_unknown("hello Z") == 1

    path = tok.save(tmp_path / "tokenizer.json")
    loaded = CharTokenizer.load(path)
    assert loaded.char_to_idx == tok.char_to_idx
    assert loaded.decode(loaded.encode(text)) == text


# ---------------------------------------------------------------------------- dataset
def test_window_dataset_shapes_and_shift():
    ids = torch.arange(50, dtype=torch.int32)
    ds = CharWindowDataset(ids, context_length=8)
    assert len(ds) == (50 - 8 - 1) // 8 + 1
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)
    assert torch.equal(y, x + 1)  # y is x shifted by one position
    x2, _ = ds[1]
    assert x2[0].item() == 8  # stride defaults to context_length

    capped = CharWindowDataset(ids, context_length=8, stride=4, max_windows=3)
    assert len(capped) == 3
    xb = torch.stack([capped[i][0] for i in range(3)])
    assert xb.shape == (3, 8)


def test_build_datasets_split_before_fit(tmp_path):
    stories = synthetic_stories(40)
    stories[0] = "Zebra Q only in one story. " + stories[0]  # rare chars in one story
    data_file = tmp_path / "tiny_train.jsonl"
    data_file.write_text("\n".join(json.dumps({"text": s}) for s in stories), encoding="utf-8")
    T = 16
    cfg = {"data_dir": str(tmp_path), "split_unit": "sequences", "train_size": 20, "val_size": 5,
           "max_stories_to_load": 40, "story_separator": "\n\n"}
    out = build_datasets(cfg, context_length=T, seed=8503)
    assert len(out["train_ds"]) == 20 and len(out["val_ds"]) == 5
    assert len(out["train_ids"]) >= chars_needed_for_windows(20, T, T)
    x, y = out["train_ds"][0]
    assert x.shape == (T,) and y.shape == (T,)

    # Same seed gives the same split; a different seed gives a different one.
    again = build_datasets(cfg, context_length=T, seed=8503)
    assert torch.equal(out["train_ids"], again["train_ids"])
    other = build_datasets(cfg, context_length=T, seed=1)
    assert not torch.equal(out["train_ids"], other["train_ids"])

    # The tokenizer only knows characters present in the training text.
    train_text = out["tokenizer"].decode(out["train_ids"].tolist())
    assert set(out["tokenizer"].char_to_idx) - {UNK_CHAR} == set(train_text)


def test_load_stories_txt_with_delimiter(tmp_path):
    f = tmp_path / "stories_train.txt"
    f.write_text("First story.\n<|endoftext|>\nSecond story\nline two.\n<|endoftext|>\nThird.", encoding="utf-8")
    assert load_stories(f) == ["First story.", "Second story\nline two.", "Third."]
    assert load_stories(f, max_stories=2) == ["First story.", "Second story\nline two."]


# ---------------------------------------------------------------------------- model
def test_tiny_forward_pass_loss_near_uniform():
    torch.manual_seed(0)
    V = 30
    model = tiny_model(V)
    idx = torch.randint(0, V, (4, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 16, V)
    # With 0.02-std init the logits start near zero, so the loss is close to ln(V).
    assert abs(loss.item() - math.log(V)) < 0.5


def test_training_steps_update_params_and_reduce_loss():
    torch.manual_seed(0)
    text = " ".join(synthetic_stories(20))
    tok = CharTokenizer().fit(text)
    ds = CharWindowDataset(torch.tensor(tok.encode(text), dtype=torch.int32), context_length=16)
    x = torch.stack([ds[i][0] for i in range(8)])
    y = torch.stack([ds[i][1] for i in range(8)])
    model = tiny_model(tok.vocab_size)
    opt = model.configure_optimizer(learning_rate=3e-3, weight_decay=0.01)
    before = model.lm_head.weight.detach().clone()

    _, first_loss = model(x, y)
    for _ in range(30):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, last_loss = model(x, y)
    assert not torch.equal(before, model.lm_head.weight)
    assert torch.isfinite(last_loss)
    assert last_loss.item() < first_loss.item()


def test_generation_returns_text():
    torch.manual_seed(0)
    tok = CharTokenizer().fit("once upon a time there was a cat")
    model = tiny_model(tok.vocab_size)
    prompt = torch.tensor([tok.encode("once")], dtype=torch.long)
    for kwargs in ({"greedy": True}, {"temperature": 0.8, "top_k": 5}):
        out = model.generate(prompt, max_new_tokens=40, **kwargs)  # longer than context_length
        assert out.shape == (1, 4 + 40)
        text = tok.decode(out[0].tolist())
        assert isinstance(text, str) and text.startswith("once") and len(text) == 44


# ---------------------------------------------------------------------------- end to end
def test_train_generate_evaluate_pipeline(tmp_path):
    import evaluate
    import generate
    import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiny_train.jsonl").write_text(
        "\n".join(json.dumps({"story": s}) for s in synthetic_stories(120)), encoding="utf-8"
    )
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "model_type": "decoder-only-gpt", "normalization": "pre-norm", "positional_embedding": "learned",
        "activation": "gelu", "d_model": 32, "n_heads": 4, "n_layers": 2, "context_length": 16,
        "dropout": 0.1, "ffn_hidden_multiplier": 4,
    }))
    out = tmp_path / "run"
    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "run_name": "pytest_smoke",
        "seed": 8503,
        "model_config": str(model_yaml),
        "data": {"data_dir": str(data_dir), "max_stories_to_load": 120, "split_unit": "sequences",
                 "train_size": 40, "val_size": 8, "processed_dir": str(out / "data_processed"),
                 "tokenizer_path": str(out / "data_processed" / "tokenizer.json")},
        "training": {"epochs": 2, "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 0.01,
                     "warmup_ratio": 0.1, "num_workers": 0, "log_interval": 2, "amp": False,
                     "eval_train_batches": 2},
        "paths": {k: str(out / k) for k in ("checkpoint_dir", "figures_dir", "logs_dir", "outputs_dir",
                                            "raw_log_dir", "manifest_dir")}
                 | {"metrics_report": str(out / "metrics_report.csv")},
    }))

    summary = train.main(["--config", str(train_yaml)])
    assert summary["epochs_completed"] == 2
    assert (out / "checkpoint_dir" / "best.pt").exists()
    assert (out / "checkpoint_dir" / "latest.pt").exists()
    assert (out / "figures_dir" / "task1_loss_curve.png").exists()
    assert list((out / "raw_log_dir").glob("*.log"))
    assert list((out / "manifest_dir").glob("*.json"))
    assert (out / "data_processed" / "tokenizer.json").exists()

    samples_txt = out / "outputs_dir" / "generated_samples.txt"
    ckpt = str(out / "checkpoint_dir" / "best.pt")
    records = generate.main(["--checkpoint", ckpt, "--prompt", "Once upon a time",
                             "--max_new_tokens", "30", "--output", str(samples_txt)])
    generate.main(["--checkpoint", ckpt, "--prompt", "Once", "--max_new_tokens", "30", "--greedy",
                   "--output", str(samples_txt), "--append"])
    assert records[0]["new_tokens"] == 30 and samples_txt.exists()

    result = evaluate.main(["--checkpoint", ckpt, "--config", str(train_yaml)])
    assert math.isfinite(result["eval"]["val_loss"])
    assert result["eval"]["split_source"].startswith("data_processed cache")
    with open(out / "metrics_report.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_metric = {(r["metric"], r["split_or_setting"]): r["value"] for r in rows}
    assert float(by_metric[("val_cross_entropy", "validation (full)")]) > 0
    assert ("distinct_1", "greedy") in by_metric
    assert by_metric[("peak_gpu_memory_allocated", "train")] in ("NA",) or torch.cuda.is_available()
