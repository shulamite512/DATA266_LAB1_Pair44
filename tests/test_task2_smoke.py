"""CPU smoke tests for Task 2 (Yelp Polarity). They use tiny synthetic data
and a temporary output folder, so they never touch real checkpoints, never
download data, and do not depend on anything in task1_llm/."""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "task2_sentiment" / "parth" / "src"
sys.path.insert(0, str(SRC))

from dataset import (EncodedReviews, add_slice_columns, clean_split, collate_batch,  # noqa: E402
                     make_loader, prepare_data, read_yelp_csv)
from evaluate import (bootstrap_ci, compute_metrics, evaluate_model,  # noqa: E402
                      expected_calibration_error, mcnemar_test)
from models import build_neural_model, count_parameters  # noqa: E402
from predict import Predictor  # noqa: E402
from tokenizer import PAD_ID, UNK_ID, TextPreprocessor, WordTokenizer  # noqa: E402
from train import train_model  # noqa: E402
from utils import (apply_smoke_overrides, get_paths, load_config, make_synthetic_df,  # noqa: E402
                   set_seed, setup_logger)

CONFIG = REPO_ROOT / "task2_sentiment" / "parth" / "config" / "train.yaml"


@pytest.fixture
def cfg(tmp_path):
    c = load_config(CONFIG)
    c["paths"]["smoke_dir"] = str(tmp_path / "smoke_runs")  # absolute, stays in tmp
    return c


def test_preprocessor_keeps_negations_and_cleans():
    pre = TextPreprocessor({"lowercase": True, "remove_punctuation": True, "remove_stopwords": True,
                            "keep_stopwords": ["not", "no", "never"], "stemming": True})
    toks = pre("The food was NOT good!!\\nWe didn't love the waiters.")
    assert "not" in toks                      # negation survives stopword removal
    assert "the" not in toks and "was" not in toks
    assert all(t.isalnum() for t in toks)     # punctuation removed
    assert "didnt" in toks                    # apostrophe removed, not split
    assert "waiter" in toks                   # stemmed


def test_word_tokenizer_roundtrip_and_unk():
    tok = WordTokenizer.build(["good food good", "bad food"], max_vocab=10, min_freq=1)
    assert tok.itos[:2] == ["<pad>", "<unk>"]
    ids = tok.encode("good unseenword food", max_length=10)
    assert ids[1] == UNK_ID and ids[0] != UNK_ID
    assert tok.encode("good good good", max_length=2) == [tok.stoi["good"]] * 2
    assert tok.encode("", max_length=5) == [UNK_ID]


def test_csv_loading_maps_1_2_labels_and_drops_malformed(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text('"1","awful place"\n"2","great place"\n"2",""\n"7","weird label"\n"1",\n')
    df = read_yelp_csv(p)
    clean, report = clean_split(df, "train", drop_duplicates=True)
    assert sorted(clean["label"].tolist()) == [0, 1]
    assert report["invalid_label"] == 1 and report["missing_text"] >= 1


def test_prepare_data_on_synthetic(cfg):
    c = apply_smoke_overrides(cfg)
    paths = get_paths(c, smoke=True)
    log = setup_logger("test", paths["logs"] / "test.log")
    splits = prepare_data(c, paths, log, smoke=True)
    assert set(splits) == {"train", "val", "test"}
    for df in splits.values():
        assert set(df["label"].unique()) <= {0, 1}
        assert {"tokens", "length_bucket", "has_negation", "has_contrast"} <= set(df.columns)
    assert (paths["outputs"] / "eda_summary.json").exists()


def test_neural_forward_and_padding_mask():
    set_seed(0)
    tok = WordTokenizer.build(["good food", "bad food service"], 50, 1)
    ds = EncodedReviews(["good food", "bad food service"], [1, 0], tok, max_length=8)
    ids, mask, y = collate_batch([ds[0], ds[1]])
    assert ids.shape == (2, 3) and ids[0, -1] == PAD_ID and not mask[0, -1]
    model = build_neural_model({"type": "neural", "arch": "meanpool_mlp", "embedding_dim": 8,
                                "hidden_dim": 16, "dropout": 0.0}, len(tok))
    model.eval()
    logits = model(ids, mask)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    # Padding must not change the pooled output of the shorter review.
    alone = model(ids[:1, :2], mask[:1, :2])
    assert torch.allclose(alone, logits[:1], atol=1e-6)
    assert count_parameters(model)[0] > 0


def test_neural_cnn_forward_and_padding_mask(cfg):
    set_seed(0)
    tok = WordTokenizer.build(["good food", "bad food service was slow today"], 50, 1)
    ds = EncodedReviews(["good food", "bad food service was slow today"], [1, 0], tok, max_length=8)
    ids, mask, y = collate_batch([ds[0], ds[1]])
    model = build_neural_model(cfg["models"]["neural_cnn"], len(tok))
    model.eval()
    logits = model(ids, mask)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    # Padding must not change the output of the shorter review (shorter than every kernel).
    alone = model(ids[:1, :2], mask[:1, :2])
    assert torch.allclose(alone, logits[:1], atol=1e-6)
    loss = torch.nn.functional.cross_entropy(model(ids, mask), y)
    loss.backward()
    assert torch.isfinite(loss) and count_parameters(model)[0] > 0


def test_one_training_step_updates_weights():
    set_seed(0)
    df = make_synthetic_df(64, 0)
    pre = TextPreprocessor({"lowercase": True, "remove_punctuation": True, "remove_stopwords": True,
                            "keep_stopwords": ["not"], "stemming": False})
    toks = df["text"].map(pre.to_string)
    tok = WordTokenizer.build(toks, 100, 1)
    loader = make_loader(EncodedReviews(toks, df["label"], tok, 32), 16, True, 0)
    model = build_neural_model({"arch": "meanpool_mlp", "embedding_dim": 16, "hidden_dim": 16,
                                "dropout": 0.1}, len(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.fc2.weight.detach().clone()
    ids, mask, y = next(iter(loader))
    loss = torch.nn.functional.cross_entropy(model(ids, mask), y)
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    assert torch.isfinite(loss) and torch.isfinite(gn)
    assert not torch.equal(before, model.fc2.weight)


def test_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.9, 0.4])   # one FP, one FN
    m, cm, report, _, pred = compute_metrics(y, p)
    assert cm.tolist() == [[1, 1], [1, 1]]
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["f1_macro"] == pytest.approx(0.5)
    assert m["mcc"] == pytest.approx(0.0)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert "negative" in report and "positive" in report
    # Perfect, fully confident predictions are perfectly calibrated.
    assert expected_calibration_error([0, 1, 1], [0.0, 1.0, 1.0])[0] == pytest.approx(0.0)
    ci = bootstrap_ci(y, pred, n_boot=50, seed=0)
    assert ci["accuracy"][0] <= 0.5 <= ci["accuracy"][1]
    t = mcnemar_test([1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 1])
    assert t["b_a_right_b_wrong"] == 2 and t["c_a_wrong_b_right"] == 1


def test_slice_columns():
    df = pd.DataFrame({"text": ["not good", "great but slow " * 40, "fine"]})
    out = add_slice_columns(df, {"short_max_words": 50, "long_min_words": 100,
                                 "negation_words": ["not"], "contrast_words": ["but"]})
    assert out["has_negation"].tolist() == [True, False, False]
    assert out["has_contrast"].tolist() == [False, True, False]
    assert out["length_bucket"].tolist() == ["short", "long", "short"]


def test_end_to_end_smoke_train_evaluate_predict(cfg):
    for name in ["logreg", "neural"]:
        s = train_model(copy.deepcopy(cfg), name, smoke=True)
        assert s["param_count"] > 0
    rows = [evaluate_model(copy.deepcopy(cfg), name, smoke=True) for name in ["logreg", "neural"]]
    assert 0.0 <= rows[1]["accuracy"] <= 1.0
    assert not np.isnan(rows[1]["mcnemar_vs_baseline_p"])  # baseline evaluated first

    c = apply_smoke_overrides(cfg)
    paths = get_paths(c, smoke=True)
    report = pd.read_csv(paths["metrics_report"])
    assert set(report["model"]) == {"logreg", "neural"}
    for f in ["predictions_neural.csv", "misclassified_examples.csv",
              "error_review_candidates_neural.csv", "slice_metrics_neural.csv"]:
        assert (paths["outputs"] / f).exists(), f
    assert (paths["figures"] / "confusion_matrix_neural.png").exists()
    assert (paths["figures"] / "loss_curve_neural.png").exists()

    for name in ["logreg", "neural"]:
        r = Predictor(c, name, paths).predict_texts(["The food was amazing and the service was great."])[0]
        assert r["label"] in (0, 1)
        assert r["probabilities"]["negative"] + r["probabilities"]["positive"] == pytest.approx(1.0)


def test_task2_does_not_import_task1():
    assert not any("task1" in str(getattr(m, "__file__", "") or "") for m in list(sys.modules.values()))


def test_config_negation_words_are_strings():
    # Unquoted `no` in YAML loads as False, which would silently drop it.
    c = load_config(CONFIG)
    assert "no" in c["preprocessing"]["keep_stopwords"]
    assert all(isinstance(w, str) for w in c["slices"]["negation_words"])
