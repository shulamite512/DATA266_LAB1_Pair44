"""Training entry point for any model defined under `models:` in train.yaml.
Logistic regression is fit once on TF-IDF features; neural models train with
AdamW, cross-entropy, and gradient clipping, saving latest and best
checkpoints every epoch. Every run writes a new raw log and a manifest."""

import argparse
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss

from dataset import EncodedReviews, make_loader, prepare_data
from models import build_logreg, build_neural_model, count_parameters, logreg_param_count
from tokenizer import WordTokenizer
from utils import (Timer, apply_smoke_overrides, checkpoint_paths, device_label, file_sha256,
                   get_device, get_model_cfg, get_paths, hardware_info, load_config,
                   peak_memory_mb, rel, reset_peak_memory, save_json, set_seed, setup_logger,
                   sync, timestamp, to_builtin, write_manifest)


# ---------------------------------------------------------------- logistic regression
def train_logreg(cfg, model_name, model_cfg, splits, paths, logger):
    device = torch.device("cpu")
    pipe = build_logreg(model_cfg, cfg["seed"])
    X_tr, y_tr = splits["train"]["tokens"].tolist(), splits["train"]["label"].to_numpy()
    X_va, y_va = splits["val"]["tokens"].tolist(), splits["val"]["label"].to_numpy()

    with Timer() as t:
        pipe.fit(X_tr, y_tr)
    logger.info(f"Fit finished in {t.seconds:.1f}s")

    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_va = pipe.predict_proba(X_va)[:, 1]
    summary = {
        "model": model_name, "type": "logreg",
        "param_count": logreg_param_count(pipe),
        "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
        "train_loss": float(log_loss(y_tr, p_tr, labels=[0, 1])),
        "val_loss": float(log_loss(y_va, p_va, labels=[0, 1])),
        "val_accuracy": float(accuracy_score(y_va, (p_va >= 0.5).astype(int))),
        "converged_iterations": int(np.max(pipe.named_steps["clf"].n_iter_)),
        "training_time_s": t.seconds,
        "train_examples_per_sec": len(X_tr) / t.seconds,
        "epochs": 1,
    }
    # Why: TfidfVectorizer keeps every pruned n-gram in `stop_words_`, which can
    # make the saved file hundreds of MB; it is not needed for prediction.
    if getattr(pipe.named_steps["tfidf"], "stop_words_", None) is not None:
        pipe.named_steps["tfidf"].stop_words_ = None
    ckpt = checkpoint_paths(paths, model_name, "logreg")["model"]
    # Why: store the preprocessing settings with the model so prediction always
    # cleans text the same way the model was trained on.
    joblib.dump({"pipeline": pipe, "preprocessing_cfg": cfg["preprocessing"], "model_cfg": model_cfg}, ckpt)
    summary["checkpoint"] = rel(ckpt)
    summary["checkpoint_sha256"] = file_sha256(ckpt)
    return summary, device


# ---------------------------------------------------------------- neural
@torch.no_grad()
def evaluate_loss(model, loader, device, loss_fn):
    model.eval()
    total, n, correct = 0.0, 0, 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(ids, mask)
        total += loss_fn(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total / n, correct / n


def plot_loss_curves(history, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = [h["epoch"] for h in history["epochs"]]
    axes[0].plot(ep, [h["train_loss"] for h in history["epochs"]], "o-", label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history["epochs"]], "o-", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(f"{title}: epoch loss")
    axes[0].legend()
    if history["steps"]:
        s = history["steps"]
        axes[1].plot([x["step"] for x in s], [x["train_loss"] for x in s], lw=1)
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("train loss (running mean)")
    axes[1].set_title(f"{title}: step loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_neural(cfg, model_name, model_cfg, splits, paths, logger):
    device = get_device()
    seed = cfg["seed"]
    tok_cfg = cfg["tokenizer"]
    max_len = tok_cfg["max_length"]

    with Timer() as t_tok:
        # Why: the vocabulary sees the training split only, so val/test words
        # the model never trained on correctly become <unk>.
        tokenizer = WordTokenizer.build(splits["train"]["tokens"], tok_cfg["max_vocab"], tok_cfg["min_freq"])
        train_ds = EncodedReviews(splits["train"]["tokens"], splits["train"]["label"], tokenizer, max_len)
        val_ds = EncodedReviews(splits["val"]["tokens"], splits["val"]["label"], tokenizer, max_len)
    logger.info(f"Vocab size {len(tokenizer)}; encoding took {t_tok.seconds:.1f}s")

    pin = device.type == "cuda"
    nw = model_cfg.get("num_workers", 0)
    train_loader = make_loader(train_ds, model_cfg["batch_size"], True, seed, nw, pin)
    val_loader = make_loader(val_ds, model_cfg.get("eval_batch_size", 512), False, seed, nw, pin)

    set_seed(seed)  # Why: re-seed right before init so weights match across reruns.
    model = build_neural_model(model_cfg, len(tokenizer)).to(device)
    total_params, trainable = count_parameters(model)
    logger.info(f"Model:\n{model}")
    logger.info(f"Parameters: total={total_params:,} trainable={trainable:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"],
                                  weight_decay=model_cfg["weight_decay"])
    ckpts = checkpoint_paths(paths, model_name, "neural")
    select_by = model_cfg.get("select_by", "val_loss")
    best_score = math.inf if select_by == "val_loss" else -math.inf
    history = {"epochs": [], "steps": []}
    log_every = model_cfg.get("log_every", 200)
    step, nan_steps, seen = 0, 0, 0

    reset_peak_memory(device)
    sync(device)
    with Timer() as t_train:
        for epoch in range(1, model_cfg["epochs"] + 1):
            model.train()
            ep_loss, ep_n, grad_norms, run = 0.0, 0, [], []
            with Timer() as t_ep:
                for ids, mask, y in train_loader:
                    ids = ids.to(device, non_blocking=pin)
                    mask = mask.to(device, non_blocking=pin)
                    y = y.to(device, non_blocking=pin)
                    loss = loss_fn(model(ids, mask), y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg["grad_clip"])
                    if not torch.isfinite(loss) or not torch.isfinite(gn):
                        # Why: skip the update instead of letting one bad batch poison
                        # the weights; the count is logged as a stability metric.
                        nan_steps += 1
                        logger.warning(f"Non-finite loss/grad at step {step}; update skipped")
                        continue
                    optimizer.step()
                    step += 1
                    bs = y.size(0)
                    seen += bs
                    ep_loss += loss.item() * bs
                    ep_n += bs
                    grad_norms.append(gn.item())
                    run.append(loss.item())
                    if step % log_every == 0:
                        history["steps"].append({"step": step, "train_loss": float(np.mean(run))})
                        logger.info(f"epoch {epoch} step {step} train_loss {np.mean(run):.4f} "
                                    f"grad_norm {gn.item():.3f}")
                        run = []
                sync(device)
            val_loss, val_acc = evaluate_loss(model, val_loader, device, loss_fn)
            rec = {
                "epoch": epoch, "train_loss": ep_loss / max(ep_n, 1), "val_loss": val_loss,
                "val_accuracy": val_acc, "grad_norm_mean": float(np.mean(grad_norms)),
                "grad_norm_max": float(np.max(grad_norms)), "epoch_time_s": t_ep.seconds,
                "train_examples_per_sec": ep_n / t_ep.seconds,
            }
            history["epochs"].append(rec)
            logger.info("EPOCH " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                            for k, v in rec.items()))

            state = to_builtin({
                "model_name": model_name, "model_cfg": model_cfg, "tokenizer_cfg": tok_cfg,
                "preprocessing_cfg": cfg["preprocessing"], "seed": seed, "epoch": epoch,
                "history": history, "vocab": tokenizer.itos,
            })
            state["model_state"] = model.state_dict()
            state["optimizer_state"] = optimizer.state_dict()
            torch.save(state, ckpts["latest"])
            score = val_loss if select_by == "val_loss" else val_acc
            better = score < best_score if select_by == "val_loss" else score > best_score
            if better:
                best_score = score
                # Why: best_* is for evaluation only, so the optimizer state (2x the
                # model size with AdamW) stays in latest_* for resuming.
                best_state = {k: v for k, v in state.items() if k != "optimizer_state"}
                best_state["best_epoch"] = epoch
                torch.save(best_state, ckpts["best"])
                logger.info(f"New best checkpoint at epoch {epoch} ({select_by}={score:.4f})")

    plot_loss_curves(history, paths["figures"] / f"loss_curve_{model_name}.png", model_name)
    save_json(history, paths["outputs"] / f"history_{model_name}.json")
    best = torch.load(ckpts["best"], map_location="cpu", weights_only=True)
    summary = {
        "model": model_name, "type": "neural", "arch": model_cfg.get("arch", "meanpool_mlp"),
        "param_count": total_params, "trainable_params": trainable, "vocab_size": len(tokenizer),
        "best_epoch": best["best_epoch"], "select_by": select_by,
        "train_loss": history["epochs"][best["best_epoch"] - 1]["train_loss"],
        "val_loss": history["epochs"][best["best_epoch"] - 1]["val_loss"],
        "val_accuracy": history["epochs"][best["best_epoch"] - 1]["val_accuracy"],
        "final_train_loss": history["epochs"][-1]["train_loss"],
        "final_val_loss": history["epochs"][-1]["val_loss"],
        "grad_norm_max_overall": max(h["grad_norm_max"] for h in history["epochs"]),
        "nan_or_skipped_steps": nan_steps,
        "training_time_s": t_train.seconds,
        "train_examples_per_sec": seen / t_train.seconds,
        "epochs": model_cfg["epochs"],
        "checkpoint": rel(ckpts["best"]),
        "checkpoint_sha256": file_sha256(ckpts["best"]),
        "latest_checkpoint": rel(ckpts["latest"]),
    }
    return summary, device


# ---------------------------------------------------------------- main
def train_model(cfg, model_name, smoke=False):
    """Train one configured model; returns the training summary dict."""
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    paths = get_paths(cfg, smoke)
    model_cfg = get_model_cfg(cfg, model_name)
    log_path = paths["logs"] / f"train_{model_name}_{timestamp()}.log"
    logger = setup_logger("train", log_path)
    set_seed(cfg["seed"])
    hw = hardware_info()
    logger.info(f"Model '{model_name}' ({model_cfg['type']}) | smoke={smoke} | seed={cfg['seed']}")
    logger.info(f"Hardware: {hw}")
    logger.info(f"Model config: {model_cfg}")

    splits = prepare_data(cfg, paths, logger, smoke=smoke)
    if model_cfg["type"] == "logreg":
        summary, device = train_logreg(cfg, model_name, model_cfg, splits, paths, logger)
    elif model_cfg["type"] == "neural":
        summary, device = train_neural(cfg, model_name, model_cfg, splits, paths, logger)
    else:
        raise ValueError(f"Unknown model type {model_cfg['type']}")

    summary.update({
        "seed": cfg["seed"], "smoke": smoke, "hyperparameters": model_cfg,
        "train_device": device_label(hw, device),
        "hardware": hw, "train_peak_memory": peak_memory_mb(device),
        "train_log": rel(log_path), "n_train": len(splits["train"]), "n_val": len(splits["val"]),
    })
    save_json(summary, paths["outputs"] / f"train_summary_{model_name}.json")
    manifest = write_manifest(paths, model_name, "train", {"config": cfg, "summary": summary})
    logger.info(f"Summary: {to_builtin({k: v for k, v in summary.items() if k != 'hardware'})}")
    logger.info(f"Manifest written to {rel(manifest)}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a Task 2 sentiment model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="key under `models:` in the config")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run, outputs go to smoke_runs/")
    args = ap.parse_args()
    train_model(load_config(args.config), args.model, smoke=args.smoke)
