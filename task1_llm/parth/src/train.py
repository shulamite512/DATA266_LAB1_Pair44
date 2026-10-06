"""Train the Task 1 character-level GPT on TinyStories from a YAML config.

The run writes an unedited raw log, a manifest, latest and best checkpoints, a loss
curve figure, a per-step history, a training summary, and a first metrics report.
Pass --smoke to apply the small overrides from train.yaml for a quick end-to-end check.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import yaml

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import build_datasets, make_dataloader, save_processed  # noqa: E402
from evaluate import build_metric_rows, evaluate_lm, write_metrics_report  # noqa: E402
from model import GPTConfig, GPTLanguageModel  # noqa: E402
from utils import (  # noqa: E402
    amp_settings, collect_environment, get_device, git_info, load_config, pip_freeze,
    repo_relative, resolve_path, set_seed, utc_iso, utc_timestamp, write_json,
)

LOGGER_NAME = "task1_train"


# ============================================================================ setup helpers
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """Linear warmup to the peak LR, then cosine decay to min_lr_ratio * peak."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Why: warmup protects the randomly initialized attention layers from large early
    # AdamW updates while its second-moment estimates are still noisy; cosine decay then
    # shrinks steps smoothly as the loss flattens.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_loss_curves(history: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    steps, losses = history["step"], history["train_loss_step"]
    ax = axes[0]
    if steps:
        ax.plot(steps, losses, alpha=0.25, linewidth=0.8, label="train loss (per step)")
        window = max(1, len(losses) // 50)
        if window > 1:
            smooth = [sum(losses[max(0, i - window + 1): i + 1]) / len(losses[max(0, i - window + 1): i + 1])
                      for i in range(len(losses))]
            ax.plot(steps, smooth, linewidth=1.6, label=f"moving avg ({window} steps)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("Training loss per step")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    epochs = history["epoch"]
    if epochs:
        ax.plot(epochs, history["train_loss_epoch"], marker="o", label="train (epoch mean, dropout on)")
        ax.plot(epochs, history["val_loss_epoch"], marker="s", label="validation")
        best = min(range(len(epochs)), key=lambda i: history["val_loss_epoch"][i])
        ax.scatter([epochs[best]], [history["val_loss_epoch"][best]], s=120, facecolors="none",
                   edgecolors="red", linewidths=2, label=f"best val (epoch {epochs[best]})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("Train vs validation loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    if steps:
        ax.plot(steps, history["lr_step"])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("learning rate")
    ax.set_title("Warmup + cosine schedule")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_checkpoint(path: Path, model, cfg: dict, model_cfg: GPTConfig, tokenizer, run_id: str,
                    epoch: int, global_step: int, val_loss: float, train_loss: float,
                    best_val_loss: float, seed: int, extra: dict | None = None) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "train_config": cfg,
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "val_loss": float(val_loss),
        "train_loss_epoch_mean": float(train_loss),
        "best_val_loss": float(best_val_loss),
        "seed": seed,
        "torch_version": torch.__version__,
        "saved_at": utc_iso(),
    }
    if extra:
        ckpt.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    # Why: write-then-rename means an interrupted save can never leave a corrupt best.pt.
    tmp.replace(path)


# ============================================================================ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the Task 1 character-level GPT")
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true", help="Apply the smoke overrides from the config")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--run_name", default=None)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.run_name:
        cfg["run_name"] = args.run_name

    seed = int(cfg["seed"])
    set_seed(seed)
    tcfg, dcfg, paths_cfg = cfg["training"], cfg["data"], cfg["paths"]
    run_id = f"{cfg.get('run_name', 'task1')}{'_smoke' if cfg['is_smoke'] else ''}_{utc_timestamp()}"
    paths = {k: resolve_path(v) for k, v in paths_cfg.items()}
    for key in ("checkpoint_dir", "figures_dir", "logs_dir", "outputs_dir", "raw_log_dir", "manifest_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    raw_log_path = paths["raw_log_dir"] / f"task1_parth_{run_id}.log"
    log = setup_logger(raw_log_path)
    log.info("run_id=%s", run_id)
    log.info("resolved config:\n%s", yaml.safe_dump(cfg, sort_keys=False))

    device = get_device()
    env = collect_environment()
    hardware = env.get("gpu_name") if device.type == "cuda" else f"CPU ({env.get('cpu')})"
    log.info("device=%s hardware=%s torch=%s cuda_build=%s", device, hardware, env["torch"],
             env["torch_cuda_build"])
    if device.type == "cuda" and tcfg.get("allow_tf32", True):
        torch.set_float32_matmul_precision("high")

    # ---------------------------------------------------------------- data
    T = int(cfg["model"]["context_length"])
    t_data = time.perf_counter()
    data = build_datasets(dcfg, context_length=T, seed=seed)
    tokenizer, train_ds, val_ds, meta = data["tokenizer"], data["train_ds"], data["val_ds"], data["meta"]
    meta["run_id"] = run_id
    tok_path = resolve_path(dcfg["tokenizer_path"])
    tokenizer.save(tok_path)
    save_processed(dcfg["processed_dir"], data["train_ids"], data["val_ids"], meta)
    log.info("data prepared in %.1fs: %s", time.perf_counter() - t_data, meta)
    log.info("tokenizer saved to %s (vocab_size=%d)", repo_relative(tok_path), tokenizer.vocab_size)

    bs = int(tcfg["batch_size"])
    workers = int(tcfg.get("num_workers", 0))
    pin = device.type == "cuda"
    train_loader = make_dataloader(train_ds, bs, True, seed, workers, pin, drop_last=True)
    val_loader = make_dataloader(val_ds, bs, False, seed, workers, pin, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError(f"batch_size={bs} is larger than the {len(train_ds)} training windows")

    # ---------------------------------------------------------------- model / optim
    model_cfg = GPTConfig.from_dict(cfg["model"], vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(model_cfg).to(device)
    n_params = model.count_parameters()
    log.info("model config: %s", model_cfg.to_dict())
    log.info("parameter count: %s", f"{n_params:,}")

    optimizer = model.configure_optimizer(float(tcfg["learning_rate"]), float(tcfg["weight_decay"]),
                                          tcfg.get("betas", [0.9, 0.95]))
    epochs = int(tcfg["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = tcfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = int(round(float(tcfg.get("warmup_ratio", 0.05)) * total_steps))
    warmup_steps = int(min(warmup_steps, total_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, float(tcfg.get("min_lr_ratio", 0.1)))
    amp_enabled, amp_dtype, use_scaler = amp_settings(device, bool(tcfg.get("amp", True)),
                                                      tcfg.get("amp_dtype", "auto"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_clip = float(tcfg.get("grad_clip", 1.0) or 0.0)
    spike_factor = float(tcfg.get("spike_factor", 2.0))
    log_interval = int(tcfg.get("log_interval", 50))
    log.info("steps/epoch=%d total_steps=%d warmup_steps=%d amp=%s dtype=%s grad_scaler=%s",
             steps_per_epoch, total_steps, warmup_steps, amp_enabled, amp_dtype, use_scaler)

    # ---------------------------------------------------------------- bookkeeping
    history = {"step": [], "train_loss_step": [], "lr_step": [], "grad_norm_step": [],
               "epoch": [], "train_loss_epoch": [], "val_loss_epoch": [], "val_acc_epoch": [],
               "epoch_time_sec": [], "epoch_tokens_per_sec": []}
    nan_loss_steps = nonfinite_grad_steps = spike_steps = clipped_steps = 0
    grad_norms: list[float] = []
    loss_ema = None
    global_step = 0
    train_tokens = 0
    train_compute_sec = 0.0
    best_val, best_epoch = float("inf"), None
    ckpt_dir = paths["checkpoint_dir"]
    history_path = paths["logs_dir"] / f"history_{run_id}.json"
    figure_path = paths["figures_dir"] / "task1_loss_curve.png"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()

    # ---------------------------------------------------------------- training loop
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_epoch = time.perf_counter()
        epoch_loss_sum, epoch_batches, epoch_tokens = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)  # (B, T)
            y = y.to(device, non_blocking=True)  # (B, T)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _, loss = model(x, y)
            loss_val = loss.item()
            global_step += 1

            if not math.isfinite(loss_val):
                nan_loss_steps += 1
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                log.warning("non-finite loss at step %d (epoch %d); step skipped", global_step, epoch)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # no-op without fp16 scaling; needed so clipping sees true grads
            max_norm = grad_clip if grad_clip > 0 else float("inf")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

            if math.isfinite(grad_norm):
                if grad_clip > 0 and grad_norm > grad_clip:
                    clipped_steps += 1
                grad_norms.append(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nonfinite_grad_steps += 1
                if use_scaler:
                    # GradScaler detects the inf, skips the update, and lowers the loss scale.
                    scaler.step(optimizer)
                    scaler.update()
                log.warning("non-finite grad norm at step %d; optimizer update skipped", global_step)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if loss_ema is not None and loss_val > spike_factor * loss_ema:
                spike_steps += 1
                log.warning("loss spike at step %d: %.4f vs EMA %.4f", global_step, loss_val, loss_ema)
            loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val

            lr_now = scheduler.get_last_lr()[0]
            history["step"].append(global_step)
            history["train_loss_step"].append(loss_val)
            history["lr_step"].append(lr_now)
            history["grad_norm_step"].append(grad_norm if math.isfinite(grad_norm) else None)
            epoch_loss_sum += loss_val
            epoch_batches += 1
            epoch_tokens += y.numel()

            if global_step % log_interval == 0 or global_step == 1:
                log.info("epoch %d step %d/%d | loss %.4f | lr %.3e | grad_norm %.3f",
                         epoch, global_step, total_steps, loss_val, lr_now, grad_norm)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t_epoch
        train_compute_sec += epoch_sec
        train_tokens += epoch_tokens
        train_loss_epoch = epoch_loss_sum / max(1, epoch_batches)

        val = evaluate_lm(model, val_loader, device, amp_enabled, amp_dtype)
        history["epoch"].append(epoch)
        history["train_loss_epoch"].append(train_loss_epoch)
        history["val_loss_epoch"].append(val["loss"])
        history["val_acc_epoch"].append(val["accuracy"])
        history["epoch_time_sec"].append(epoch_sec)
        history["epoch_tokens_per_sec"].append(epoch_tokens / epoch_sec if epoch_sec > 0 else None)

        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch = val["loss"], epoch
        common = dict(model=model, cfg=cfg, model_cfg=model_cfg, tokenizer=tokenizer, run_id=run_id,
                      epoch=epoch, global_step=global_step, val_loss=val["loss"],
                      train_loss=train_loss_epoch, best_val_loss=best_val, seed=seed)
        save_checkpoint(ckpt_dir / "latest.pt", **common, extra={
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        })
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", **common)

        log.info("EPOCH %d/%d | train_loss %.4f | val_loss %.4f | val_ppl %.3f | val_bpc %.4f | "
                 "val_acc %.4f | %.0f tok/s | %.1fs%s",
                 epoch, epochs, train_loss_epoch, val["loss"], math.exp(val["loss"]),
                 val["loss"] / math.log(2), val["accuracy"],
                 epoch_tokens / epoch_sec if epoch_sec > 0 else float("nan"), epoch_sec,
                 " | new best -> best.pt" if improved else "")
        write_json(history_path, history)
        plot_loss_curves(history, figure_path)

    # ---------------------------------------------------------------- wrap up
    total_sec = time.perf_counter() - run_start
    peak_alloc = peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    summary = {
        "run_id": run_id,
        "is_smoke": cfg["is_smoke"],
        "device": device.type,
        "hardware": hardware,
        "amp_enabled": amp_enabled,
        "amp_dtype": str(amp_dtype) if amp_enabled else "float32",
        "parameter_count": n_params,
        "epochs_completed": len(history["epoch"]),
        "steps_completed": global_step,
        "batch_size": bs,
        "context_length": T,
        "tokens_per_epoch": steps_per_epoch * bs * T,
        "train_tokens_total": train_tokens,
        "train_compute_time_sec": train_compute_sec,
        "train_tokens_per_sec": train_tokens / train_compute_sec if train_compute_sec > 0 else None,
        "total_training_time_sec": total_sec,
        "peak_gpu_memory_allocated_mb": peak_alloc,
        "peak_gpu_memory_reserved_mb": peak_reserved,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss_epoch_mean": history["train_loss_epoch"][-1] if history["epoch"] else None,
        "final_val_loss": history["val_loss_epoch"][-1] if history["epoch"] else None,
        "grad_clip": grad_clip,
        "grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_clipped_fraction": clipped_steps / max(1, len(grad_norms)),
        "nan_loss_steps": nan_loss_steps,
        "nonfinite_grad_steps": nonfinite_grad_steps,
        "loss_spike_steps": spike_steps,
        "spike_factor": spike_factor,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "split": meta,
        "artifacts": {
            "best_checkpoint": repo_relative(ckpt_dir / "best.pt"),
            "latest_checkpoint": repo_relative(ckpt_dir / "latest.pt"),
            "raw_log": repo_relative(raw_log_path),
            "history": repo_relative(history_path),
            "loss_curve": repo_relative(figure_path),
            "tokenizer": repo_relative(tok_path),
        },
        "finished_at": utc_iso(),
    }
    summary_path = write_json(paths["logs_dir"] / f"train_summary_{run_id}.json", summary)

    rows = build_metric_rows(summary, None, None)
    report_path = write_metrics_report(paths["metrics_report"], rows)

    freeze_path = paths["manifest_dir"] / f"task1_parth_{run_id}_pip_freeze.txt"
    freeze_path.write_text(pip_freeze() + "\n", encoding="utf-8")
    manifest_path = write_json(paths["manifest_dir"] / f"task1_parth_{run_id}.json", {
        "run_id": run_id,
        "task": "task1_llm",
        "member": "parth",
        "command": "python task1_llm/parth/src/train.py " + " ".join(sys.argv[1:]) if argv is None
                   else "train.main(" + " ".join(argv) + ")",
        "created_at": utc_iso(),
        "git": git_info(),
        "environment": env,
        "pip_freeze_file": repo_relative(freeze_path),
        "config": cfg,
        "model_config": model_cfg.to_dict(),
        "data_split": meta,
        "checkpoint_map": {
            "best": {"path": summary["artifacts"]["best_checkpoint"], "epoch": best_epoch,
                     "val_loss": best_val, "used_for": "metrics_report.csv, generated samples"},
            "latest": {"path": summary["artifacts"]["latest_checkpoint"], "epoch": len(history["epoch"])},
        },
        "artifacts": {**summary["artifacts"], "summary": repo_relative(summary_path),
                      "metrics_report": repo_relative(report_path)},
    })

    log.info("training finished in %.1fs | best epoch %s val_loss %.4f | %.0f train tok/s | peak mem %s MB",
             total_sec, best_epoch, best_val, summary["train_tokens_per_sec"] or float("nan"),
             f"{peak_alloc:.1f}" if peak_alloc is not None else "NA")
    log.info("summary: %s", repo_relative(summary_path))
    log.info("manifest: %s", repo_relative(manifest_path))
    log.info("metrics report (training-side only, run evaluate.py next): %s", repo_relative(report_path))
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    return summary


if __name__ == "__main__":
    main()
