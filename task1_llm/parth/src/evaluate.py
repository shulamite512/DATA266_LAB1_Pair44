"""Compute every Task 1 metric for a checkpoint and write metrics_report.csv.

Loss-based metrics come from a fresh eval-mode pass over the exact split used in
training. Diversity metrics come from the samples written by generate.py, and speed,
memory, and stability numbers come from the training summary. Anything not yet
produced is written as NEEDS_RUN, and anything not applicable is written as NA.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import CharWindowDataset, build_datasets, load_processed, make_dataloader  # noqa: E402
from generate import load_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    amp_settings, get_device, load_config, read_json, repo_relative, resolve_path, set_seed,
    utc_iso, utc_timestamp, write_csv, write_json,
)

NEEDS_RUN = "NEEDS_RUN"
NA = "NA"
REPORT_COLUMNS = ["metric", "value", "unit", "split_or_setting", "source", "notes"]


# ============================================================================ loss metrics
@torch.no_grad()
def evaluate_lm(model, loader, device, amp_enabled: bool = False, amp_dtype=torch.float32,
                max_batches: int | None = None) -> dict:
    """Token-weighted mean cross-entropy (nats) and top-1 next-character accuracy."""
    was_training = model.training
    model.eval()
    loss_sum, correct, tokens = 0.0, 0, 0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, _ = model(x)  # (B, T, V)
        logits = logits.float()
        V = logits.size(-1)
        # Why: summing per-token losses and dividing by the token count gives the exact
        # per-character mean even when the final batch is smaller than the rest.
        loss_sum += F.cross_entropy(logits.view(-1, V), y.view(-1), reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        tokens += y.numel()
    model.train(was_training)
    if tokens == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "tokens": 0}
    return {"loss": loss_sum / tokens, "accuracy": correct / tokens, "tokens": tokens}


# ============================================================================ diversity metrics
def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(token_lists: list[list[str]], n: int) -> float | None:
    """Corpus-level distinct-n: unique n-grams / total n-grams across all samples."""
    all_ngrams = [g for toks in token_lists for g in _ngrams(toks, n)]
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else None


def repeated_ngram_rate(token_lists: list[list[str]], n: int = 4) -> float | None:
    """Share of n-gram occurrences that repeat an earlier n-gram within the same sample."""
    repeats, total = 0, 0
    for toks in token_lists:
        grams = _ngrams(toks, n)
        total += len(grams)
        repeats += len(grams) - len(set(grams))
    return repeats / total if total else None


def generation_metrics(jsonl_path: Path) -> dict | None:
    """Diversity and speed per decoding setting; None if no samples exist yet."""
    if not jsonl_path.exists():
        return None
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return None
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("decoding", "unknown"), []).append(r)
    out = {}
    for label, recs in groups.items():
        # Why: diversity is measured on the generated continuation only, at word level
        # (whitespace split), since character n-grams mostly measure spelling, not repetition.
        words = [r["continuation"].split() for r in recs]
        new_tokens = sum(r["new_tokens"] for r in recs)
        seconds = sum(r["seconds"] for r in recs)
        out[label] = {
            "num_samples": len(recs),
            "distinct_1": distinct_n(words, 1),
            "distinct_2": distinct_n(words, 2),
            "distinct_3": distinct_n(words, 3),
            "repeated_4gram_rate": repeated_ngram_rate(words, 4),
            "generation_tokens_per_sec": new_tokens / seconds if seconds > 0 else None,
            "devices": sorted({r.get("device", "unknown") for r in recs}),
            "run_ids": sorted({str(r.get("run_id")) for r in recs}),
        }
    return out


# ============================================================================ report
def _fmt(value, digits: int = 6):
    if value is None:
        return NEEDS_RUN
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return round(value, digits)
    return value


def build_metric_rows(summary: dict | None, eval_results: dict | None,
                      gen_results: dict | None) -> list[dict]:
    """Assemble metrics_report.csv rows. Missing inputs become NEEDS_RUN, never guesses."""
    s = summary or {}
    e = eval_results or {}
    rows: list[dict] = []

    def add(metric, value, unit="", split="", source="", notes=""):
        rows.append({"metric": metric, "value": _fmt(value), "unit": unit,
                     "split_or_setting": split, "source": source, "notes": notes})

    ckpt_src = f"evaluate.py on {e.get('checkpoint', 'checkpoint')}" if e else "evaluate.py"
    train_loss = e.get("train_loss")
    val_loss = e.get("val_loss")

    add("train_cross_entropy", train_loss, "nats/char", "train (eval mode, subset)", ckpt_src,
        f"{e.get('train_tokens_evaluated', NEEDS_RUN)} tokens, dropout off")
    add("train_cross_entropy_epoch_mean", s.get("final_train_loss_epoch_mean"), "nats/char",
        "train (final epoch, dropout on)", "train.py summary", "running mean over the last epoch")
    add("val_cross_entropy", val_loss, "nats/char", "validation (full)", ckpt_src,
        f"{e.get('val_tokens_evaluated', NEEDS_RUN)} tokens")
    add("perplexity", math.exp(val_loss) if val_loss is not None else None, "", "validation",
        ckpt_src, "exp(val_cross_entropy)")
    add("bits_per_character", val_loss / math.log(2) if val_loss is not None else None, "bits/char",
        "validation", ckpt_src, "val_cross_entropy / ln(2)")
    gap = (val_loss - train_loss) if (val_loss is not None and train_loss is not None) else None
    add("generalization_gap", gap, "nats/char", "val - train", ckpt_src, "both measured in eval mode")
    add("top1_next_char_accuracy", e.get("val_accuracy"), "fraction", "validation", ckpt_src)
    add("top1_next_char_accuracy", e.get("train_accuracy"), "fraction", "train (eval mode, subset)", ckpt_src)

    if gen_results:
        for label, g in gen_results.items():
            src = "evaluate.py on generated_samples.jsonl"
            note = f"{g['num_samples']} samples, word-level, continuation only"
            add("distinct_1", g["distinct_1"], "fraction", label, src, note)
            add("distinct_2", g["distinct_2"], "fraction", label, src, note)
            add("distinct_3", g["distinct_3"], "fraction", label, src, note)
            add("repeated_4gram_rate", g["repeated_4gram_rate"], "fraction", label, src, note)
            add("generation_tokens_per_sec", g["generation_tokens_per_sec"], "chars/sec", label, src,
                f"device={','.join(g['devices'])}, batch size 1, no KV cache")
    else:
        for metric in ("distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate",
                       "generation_tokens_per_sec"):
            add(metric, None, "", "generation", "generate.py", "run generate.py first")

    tsrc = "train.py summary"
    add("grad_norm_mean", s.get("grad_norm_mean"), "L2", "train", tsrc, "pre-clip global norm")
    add("grad_norm_max", s.get("grad_norm_max"), "L2", "train", tsrc, "pre-clip global norm")
    add("grad_clipped_step_fraction", s.get("grad_clipped_fraction"), "fraction", "train", tsrc,
        f"grad_clip={s.get('grad_clip', NEEDS_RUN)}")
    add("nan_loss_steps", s.get("nan_loss_steps"), "count", "train", tsrc, "non-finite loss, step skipped")
    add("nonfinite_grad_steps", s.get("nonfinite_grad_steps"), "count", "train", tsrc,
        "fp16 loss-scale overflows land here; bf16/fp32 should be 0")
    add("loss_spike_steps", s.get("loss_spike_steps"), "count", "train", tsrc,
        f"loss > {s.get('spike_factor', NEEDS_RUN)} x EMA of recent loss")
    add("parameter_count", e.get("parameter_count", s.get("parameter_count")), "params", "model", ckpt_src)
    add("train_tokens_per_sec", s.get("train_tokens_per_sec"), "chars/sec", "train", tsrc,
        "training steps only, validation passes excluded")

    on_cuda = s.get("device") == "cuda" if s else None
    mem_val = s.get("peak_gpu_memory_allocated_mb") if on_cuda else (NA if s else None)
    add("peak_gpu_memory_allocated", mem_val, "MB", "train", tsrc, "torch.cuda.max_memory_allocated")
    mem_res = s.get("peak_gpu_memory_reserved_mb") if on_cuda else (NA if s else None)
    add("peak_gpu_memory_reserved", mem_res, "MB", "train", tsrc, "torch.cuda.max_memory_reserved")
    add("total_training_time", s.get("total_training_time_sec"), "seconds", "train", tsrc,
        "wall clock, includes per-epoch validation and checkpointing")
    add("epochs_completed", s.get("epochs_completed"), "epochs", "train", tsrc)
    add("best_epoch", s.get("best_epoch"), "epoch", "train", tsrc, "epoch with lowest val loss")
    add("hardware", s.get("hardware"), "", "train", tsrc)
    add("run_id", s.get("run_id") or e.get("run_id"), "", "", tsrc)
    return rows


def write_metrics_report(path, rows: list[dict]) -> Path:
    return write_csv(path, rows, REPORT_COLUMNS)


# ============================================================================ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a Task 1 checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--samples", default=None, help="generated_samples.jsonl (defaults to outputs dir)")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="Use the smoke overrides in train.yaml")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke)
    device = get_device()
    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, device)
    run_id = ckpt.get("run_id")
    seed = int(ckpt.get("seed", cfg["seed"]))
    set_seed(seed)
    tcfg = cfg["training"]
    paths = cfg["paths"]
    T = model.config.context_length

    # Why: the split is rebuilt from the checkpoint's own data config and seed, not the
    # current YAML, so evaluation always sees exactly the split the model trained against.
    data_cfg = ckpt["train_config"]["data"]
    cached = load_processed(data_cfg["processed_dir"])
    if cached is not None and cached[2].get("run_id") == run_id:
        train_ids, val_ids, meta = cached
        stride = int(meta["window_stride"])
        cap = int(meta["train_size_requested"]) if meta["split_unit"] == "sequences" else None
        vcap = int(meta["val_size_requested"]) if meta["split_unit"] == "sequences" else None
        train_ds = CharWindowDataset(train_ids, T, stride, cap)
        val_ds = CharWindowDataset(val_ids, T, stride, vcap)
        split_source = "data_processed cache (run_id match)"
    else:
        built = build_datasets(data_cfg, context_length=T, seed=seed, tokenizer=tokenizer)
        train_ds, val_ds = built["train_ds"], built["val_ds"]
        split_source = "rebuilt from checkpoint data config"

    bs = int(tcfg["batch_size"])
    workers = int(tcfg.get("num_workers", 0))
    pin = device.type == "cuda"
    max_train_batches = args.max_train_batches or int(tcfg.get("eval_train_batches", 100))
    n_train = min(len(train_ds), max_train_batches * bs)
    # Why: evenly spaced windows cover the whole training stream instead of only its start.
    step = max(1, len(train_ds) // n_train)
    train_subset = Subset(train_ds, list(range(0, len(train_ds), step))[:n_train])
    train_loader = make_dataloader(train_subset, bs, False, seed, workers, pin)
    val_loader = make_dataloader(val_ds, bs, False, seed, workers, pin)

    amp_enabled, amp_dtype, _ = amp_settings(device, bool(tcfg.get("amp", True)), tcfg.get("amp_dtype", "auto"))
    val = evaluate_lm(model, val_loader, device, amp_enabled, amp_dtype)
    train = evaluate_lm(model, train_loader, device, amp_enabled, amp_dtype)

    summary_path = resolve_path(paths["logs_dir"]) / f"train_summary_{run_id}.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    samples = resolve_path(args.samples) if args.samples else \
        resolve_path(paths["outputs_dir"]) / "generated_samples.jsonl"
    gen = generation_metrics(samples)
    if gen:
        foreign = {rid for g in gen.values() for rid in g["run_ids"]} - {str(run_id)}
        if foreign:
            print(f"[warn] samples file contains other run_ids {sorted(foreign)}; regenerate before reporting")

    eval_results = {
        "run_id": run_id,
        "checkpoint": repo_relative(resolve_path(args.checkpoint)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split_source": split_source,
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_tokens_evaluated": val["tokens"],
        "train_loss": train["loss"],
        "train_accuracy": train["accuracy"],
        "train_tokens_evaluated": train["tokens"],
        "parameter_count": model.count_parameters(),
        "amp_dtype": str(amp_dtype) if amp_enabled else "float32",
        "device": device.type,
        "evaluated_at": utc_iso(),
    }
    rows = build_metric_rows(summary, eval_results, gen)
    report_path = write_metrics_report(paths["metrics_report"], rows)
    outputs_dir = resolve_path(paths["outputs_dir"])
    write_json(outputs_dir / "eval_results.json", {"eval": eval_results, "generation": gen})
    write_json(resolve_path(paths["manifest_dir"]) / f"task1_parth_{run_id}_eval_{utc_timestamp()}.json", {
        "run_id": run_id,
        "checkpoint": eval_results["checkpoint"],
        "metrics_report": repo_relative(report_path),
        "samples_file": repo_relative(samples) if samples.exists() else None,
        "summary_file": repo_relative(summary_path) if summary_path.exists() else None,
        "eval": eval_results,
    })

    width = max(len(r["metric"]) for r in rows)
    for r in rows:
        print(f"{r['metric']:<{width}}  {str(r['value']):<14} {r['unit']:<10} {r['split_or_setting']}")
    print(f"\nWrote {repo_relative(report_path)}")
    return {"eval": eval_results, "generation": gen, "rows": rows}


if __name__ == "__main__":
    main()
