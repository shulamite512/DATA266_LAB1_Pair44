"""Evaluates a trained model on the Yelp Polarity test split and writes every
metric in the lab's Task 2 list: accuracy, P/R/F1 (macro, micro, weighted),
confusion matrix, ROC-AUC, PR-AUC, MCC, Brier, ECE, bootstrap CIs, McNemar
against the baseline, per-slice results, and the 20-error review sample."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

from dataset import prepare_data, slice_masks
from predict import Predictor
from utils import (LABEL_NAMES, Timer, apply_smoke_overrides, device_label, get_model_cfg,
                   get_paths, hardware_info, load_config, load_json, peak_memory_mb, rel,
                   reset_peak_memory, save_json, set_seed, setup_logger, sync, timestamp,
                   upsert_csv_row, write_manifest)


# ---------------------------------------------------------------- metric functions
def expected_calibration_error(y_true, p_pos, n_bins=15):
    """Weighted gap between confidence and accuracy over equal-width confidence bins."""
    y_true = np.asarray(y_true)
    p_pos = np.asarray(p_pos, dtype=float)
    pred = (p_pos >= 0.5).astype(int)
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = (pred == y_true).astype(float)
    # Why: binary confidence lives in [0.5, 1], so the bins cover that range only.
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "count": int(m.sum()),
                         "accuracy": float(correct[m].mean()), "confidence": float(conf[m].mean())})
    return float(ece), bins


def _counts_metrics(tp, fp, fn, tn):
    """Accuracy, macro-F1, and MCC from confusion counts (arrays allowed)."""
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(acc), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(acc), where=(2 * tn + fn + fp) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(acc), where=denom > 0)
    return acc, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0, alpha=0.05, chunk=100):
    """Percentile bootstrap CIs for accuracy, macro-F1, and MCC (resampling test reviews)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {"accuracy": [], "macro_f1": [], "mcc": []}
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        t, p = y_true[idx], y_pred[idx]
        tp = ((t == 1) & (p == 1)).sum(1).astype(float)
        tn = ((t == 0) & (p == 0)).sum(1).astype(float)
        fp = ((t == 0) & (p == 1)).sum(1).astype(float)
        fn = ((t == 1) & (p == 0)).sum(1).astype(float)
        acc, mf1, mcc = _counts_metrics(tp, fp, fn, tn)
        stats["accuracy"].append(acc)
        stats["macro_f1"].append(mf1)
        stats["mcc"].append(mcc)
    out = {}
    for k, v in stats.items():
        v = np.concatenate(v)
        out[k] = (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
    return out


def mcnemar_test(y_true, pred_a, pred_b):
    """Paired test on the reviews where exactly one of the two models is correct."""
    ca = np.asarray(pred_a) == np.asarray(y_true)
    cb = np.asarray(pred_b) == np.asarray(y_true)
    b = int((ca & ~cb).sum())   # A right, B wrong
    c = int((~ca & cb).sum())   # A wrong, B right
    if b + c == 0:
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "statistic": 0.0,
                "p_value": 1.0, "method": "no discordant pairs"}
    if b + c < 25:
        # Why: with few discordant pairs the chi-square approximation is poor,
        # so fall back to the exact binomial version.
        p = binomtest(b, b + c, 0.5).pvalue
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "statistic": float(min(b, c)),
                "p_value": float(p), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "statistic": float(stat),
            "p_value": float(chi2.sf(stat, 1)), "method": "chi2 with continuity correction"}


def compute_metrics(y_true, p_pos, threshold=0.5, ece_bins=15):
    y_true = np.asarray(y_true)
    p_pos = np.asarray(p_pos, dtype=float)
    y_pred = (p_pos >= threshold).astype(int)
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ["macro", "micro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, labels=[0, 1],
                                                     zero_division=0)
        m.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    both = len(np.unique(y_true)) == 2
    m["roc_auc"] = roc_auc_score(y_true, p_pos) if both else float("nan")
    m["pr_auc"] = average_precision_score(y_true, p_pos) if both else float("nan")
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["brier"] = brier_score_loss(y_true, p_pos)
    m["ece"], bins = expected_calibration_error(y_true, p_pos, ece_bins)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])})
    m = {k: float(v) if not isinstance(v, int) else v for k, v in m.items()}
    report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0,
                                   target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], digits=4)
    return m, cm, report, bins, y_pred


def slice_table(df, y_true, y_pred):
    rows = []
    for name, mask in slice_masks(df).items():
        n = int(mask.sum())
        if n == 0:
            rows.append({"slice": name, "n": 0, "macro_f1": np.nan, "error_rate": np.nan})
            continue
        rows.append({
            "slice": name, "n": n,
            "positive_fraction": float(y_true[mask].mean()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", labels=[0, 1],
                                       zero_division=0)),
            "error_rate": float((y_true[mask] != y_pred[mask]).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- error review
def select_error_review(pred_df, slices, k, min_slice_n, seed):
    """Pick 5 confident FPs, 5 confident FNs, 5 near-threshold, 5 from the worst slice."""
    err = pred_df[~pred_df["correct"]]
    picked, groups = set(), []

    def take(frame, bucket):
        frame = frame[~frame["id"].isin(picked)].head(k).copy()
        frame.insert(0, "review_bucket", bucket)
        picked.update(frame["id"])
        groups.append(frame)

    take(err[err["error_type"] == "false_positive"].sort_values("confidence", ascending=False),
         "confident_false_positive")
    take(err[err["error_type"] == "false_negative"].sort_values("confidence", ascending=False),
         "confident_false_negative")
    take(err.assign(d=(err["prob_positive"] - 0.5).abs()).sort_values("d").drop(columns="d"),
         "near_threshold")
    cand = slices[(slices["slice"] != "all") & (slices["n"] >= min_slice_n)].dropna(subset=["error_rate"])
    worst = None
    if len(cand):
        worst = cand.sort_values("error_rate", ascending=False).iloc[0]["slice"]
        mask = pd.Series(slice_masks(pred_df)[worst], index=pred_df.index)
        pool = err[mask[err.index]]
        pool = pool[~pool["id"].isin(picked)]
        take(pool.sample(n=min(k, len(pool)), random_state=seed), f"slice:{worst}")
    out = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    out["manual_error_type"] = ""   # fill in by hand after reading each review
    out["proposed_testable_fix"] = ""
    return out, worst


# ---------------------------------------------------------------- plots
def plot_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true neg", "true pos"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(bins, ece, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    if bins:
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label="model")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title} (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def evaluate_model(cfg, model_name, smoke=False):
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    paths = get_paths(cfg, smoke)
    ev = cfg["evaluation"]
    model_cfg = get_model_cfg(cfg, model_name)
    log_path = paths["logs"] / f"eval_{model_name}_{timestamp()}.log"
    logger = setup_logger("eval", log_path)
    set_seed(cfg["seed"])
    hw = hardware_info()

    train_summary_path = paths["outputs"] / f"train_summary_{model_name}.json"
    if not train_summary_path.exists():
        raise FileNotFoundError(f"{rel(train_summary_path)} missing. Run train.py --model {model_name} first.")
    train_summary = load_json(train_summary_path)

    test = prepare_data(cfg, paths, logger, smoke=smoke)["test"]
    predictor = Predictor(cfg, model_name, paths)
    device = predictor.device
    y = test["label"].to_numpy()

    reset_peak_memory(device)
    sync(device)
    with Timer() as t:
        p_pos = predictor.predict_proba_tokens(test["tokens"])
        sync(device)
    infer_mem = peak_memory_mb(device)
    logger.info(f"Inference on {len(test)} reviews took {t.seconds:.2f}s on {device_label(hw, device)}")

    metrics, cm, report, bins, y_pred = compute_metrics(y, p_pos, ev["threshold"], ev["ece_bins"])
    ci = bootstrap_ci(y, y_pred, ev["bootstrap_samples"], cfg["seed"])
    logger.info("Metrics: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in metrics.items()))
    logger.info(f"Bootstrap 95% CIs ({ev['bootstrap_samples']} resamples): {ci}")
    logger.info(f"Classification report:\n{report}")

    # ---- per-review predictions
    conf = np.maximum(p_pos, 1 - p_pos)
    pred_df = test[["id", "text", "label", "n_words_raw", "n_tokens", "length_bucket",
                    "has_negation", "has_contrast"]].rename(columns={"label": "true_label"}).copy()
    pred_df["predicted_label"] = y_pred
    pred_df["prob_positive"] = p_pos
    pred_df["confidence"] = conf
    pred_df["correct"] = pred_df["true_label"] == pred_df["predicted_label"]
    pred_df["error_type"] = np.where(pred_df["correct"], "correct",
                                     np.where(pred_df["predicted_label"] == 1, "false_positive",
                                              "false_negative"))
    pred_df["model_name"] = model_name
    out = paths["outputs"]
    pred_df.to_csv(out / f"predictions_{model_name}.csv", index=False)

    mis = pred_df[~pred_df["correct"]].sort_values("confidence", ascending=False)
    mis_cols = ["id", "text", "true_label", "predicted_label", "confidence", "prob_positive",
                "model_name", "error_type", "length_bucket", "has_negation", "has_contrast", "n_words_raw"]
    mis[mis_cols].to_csv(out / f"misclassified_examples_{model_name}.csv", index=False)
    combined = pd.concat([pd.read_csv(f) for f in sorted(out.glob("misclassified_examples_*.csv"))],
                         ignore_index=True)
    combined.to_csv(out / "misclassified_examples.csv", index=False)

    # ---- slices and error review sample
    slices = slice_table(pred_df, y, y_pred)
    slices.insert(0, "model", model_name)
    slices.to_csv(out / f"slice_metrics_{model_name}.csv", index=False)
    review, worst = select_error_review(pred_df, slices, ev["error_review_k"],
                                        ev["min_slice_n"], cfg["seed"])
    review.to_csv(out / f"error_review_candidates_{model_name}.csv", index=False)
    logger.info(f"Error review sample: {len(review)} rows (worst slice: {worst})")

    # ---- McNemar against the baseline, using identical test ids
    mcn = {}
    base = ev["baseline_model"]
    base_pred_path = out / f"predictions_{base}.csv"
    if model_name != base and base_pred_path.exists():
        bdf = pd.read_csv(base_pred_path, usecols=["id", "predicted_label"])
        merged = pred_df[["id", "text", "true_label", "predicted_label", "confidence"]].merge(
            bdf.rename(columns={"predicted_label": "baseline_pred"}), on="id", how="inner")
        if len(merged) != len(pred_df):
            logger.warning("Baseline predictions cover a different test set; McNemar uses the overlap only")
        mcn = mcnemar_test(merged["true_label"], merged["baseline_pred"], merged["predicted_label"])
        mcn.update({"baseline": base, "model": model_name, "n": len(merged)})
        upsert_csv_row(out / "mcnemar_results.csv", mcn, ["baseline", "model"])
        base_right = merged["baseline_pred"] == merged["true_label"]
        model_right = merged["predicted_label"] == merged["true_label"]
        merged["disagreement"] = np.where(base_right & ~model_right, f"{base}_right_{model_name}_wrong",
                                          np.where(~base_right & model_right,
                                                   f"{base}_wrong_{model_name}_right", "same"))
        merged[merged["disagreement"] != "same"].to_csv(
            out / f"disagreements_{base}_vs_{model_name}.csv", index=False)
        logger.info(f"McNemar vs {base}: {mcn}")
    elif model_name != base:
        logger.warning(f"No {rel(base_pred_path)} yet; evaluate the baseline first to get McNemar")

    # ---- figures
    plot_confusion(cm, paths["figures"] / f"confusion_matrix_{model_name}.png", model_name)
    plot_reliability(bins, metrics["ece"], paths["figures"] / f"reliability_{model_name}.png", model_name)

    # ---- one row in metrics_report.csv per model
    row = {
        "model": model_name, "type": model_cfg["type"],
        "arch": model_cfg.get("arch", "tfidf_logreg" if model_cfg["type"] == "logreg" else "meanpool_mlp"),
        "hyperparameters": str(train_summary["hyperparameters"]),
        "seed": cfg["seed"], "n_test": len(test),
        **metrics,
        "accuracy_ci95_low": ci["accuracy"][0], "accuracy_ci95_high": ci["accuracy"][1],
        "macro_f1_ci95_low": ci["macro_f1"][0], "macro_f1_ci95_high": ci["macro_f1"][1],
        "mcc_ci95_low": ci["mcc"][0], "mcc_ci95_high": ci["mcc"][1],
        "mcnemar_vs_baseline_p": mcn.get("p_value", np.nan),
        "mcnemar_vs_baseline_stat": mcn.get("statistic", np.nan),
        "param_count": train_summary["param_count"],
        "train_loss": train_summary["train_loss"], "val_loss": train_summary["val_loss"],
        "training_time_s": train_summary["training_time_s"],
        "train_examples_per_sec": train_summary["train_examples_per_sec"],
        "inference_time_s": t.seconds,
        "inference_examples_per_sec": len(test) / t.seconds,
        "train_peak_gpu_mb": train_summary["train_peak_memory"]["gpu_peak_mb"],
        "train_peak_cpu_rss_mb": train_summary["train_peak_memory"]["cpu_peak_rss_mb"],
        "eval_peak_gpu_mb": infer_mem["gpu_peak_mb"],
        "eval_peak_cpu_rss_mb": infer_mem["cpu_peak_rss_mb"],
        "train_device": train_summary["train_device"],
        "eval_device": device_label(hw, device),
        "checkpoint": train_summary["checkpoint"],
        "checkpoint_sha256": train_summary["checkpoint_sha256"],
        "train_log": train_summary["train_log"], "eval_log": rel(log_path),
    }
    for _, s in slices.iterrows():
        row[f"slice_{s['slice']}_n"] = s["n"]
        row[f"slice_{s['slice']}_macro_f1"] = s["macro_f1"]
        row[f"slice_{s['slice']}_error_rate"] = s["error_rate"]
    upsert_csv_row(paths["metrics_report"], row, ["model"])

    full = {"row": row, "bootstrap_ci95": ci, "mcnemar": mcn, "calibration_bins": bins,
            "confusion_matrix": cm, "classification_report": report, "worst_slice": worst}
    save_json(full, out / f"metrics_{model_name}.json")
    write_manifest(paths, model_name, "eval", {"config": cfg, "metrics": row})
    logger.info(f"Wrote {rel(paths['metrics_report'])} and outputs for {model_name}")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a trained Task 2 model on the test split")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    evaluate_model(load_config(args.config), args.model, smoke=args.smoke)
