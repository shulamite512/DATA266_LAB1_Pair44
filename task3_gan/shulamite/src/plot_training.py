from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_cyclegan as gan


def main() -> None:
    metrics = json.loads((gan.OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    history = metrics["history"]
    epochs = [item["epoch"] for item in history]
    plots_dir = gan.OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("generator", "Total generator loss (adv + 10*cycle + 5*identity)"),
        ("discriminator", "Discriminator loss"),
        ("cycle", "Cycle-consistency L1 (A->B->A + B->A->B)"),
        ("identity", "Identity L1"),
    ]
    for axis, (key, title) in zip(axes.flat, panels):
        axis.plot(epochs, [item[key] for item in history], marker="o", markersize=3)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
    figure.suptitle(f"CycleGAN {gan.RUN_NAME} training losses")
    figure.tight_layout()
    figure.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item["gradient_norm"] for item in history], marker="o", markersize=3)
    axes[0].set_title(f"Mean max(G, D) gradient norm per epoch (NaN count total: {int(metrics['total_nan_count'])})")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [item.get("trained_learning_rate", item["learning_rate"]) for item in history], marker="o", markersize=3)
    axes[1].set_title("Learning rate used during epoch")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(plots_dir / "stability.png", dpi=150)
    plt.close(figure)

    progress = gan.OUTPUT_DIR / "progress_fid.csv"
    if progress.exists():
        with progress.open(encoding="utf-8") as file:
            rows = sorted(csv.DictReader(file), key=lambda row: int(row["epoch"]))
        figure, axis = plt.subplots(figsize=(7, 4))
        for key in ("fid_photo_to_monet", "fid_monet_to_photo", "mean_fid"):
            axis.plot([int(row["epoch"]) for row in rows], [float(row[key]) for row in rows], marker="o", label=key)
        axis.set_title("Class-notebook FID per checkpoint (300 images)")
        axis.set_xlabel("epoch")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(plots_dir / "fid_per_checkpoint.png", dpi=150)
        plt.close(figure)
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
