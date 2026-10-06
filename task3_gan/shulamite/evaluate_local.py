"""Local evaluation entry point (lab-required name).

Usage (from the repo root): python task3_gan/shulamite/evaluate_local.py --run run2
Selects the checkpoint, writes pred_A2B / pred_B2A images, submission.csv (class-notebook FID/MiFID),
and every other Task 3 metric, then plots the training curves.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from advanced_metrics import main as calculate_metrics
from evaluate_cyclegan import main as generate_translations
from plot_training import main as plot_training

if __name__ == "__main__":
    generate_translations()
    calculate_metrics()
    plot_training()
