from __future__ import annotations

import json
import sys
import time
import csv
from pathlib import Path

import lpips
import numpy as np
import scipy.linalg
import scipy.spatial.distance
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_cyclegan as gan

EVALUATION_DIR = gan.OUTPUT_DIR / "evaluation"
SUBMISSION_DIR = EVALUATION_DIR / "submission_images"
REVERSE_DIR = EVALUATION_DIR / "reverse_images"
METRICS_PATH = EVALUATION_DIR / "advanced_metrics.json"
BATCH_SIZE = 16
NOTEBOOK_EVAL_COUNT = 300
MIFID_COSINE_EPS = 0.1
PRECISION_RECALL_K = 3


def load_uint8(path: Path) -> torch.Tensor:
    image = gan.image_to_tensor(path)
    return (((image + 1) * 127.5).clamp(0, 255)).byte()


def batch_paths(paths: list[Path], batch_size: int):
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]


INCEPTION_TRANSFORM = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_inception_model(device: torch.device) -> torch.nn.Module:
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    inception.fc = torch.nn.Identity()
    return inception.to(device).eval()


@torch.no_grad()
def inception_features(model: torch.nn.Module, paths: list[Path], device: torch.device) -> np.ndarray:
    features = []
    for paths_batch in batch_paths(paths, BATCH_SIZE):
        images = [INCEPTION_TRANSFORM(Image.open(path).convert("RGB")) for path in paths_batch]
        features.append(model(torch.stack(images).to(device)).cpu().numpy())
    return np.concatenate(features, axis=0)


def generative_precision_recall(real_features: np.ndarray, generated_features: np.ndarray, k: int = PRECISION_RECALL_K) -> tuple[float, float]:
    # Kynkaanniemi et al. (2019): k-NN manifold estimate in Inception feature space.
    real_radii = np.sort(scipy.spatial.distance.cdist(real_features, real_features), axis=1)[:, k]
    generated_radii = np.sort(scipy.spatial.distance.cdist(generated_features, generated_features), axis=1)[:, k]
    cross = scipy.spatial.distance.cdist(generated_features, real_features)
    precision = float(np.mean(np.any(cross <= real_radii[None, :], axis=1)))
    recall = float(np.mean(np.any(cross.T <= generated_radii[None, :], axis=1)))
    return precision, recall


def notebook_fid_mifid(real_paths: list[Path], generated_paths: list[Path], device: torch.device) -> tuple[float, float, float, float, float]:
    count = min(NOTEBOOK_EVAL_COUNT, len(real_paths), len(generated_paths))
    model = load_inception_model(device)
    real_features = inception_features(model, sorted(real_paths)[:count], device)
    generated_features = inception_features(model, sorted(generated_paths)[:count], device)
    real_mean, real_covariance = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    generated_mean, generated_covariance = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    covariance_sqrt = scipy.linalg.sqrtm(real_covariance.dot(generated_covariance))
    if not np.isfinite(covariance_sqrt).all():
        offset = np.eye(real_covariance.shape[0]) * 1e-6
        covariance_sqrt = scipy.linalg.sqrtm((real_covariance + offset).dot(generated_covariance + offset))
    if np.iscomplexobj(covariance_sqrt):
        covariance_sqrt = covariance_sqrt.real
    difference = real_mean - generated_mean
    fid = float(difference.dot(difference) + np.trace(real_covariance + generated_covariance - 2 * covariance_sqrt))
    real_norm = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
    generated_norm = generated_features / np.linalg.norm(generated_features, axis=1, keepdims=True)
    # Class protocol (Part3_Evaluation_Script.ipynb): MiFID = mean cosine distance of index-paired features.
    class_mifid = float(np.mean(1.0 - np.sum(real_norm * generated_norm, axis=1)))
    # Original Kaggle MiFID, reported for reference only: FID / thresholded memorisation distance.
    memorization_distance = float(np.mean(np.min(1.0 - generated_norm @ real_norm.T, axis=1)))
    thresholded_distance = memorization_distance if memorization_distance < MIFID_COSINE_EPS else 1.0
    kaggle_mifid = fid / (thresholded_distance + 1e-15)
    precision, recall = generative_precision_recall(real_features, generated_features)
    return fid, class_mifid, kaggle_mifid, precision, recall


def save_reverse_outputs(model: torch.nn.Module, monet_paths: list[Path]) -> list[Path]:
    if REVERSE_DIR.exists():
        for path in REVERSE_DIR.glob("*.jpg"):
            path.unlink()
    REVERSE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    model.eval()
    with torch.no_grad():
        for index, path in enumerate(monet_paths):
            translated = model(gan.image_to_tensor(path).unsqueeze(0).to(gan.DEVICE))[0]
            output_path = REVERSE_DIR / f"{index:05d}.jpg"
            gan.save_image(translated, output_path)
            output_paths.append(output_path)
    return output_paths


def evaluate_direction(real_paths: list[Path], generated_paths: list[Path], source_paths: list[Path], label: str, device: torch.device) -> dict:
    direction_start = time.perf_counter()
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(subset_size=min(50, len(real_paths)), normalize=False).to(device)
    real_images = [load_uint8(path) for path in real_paths]
    fake_images = [load_uint8(path) for path in generated_paths]
    for images in batch_paths(real_images, BATCH_SIZE):
        batch = torch.stack(images).to(device)
        fid.update(batch, real=True)
        kid.update(batch, real=True)
    for images in batch_paths(fake_images, BATCH_SIZE):
        batch = torch.stack(images).to(device)
        fid.update(batch, real=False)
        kid.update(batch, real=False)
    fid_value = float(fid.compute().item())
    kid_mean, kid_std = kid.compute()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    cosine_values = []
    lpips_values = []
    for source_path, generated_path in zip(source_paths[:100], generated_paths[:100]):
        source = gan.image_to_tensor(source_path).unsqueeze(0).to(device)
        generated = gan.image_to_tensor(generated_path).unsqueeze(0).to(device)
        cosine_values.append(float(F.cosine_similarity(source.flatten(1), generated.flatten(1)).item()))
        with torch.no_grad():
            lpips_values.append(float(lpips_model(source, generated).mean().item()))
    elapsed = time.perf_counter() - direction_start
    return {
        f"fid_{label}": fid_value,
        f"kid_{label}_mean": float(kid_mean.item()),
        f"kid_{label}_std": float(kid_std.item()),
        f"lpips_{label}_mean": float(np.mean(lpips_values)),
        f"content_cosine_{label}_mean": float(np.mean(cosine_values)),
        f"evaluated_real_images_{label}": len(real_images),
        f"evaluated_generated_images_{label}": len(fake_images),
        f"images_per_second_{label}": len(fake_images) / max(elapsed, 1e-9),
    }


def main() -> None:
    monet_paths = sorted(gan.MONET_DIR.glob("*.jpg"))
    photo_paths = sorted(gan.PHOTO_DIR.glob("*.jpg"))[:gan.MAX_PHOTOS]
    generated_paths = sorted(SUBMISSION_DIR.glob("*.jpg"))
    if not generated_paths:
        raise FileNotFoundError("Run evaluate_cyclegan.py before advanced_metrics.py.")
    device = gan.DEVICE
    start = time.perf_counter()
    # Use the checkpoint evaluate_cyclegan.py selected (and built the submission images from).
    best_epoch = json.loads((EVALUATION_DIR / "evaluation_metrics.json").read_text(encoding="utf-8"))["selected_epoch"]
    checkpoint = torch.load(gan.CHECKPOINT_DIR / f"cyclegan_epoch_{best_epoch}.pt", map_location=device, weights_only=True)
    # Domain A = Monet, domain B = photo, so generator_ab translates Monet -> photo.
    generator_ab = gan.Generator().to(device)
    generator_ab.load_state_dict(checkpoint["generator_ab_ema" if gan.EMA_DECAY and "generator_ab_ema" in checkpoint else "generator_ab"])
    reverse_paths = save_reverse_outputs(generator_ab, monet_paths)
    metrics = {}
    metrics.update(evaluate_direction(monet_paths, generated_paths, photo_paths, "photo_to_monet", device))
    metrics.update(evaluate_direction(photo_paths, reverse_paths, monet_paths, "monet_to_photo", device))
    fid_photo_to_monet, mifid_photo_to_monet, kaggle_mifid_photo_to_monet, precision_photo_to_monet, recall_photo_to_monet = notebook_fid_mifid(monet_paths, generated_paths, device)
    fid_monet_to_photo, mifid_monet_to_photo, kaggle_mifid_monet_to_photo, precision_monet_to_photo, recall_monet_to_photo = notebook_fid_mifid(photo_paths, reverse_paths, device)
    metrics.update({
        "notebook_fid_photo_to_monet": fid_photo_to_monet,
        "notebook_mifid_photo_to_monet": mifid_photo_to_monet,
        "kaggle_style_mifid_photo_to_monet": kaggle_mifid_photo_to_monet,
        "generative_precision_photo_to_monet": precision_photo_to_monet,
        "generative_recall_photo_to_monet": recall_photo_to_monet,
        "notebook_fid_monet_to_photo": fid_monet_to_photo,
        "notebook_mifid_monet_to_photo": mifid_monet_to_photo,
        "kaggle_style_mifid_monet_to_photo": kaggle_mifid_monet_to_photo,
        "generative_precision_monet_to_photo": precision_monet_to_photo,
        "generative_recall_monet_to_photo": recall_monet_to_photo,
        "notebook_mean_fid": (fid_photo_to_monet + fid_monet_to_photo) / 2.0,
        "notebook_mean_mifid": (mifid_photo_to_monet + mifid_monet_to_photo) / 2.0,
        "notebook_evaluation_count": min(NOTEBOOK_EVAL_COUNT, len(monet_paths), len(photo_paths)),
    })
    metrics.update({"evaluation_seconds": time.perf_counter() - start, "note": f"Generative precision/recall: k-NN (k={PRECISION_RECALL_K}) manifold estimate on Inception-v3 pool features over {NOTEBOOK_EVAL_COUNT} images per set. notebook_mifid follows the class Part3_Evaluation_Script.ipynb (used for submission.csv); kaggle_style_mifid uses cosine-distance threshold {MIFID_COSINE_EPS} and is reference only."})
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (EVALUATION_DIR / "submission.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader()
        writer.writerow({"ID": 1, "FID": metrics["notebook_mean_fid"], "MiFID": metrics["notebook_mean_mifid"]})
    with (EVALUATION_DIR / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()