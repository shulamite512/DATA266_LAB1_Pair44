from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_cyclegan as gan

EVALUATION_DIR = gan.OUTPUT_DIR / "evaluation"
TRANSLATION_DIR = EVALUATION_DIR / "translations"
SUBMISSION_DIR = EVALUATION_DIR / "submission_images"


def load_models(checkpoint_path: Path) -> dict[str, torch.nn.Module]:
    models = {
        "generator_ab": gan.Generator().to(gan.DEVICE),
        "generator_ba": gan.Generator().to(gan.DEVICE),
        "discriminator_a": gan.Discriminator().to(gan.DEVICE),
        "discriminator_b": gan.Discriminator().to(gan.DEVICE),
    }
    checkpoint = torch.load(checkpoint_path, map_location=gan.DEVICE, weights_only=True)
    for name, model in models.items():
        # Runs trained with EMA are evaluated with the averaged generator weights.
        key = f"{name}_ema" if gan.EMA_DECAY and f"{name}_ema" in checkpoint else name
        model.load_state_dict(checkpoint[key])
        model.eval()
    return models


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def cycle_l1(paths: list[Path], forward: torch.nn.Module, backward: torch.nn.Module) -> float:
    distances = []
    with torch.no_grad():
        for path in paths:
            original = gan.image_to_tensor(path).unsqueeze(0).to(gan.DEVICE)
            reconstructed = backward(forward(original))
            distances.append(torch.mean(torch.abs(original - reconstructed)).item())
    return float(np.mean(distances))


def save_audit_samples(models: dict[str, torch.nn.Module], monet_paths: list[Path], photo_paths: list[Path]) -> None:
    if TRANSLATION_DIR.exists():
        shutil.rmtree(TRANSLATION_DIR)
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (monet_path, photo_path) in enumerate(zip(monet_paths[:30], photo_paths[:30]), start=1):
        monet = gan.image_to_tensor(monet_path).unsqueeze(0).to(gan.DEVICE)
        photo = gan.image_to_tensor(photo_path).unsqueeze(0).to(gan.DEVICE)
        with torch.no_grad():
            photo_to_monet = models["generator_ba"](photo)[0]
            monet_to_photo = models["generator_ab"](monet)[0]
        Image.open(monet_path).convert("RGB").resize((gan.IMAGE_SIZE, gan.IMAGE_SIZE)).save(TRANSLATION_DIR / f"sample_{index:02d}_input_monet.jpg")
        Image.open(photo_path).convert("RGB").resize((gan.IMAGE_SIZE, gan.IMAGE_SIZE)).save(TRANSLATION_DIR / f"sample_{index:02d}_input_photo.jpg")
        tensor_to_image(photo_to_monet).save(TRANSLATION_DIR / f"sample_{index:02d}_photo_to_monet.jpg")
        tensor_to_image(monet_to_photo).save(TRANSLATION_DIR / f"sample_{index:02d}_monet_to_photo.jpg")
        manifest.append({"sample": index, "input_monet": monet_path.name, "input_photo": photo_path.name, "rater_1_style": None, "rater_1_content": None, "rater_1_artifacts": None, "rater_2_style": None, "rater_2_content": None, "rater_2_artifacts": None})
    (EVALUATION_DIR / "human_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_submission(models: dict[str, torch.nn.Module], photo_paths: list[Path]) -> None:
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, path in enumerate(photo_paths):
            translated = models["generator_ba"](gan.image_to_tensor(path).unsqueeze(0).to(gan.DEVICE))[0]
            tensor_to_image(translated).save(SUBMISSION_DIR / f"{index:05d}.jpg", quality=95)
    with zipfile.ZipFile(EVALUATION_DIR / "images.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image_path in sorted(SUBMISSION_DIR.glob("*.jpg")):
            archive.write(image_path, image_path.name)


def checkpoint_fid(checkpoint_path: Path, monet_paths: list[Path], photo_paths: list[Path]) -> dict:
    from advanced_metrics import NOTEBOOK_EVAL_COUNT, notebook_fid_mifid

    models = load_models(checkpoint_path)
    scratch = EVALUATION_DIR / "checkpoint_selection_tmp"
    results = {}
    for label, generator, sources, reals in (("photo_to_monet", models["generator_ba"], photo_paths, monet_paths), ("monet_to_photo", models["generator_ab"], monet_paths, photo_paths)):
        folder = scratch / label
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        generated = []
        with torch.no_grad():
            for index, path in enumerate(sources[:NOTEBOOK_EVAL_COUNT]):
                target = folder / f"{index:05d}.jpg"
                tensor_to_image(generator(gan.image_to_tensor(path).unsqueeze(0).to(gan.DEVICE))[0]).save(target, quality=95)
                generated.append(target)
        results[f"fid_{label}"], results[f"mifid_{label}"], *_ = notebook_fid_mifid(reals, generated, gan.DEVICE)
    shutil.rmtree(scratch)
    results["mean_fid"] = (results["fid_photo_to_monet"] + results["fid_monet_to_photo"]) / 2.0
    return results


def select_checkpoint(training_metrics: dict, monet_paths: list[Path], photo_paths: list[Path]) -> int:
    if gan.RUN["select_checkpoint_by"] == "cycle":
        return min(training_metrics["history"], key=lambda item: item["cycle"])["epoch"]
    # Score every saved checkpoint from the LR-decay phase with the class-notebook FID (mean of both directions).
    candidates = sorted((int(path.stem.rsplit("_", 1)[1]), path) for path in gan.CHECKPOINT_DIR.glob("cyclegan_epoch_*.pt"))
    candidates = [(epoch, path) for epoch, path in candidates if epoch >= gan.DECAY_START_EPOCH] or candidates
    scores = []
    for epoch, path in candidates:
        result = {"epoch": epoch, **checkpoint_fid(path, monet_paths, photo_paths)}
        print(f"Checkpoint epoch {epoch}: mean FID={result['mean_fid']:.2f} (photo->monet {result['fid_photo_to_monet']:.2f}, monet->photo {result['fid_monet_to_photo']:.2f})", flush=True)
        scores.append(result)
    best = min(scores, key=lambda item: item["mean_fid"])
    (EVALUATION_DIR / "checkpoint_selection.json").write_text(json.dumps({"criterion": "class-notebook mean FID (300 images per set)", "selected_epoch": best["epoch"], "scores": scores}, indent=2), encoding="utf-8")
    return best["epoch"]


def main() -> None:
    training_metrics = json.loads((gan.OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    best_epoch = select_checkpoint(training_metrics, sorted(gan.MONET_DIR.glob("*.jpg")), sorted(gan.PHOTO_DIR.glob("*.jpg"))[:gan.MAX_PHOTOS])
    checkpoint_path = gan.CHECKPOINT_DIR / f"cyclegan_epoch_{best_epoch}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models(checkpoint_path)
    monet_paths = sorted(gan.MONET_DIR.glob("*.jpg"))
    photo_paths = sorted(gan.PHOTO_DIR.glob("*.jpg"))[:gan.MAX_PHOTOS]
    metrics = {"checkpoint": checkpoint_path.relative_to(gan.PROJECT_ROOT).as_posix(), "selected_epoch": best_epoch, "photo_to_monet_to_photo_l1": cycle_l1(photo_paths[:100], models["generator_ba"], models["generator_ab"]), "monet_to_photo_to_monet_l1": cycle_l1(monet_paths[:100], models["generator_ab"], models["generator_ba"]), "audit_sample_count": 30, "submission_image_count": len(photo_paths)}
    (EVALUATION_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_audit_samples(models, monet_paths, photo_paths)
    build_submission(models, photo_paths)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {EVALUATION_DIR}")


if __name__ == "__main__":
    main()