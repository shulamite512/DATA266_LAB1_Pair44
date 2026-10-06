from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from diff_augment import diff_augment

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # task3_gan/ (this file is task3_gan/shulamite/src/)
DATA_ROOT = PROJECT_ROOT / "data"
MONET_DIR = DATA_ROOT / "monet_jpg"
PHOTO_DIR = DATA_ROOT / "photo_jpg"

# Every run is defined here; select one with `--run <name>` (default: run2) on any of the three scripts.
RUNS = {
    # Original run: 30 epochs, LR decays linearly from epoch 1, resize + flip only, fp32.
    "run1": {"epochs": 30, "decay_start_epoch": 0, "load_size": 256, "mixed_precision": False, "num_workers": 0, "save_every": 1, "select_checkpoint_by": "cycle", "folder": ""},
    # Run 2: paper-style schedule (constant LR, then linear decay), resize-286 + random-crop-256 + flip, bf16 autocast.
    "run2": {"epochs": 100, "decay_start_epoch": 50, "load_size": 286, "mixed_precision": True, "num_workers": 4, "save_every": 5, "select_checkpoint_by": "fid", "folder": "run2"},
    # Run 3: run2 + EMA of generator weights (decay 0.999), which is what gets evaluated and submitted.
    "run3": {"epochs": 100, "decay_start_epoch": 50, "load_size": 286, "mixed_precision": True, "num_workers": 4, "save_every": 5, "select_checkpoint_by": "fid", "folder": "run3", "ema_decay": 0.999},
    # Run 4: run3 + DiffAugment (color, translation, cutout) on all discriminator inputs, against D memorising 300 Monets.
    "run4": {"epochs": 100, "decay_start_epoch": 50, "load_size": 286, "mixed_precision": True, "num_workers": 4, "save_every": 5, "select_checkpoint_by": "fid", "folder": "run4", "ema_decay": 0.999, "diff_augment": "color,translation,cutout"},
    # Quick end-to-end pipeline check (a few minutes): run2 settings on 64 photos for 2 epochs.
    "smoke": {"epochs": 2, "decay_start_epoch": 1, "load_size": 286, "mixed_precision": True, "num_workers": 2, "save_every": 1, "select_checkpoint_by": "fid", "folder": "smoke", "max_photos": 64, "ema_decay": 0.999, "diff_augment": "color,translation,cutout"},
}
RUN_NAME = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "run2"
if RUN_NAME not in RUNS:
    raise SystemExit(f"Unknown run '{RUN_NAME}'. Choose one of: {', '.join(RUNS)} (e.g. python train_cyclegan.py --run run2)")
RUN = RUNS[RUN_NAME]
RUN_ROOT = PROJECT_ROOT / "shulamite" / RUN["folder"]
OUTPUT_DIR = RUN_ROOT / "outputs"
CHECKPOINT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
LOAD_SIZE = RUN["load_size"]
BATCH_SIZE = 8
EPOCHS = RUN["epochs"]
DECAY_START_EPOCH = RUN["decay_start_epoch"]
MIXED_PRECISION = RUN["mixed_precision"] and DEVICE.type == "cuda"
MAX_PHOTOS = RUN.get("max_photos", 8000)
CYCLE_WEIGHT = RUN.get("cycle_weight", 10.0)
IDENTITY_WEIGHT = RUN.get("identity_weight", 5.0)
EMA_DECAY = RUN.get("ema_decay")
DIFF_AUGMENT_POLICY = RUN.get("diff_augment", "")
SEED = 266
RUN_EVALUATION_AFTER_TRAINING = True


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def start_run_log() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Run started {time.strftime('%Y-%m-%d %H:%M:%S')} | torch {torch.__version__} | python {sys.version.split()[0]}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Run: {RUN_NAME} | config: {json.dumps(RUN)}")
    print(f"Config: image_size={IMAGE_SIZE} load_size={LOAD_SIZE} batch_size={BATCH_SIZE} epochs={EPOCHS} decay_start_epoch={DECAY_START_EPOCH} mixed_precision={MIXED_PRECISION} max_photos={MAX_PHOTOS} seed={SEED}")
    return log_path


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def image_to_tensor(path: Path, augment: bool = False) -> torch.Tensor:
    # Evaluation: plain resize to IMAGE_SIZE. Training: resize to LOAD_SIZE, random IMAGE_SIZE crop, random flip.
    image = Image.open(path).convert("RGB")
    if not augment:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    else:
        image = image.resize((LOAD_SIZE, LOAD_SIZE))
        if LOAD_SIZE > IMAGE_SIZE:
            left, top = random.randint(0, LOAD_SIZE - IMAGE_SIZE), random.randint(0, LOAD_SIZE - IMAGE_SIZE)
            image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class UnpairedDataset(Dataset):
    def __init__(self, monet_paths: list[Path], photo_paths: list[Path]):
        self.monet_paths = monet_paths
        self.photo_paths = photo_paths

    def __len__(self) -> int:
        return max(len(self.monet_paths), len(self.photo_paths))

    def __getitem__(self, index: int):
        monet_path = self.monet_paths[index % len(self.monet_paths)]
        photo_path = self.photo_paths[index % len(self.photo_paths)]
        return image_to_tensor(monet_path, True), image_to_tensor(photo_path, True)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True)]
        layers += [nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(True)]
        layers += [nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(True)]
        layers += [ResidualBlock(256) for _ in range(9)]
        layers += [nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(64), nn.ReLU(True)]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path, quality=95)


class ReplayBuffer:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.images: list[torch.Tensor] = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        returned = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                returned.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.capacity)
                returned.append(self.images[index].clone())
                self.images[index] = image.clone()
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


def autocast():
    # bf16 autocast keeps fp32 range, so no gradient scaler is needed; disabled for fp32 runs.
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=MIXED_PRECISION)


def learning_rate_factor(epoch: int) -> float:
    if RUN_NAME == "run1":
        return max(0.0, 1.0 - epoch / max(EPOCHS - 1, 1))
    # Constant until DECAY_START_EPOCH, then linear decay that stays above 0 for the final epoch.
    return 1.0 - max(0, epoch - DECAY_START_EPOCH) / (EPOCHS - DECAY_START_EPOCH)


def augment(images: torch.Tensor) -> torch.Tensor:
    # DiffAugment on discriminator inputs only (no-op when the run has no policy).
    return diff_augment(images, DIFF_AUGMENT_POLICY)


@torch.no_grad()
def update_ema(models: dict[str, nn.Module]) -> None:
    # Exponential moving average of generator weights; the EMA copies are what evaluation uses.
    if not EMA_DECAY:
        return
    for name in ("generator_ab", "generator_ba"):
        ema_parameters = list(models[f"{name}_ema"].parameters())
        torch._foreach_lerp_(ema_parameters, [parameter.detach() for parameter in models[name].parameters()], 1.0 - EMA_DECAY)


def train_epoch(models: dict[str, nn.Module], loader: DataLoader, optimizers: dict[str, torch.optim.Optimizer], loss_gan: nn.Module, loss_cycle: nn.Module, fake_a_buffer: ReplayBuffer, fake_b_buffer: ReplayBuffer) -> dict[str, float]:
    generator_ab, generator_ba = models["generator_ab"], models["generator_ba"]
    discriminator_a, discriminator_b = models["discriminator_a"], models["discriminator_b"]
    totals = {key: 0.0 for key in ("generator", "discriminator", "cycle", "identity")}
    totals["gradient_norm"] = 0.0
    totals["nan_count"] = 0.0
    for real_a, real_b in loader:
        real_a, real_b = real_a.to(DEVICE, non_blocking=True), real_b.to(DEVICE, non_blocking=True)
        with autocast():
            fake_a = generator_ba(real_b)
            fake_b = generator_ab(real_a)
            cycle_a = generator_ba(fake_b)
            cycle_b = generator_ab(fake_a)
            identity_a = generator_ba(real_a)
            identity_b = generator_ab(real_b)
            prediction_fake_b, prediction_fake_a = discriminator_b(augment(fake_b)), discriminator_a(augment(fake_a))
            generator_loss = loss_gan(prediction_fake_b, torch.ones_like(prediction_fake_b)) + loss_gan(prediction_fake_a, torch.ones_like(prediction_fake_a))
            cycle_loss = loss_cycle(cycle_a, real_a) + loss_cycle(cycle_b, real_b)
            identity_loss = loss_cycle(identity_a, real_a) + loss_cycle(identity_b, real_b)
            total_generator_loss = generator_loss + CYCLE_WEIGHT * cycle_loss + IDENTITY_WEIGHT * identity_loss
        optimizers["generators"].zero_grad()
        total_generator_loss.backward()
        generator_gradient_norm = torch.sqrt(sum(parameter.grad.detach().pow(2).sum() for parameter in optimizers["generators"].param_groups[0]["params"] if parameter.grad is not None))
        if not torch.isfinite(total_generator_loss) or not torch.isfinite(generator_gradient_norm):
            totals["nan_count"] += 1
        optimizers["generators"].step()
        update_ema(models)
        with autocast():
            real_prediction_a = discriminator_a(augment(real_a))
            fake_prediction_a = discriminator_a(augment(fake_a_buffer.push_and_pop(fake_a)))
            real_prediction_b = discriminator_b(augment(real_b))
            fake_prediction_b = discriminator_b(augment(fake_b_buffer.push_and_pop(fake_b)))
            discriminator_loss = (loss_gan(real_prediction_a, torch.ones_like(real_prediction_a)) + loss_gan(fake_prediction_a, torch.zeros_like(fake_prediction_a)) + loss_gan(real_prediction_b, torch.ones_like(real_prediction_b)) + loss_gan(fake_prediction_b, torch.zeros_like(fake_prediction_b))) * 0.5
        optimizers["discriminators"].zero_grad()
        discriminator_loss.backward()
        discriminator_gradient_norm = torch.sqrt(sum(parameter.grad.detach().pow(2).sum() for parameter in optimizers["discriminators"].param_groups[0]["params"] if parameter.grad is not None))
        if not torch.isfinite(discriminator_loss) or not torch.isfinite(discriminator_gradient_norm):
            totals["nan_count"] += 1
        optimizers["discriminators"].step()
        totals["gradient_norm"] += max(generator_gradient_norm.item(), discriminator_gradient_norm.item())
        for key, value in {"generator": total_generator_loss, "discriminator": discriminator_loss, "cycle": cycle_loss, "identity": identity_loss}.items():
            totals[key] += value.item()
    averaged = {key: value / len(loader) for key, value in totals.items()}
    averaged["nan_count"] = totals["nan_count"]
    return averaged


def write_training_metrics(models: dict[str, nn.Module], history: list[dict], start: float, pairs_per_epoch: int) -> None:
    training_seconds = time.perf_counter() - start
    metrics = {
        "run": RUN_NAME,
        "config": RUN,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "epochs": EPOCHS,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "training_image_pairs_per_second": pairs_per_epoch * len(history) / max(training_seconds, 1e-9),
        "parameter_count": sum(parameter.numel() for name, model in models.items() if not name.endswith("_ema") for parameter in model.parameters()),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2) if DEVICE.type == "cuda" else None,
        "total_nan_count": sum(epoch["nan_count"] for epoch in history),
        "max_gradient_norm": max(epoch["gradient_norm"] for epoch in history),
        "history": history,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_attached_evaluation() -> None:
    from evaluate_cyclegan import main as generate_translations
    from advanced_metrics import main as calculate_metrics
    from plot_training import main as plot_training

    generate_translations()
    calculate_metrics()
    plot_training()


def main() -> None:
    start_run_log()
    set_seed()
    if not MONET_DIR.exists() or not PHOTO_DIR.exists():
        raise FileNotFoundError("Expected data/monet_jpg and data/photo_jpg in the current project folder.")
    monet_paths = sorted(MONET_DIR.glob("*.jpg"))
    photo_paths = sorted(PHOTO_DIR.glob("*.jpg"))[:MAX_PHOTOS]
    if len(photo_paths) < min(200, MAX_PHOTOS):
        raise ValueError(f"Need at least {min(200, MAX_PHOTOS)} photo samples, found {len(photo_paths)}.")
    dataset = UnpairedDataset(monet_paths, photo_paths)
    workers = RUN["num_workers"]
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=DEVICE.type == "cuda", persistent_workers=workers > 0)
    torch.backends.cudnn.benchmark = True
    models = {"generator_ab": Generator().to(DEVICE), "generator_ba": Generator().to(DEVICE), "discriminator_a": Discriminator().to(DEVICE), "discriminator_b": Discriminator().to(DEVICE)}
    if EMA_DECAY:
        for name in ("generator_ab", "generator_ba"):
            models[f"{name}_ema"] = copy.deepcopy(models[name]).requires_grad_(False)
    optimizers = {"generators": torch.optim.Adam(list(models["generator_ab"].parameters()) + list(models["generator_ba"].parameters()), lr=2e-4, betas=(0.5, 0.999)), "discriminators": torch.optim.Adam(list(models["discriminator_a"].parameters()) + list(models["discriminator_b"].parameters()), lr=2e-4, betas=(0.5, 0.999))}
    schedulers = {name: torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor) for name, optimizer in optimizers.items()}
    fake_a_buffer, fake_b_buffer = ReplayBuffer(), ReplayBuffer()
    loss_gan, loss_cycle = nn.MSELoss(), nn.L1Loss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.perf_counter()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    print(f"Using device: {DEVICE}; Monet images: {len(monet_paths)}; photo images: {len(photo_paths)}")
    for epoch in range(EPOCHS):
        trained_learning_rate = optimizers["generators"].param_groups[0]["lr"]
        losses = train_epoch(models, loader, optimizers, loss_gan, loss_cycle, fake_a_buffer, fake_b_buffer)
        for scheduler in schedulers.values():
            scheduler.step()
        losses["epoch"] = epoch + 1
        losses["learning_rate"] = optimizers["generators"].param_groups[0]["lr"]
        losses["trained_learning_rate"] = trained_learning_rate
        history.append(losses)
        if (epoch + 1) % RUN["save_every"] == 0 or epoch + 1 == EPOCHS:
            torch.save({name: model.state_dict() for name, model in models.items()}, CHECKPOINT_DIR / f"cyclegan_epoch_{epoch + 1}.pt")
        write_training_metrics(models, history, start, len(dataset))
        with torch.no_grad():
            real_a, real_b = next(iter(loader))
            save_image(models["generator_ab"](real_a[:1].to(DEVICE))[0], OUTPUT_DIR / f"monet_to_photo_epoch_{epoch + 1}.jpg")
            save_image(models["generator_ba"](real_b[:1].to(DEVICE))[0], OUTPUT_DIR / f"photo_to_monet_epoch_{epoch + 1}.jpg")
        print(f"Epoch {epoch + 1}/{EPOCHS} | generator={losses['generator']:.4f} | discriminator={losses['discriminator']:.4f} | cycle={losses['cycle']:.4f} | identity={losses['identity']:.4f} | grad_norm={losses['gradient_norm']:.2f} | nan={int(losses['nan_count'])} | lr={trained_learning_rate:.2e} | elapsed={(time.perf_counter() - start) / 60:.1f}min", flush=True)
    if RUN_EVALUATION_AFTER_TRAINING:
        run_attached_evaluation()


if __name__ == "__main__":
    main()