"""Differentiable augmentation for data-efficient GAN training (Zhao et al., 2020, "Differentiable
Augmentation for Data-Efficient GAN Training"). Applied to every image the discriminators see, real and
fake, so a discriminator cannot memorise the small Monet set (300 images). Inputs are in [-1, 1]."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str) -> torch.Tensor:
    if not policy:
        return x
    for name in policy.split(","):
        for function in AUGMENTS[name]:
            x = function(x)
    return x.contiguous()


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cutout_size[0], device=x.device),
        torch.arange(cutout_size[1], device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


AUGMENTS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
