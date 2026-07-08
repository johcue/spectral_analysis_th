"""
spectral.data
=============

Dataset / model construction and reproducible seeding. Requires torch +
torchvision. Kept separate from ``generate`` so the heavy IO/model setup is
easy to reuse or mock.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Subset


def seed_everything(seed: int = 42):
    """Seed python / numpy / torch. Call this *immediately* before any stage
    that consumes randomness (PGD random start, noise controls) so a stage can
    be re-run in isolation and still reproduce its CSV.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(cfg) -> DataLoader:
    """ImageNet-val subset loader in raw [0,1] pixel space (batch size 1)."""
    transform_raw = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(cfg.imagenet_val_dir, transform=transform_raw)
    dataset = Subset(dataset, list(range(cfg.n_images)))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def load_model(cfg, device):
    """ResNet-50 (ImageNet1K_V1) in eval mode + loss + normalization stats."""
    model = models.resnet50(weights="IMAGENET1K_V1").to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return model, criterion, mean, std
