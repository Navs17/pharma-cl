"""Image transforms and dataset helpers (binary good/defective ImageFolders)."""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config


def build_transforms(train: bool):
    """Augment on train (small dataset -> augmentation matters), plain on eval."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def product_folder(product: str, split: str) -> Path:
    return config.PROCESSED_DIR / product / split


def load_imagefolder(product: str, split: str, train_aug: bool | None = None) -> ImageFolder:
    """Return an ImageFolder. class_to_idx is enforced as good=0, defective=1."""
    if train_aug is None:
        train_aug = (split == "train")
    ds = ImageFolder(str(product_folder(product, split)),
                     transform=build_transforms(train_aug))
    # ImageFolder sorts alphabetically -> defective=0, good=1. Force our convention.
    desired = {"good": 0, "defective": 1}
    if ds.class_to_idx != desired:
        remap = {ds.class_to_idx[c]: desired[c] for c in ds.classes}
        ds.samples = [(p, remap[y]) for p, y in ds.samples]
        ds.targets = [remap[y] for y in ds.targets]
        ds.class_to_idx = desired
        ds.classes = ["good", "defective"]
    return ds


def make_loader(ds, train: bool):
    return torch.utils.data.DataLoader(
        ds, batch_size=config.BATCH_SIZE, shuffle=train,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
