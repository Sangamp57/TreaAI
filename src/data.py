"""Datasets, transforms and splits for the Yunnan species image pack."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(0.25, 0.25, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Resize(int(round(size * 1.14))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SpeciesDataset(Dataset):
    """Reads images from disk once into a PIL cache (the whole pack is ~57 MB)."""

    def __init__(self, frame: pd.DataFrame, classes: list[str], size: int, train: bool,
                 cache: dict[str, Image.Image] | None = None):
        self.frame = frame.reset_index(drop=True)
        self.classes = classes
        self.index = {c: i for i, c in enumerate(classes)}
        self.tf = build_transforms(size, train)
        self.cache = cache if cache is not None else {}

    def __len__(self) -> int:
        return len(self.frame)

    def _image(self, path: str) -> Image.Image:
        img = self.cache.get(path)
        if img is None:
            img = Image.open(path).convert("RGB")
            self.cache[path] = img
        return img

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        return self.tf(self._image(str(row["path"]))), self.index[row["species_code"]]


def load_split(val_frac: float = 0.2, seed: int = config.SEED):
    """Fixed test split from the pack manifest; carve a stratified validation split from train.

    The provided test set is never used for training or threshold tuning.
    """
    df = config.load_manifest()
    classes = config.class_names()
    df = df[df["exists"]].copy()
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # one image per class is not enough for a stable val split, so stratify on species
    train_idx, val_idx = train_test_split(
        train_df.index, test_size=val_frac, stratify=train_df["species_code"], random_state=seed
    )
    return (
        train_df.loc[train_idx].reset_index(drop=True),
        train_df.loc[val_idx].reset_index(drop=True),
        test_df.reset_index(drop=True),
        classes,
    )


def make_loader(frame, classes, size, train, batch_size=32, shuffle=None):
    ds = SpeciesDataset(frame, classes, size, train)
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=False, drop_last=False)


def hard_case_frame() -> pd.DataFrame:
    """Hard-case images are stored as hard_cases/images/<CODE>/*.jpg, so the folder is the label."""
    rows = []
    if config.HARD_DIR.exists():
        for p in sorted(config.HARD_DIR.glob("*/*.jpg")):
            rows.append({"path": p, "species_code": p.parent.name, "split": "hard"})
    return pd.DataFrame(rows)


def set_seed(seed: int = config.SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    tr, va, te, classes = load_split()
    print(f"train={len(tr)} val={len(va)} test={len(te)} classes={len(classes)}")
    print(tr.groupby("species_code").size().describe())
    hard = hard_case_frame()
    print(f"hard cases={len(hard)} species={hard['species_code'].nunique()}")
    ds = SpeciesDataset(tr, classes, 256, True)
    x, y = ds[0]
    print("batch item:", tuple(x.shape), y, classes[y])
