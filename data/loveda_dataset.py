"""
LoveDA dataset wrapper for semantic segmentation training/evaluation.

Returns:
    image: FloatTensor (3, H, W), ImageNet-normalized
    mask : LongTensor  (H, W), values 0..7 where 0 is ignore/no-data
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

import config

try:
    from torchgeo.datasets import LoveDA
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import torchgeo LoveDA dataset. "
        "Install dependencies in requirements.txt first."
    ) from exc


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LoveDASemanticDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = config.IMAGE_SIZE,
        download: bool = False,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.dataset = LoveDA(root=root, split=split, download=download)
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        image = sample["image"].float() / 255.0  # (3, H, W)
        mask = sample["mask"].long()             # (H, W)

        if self.image_size is not None:
            image, mask = self._resize_pair(image, mask, self.image_size)

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=(2,))
                mask = torch.flip(mask, dims=(1,))
            if random.random() < 0.5:
                image = torch.flip(image, dims=(1,))
                mask = torch.flip(mask, dims=(0,))

        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        return image, mask

    @staticmethod
    def _resize_pair(
        image: torch.Tensor, mask: torch.Tensor, image_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
        return image, mask


def _build_subset(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")

    num_samples = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    return Subset(dataset, indices)


def get_datasets(
    root: str = config.DATA_ROOT,
    download: bool = False,
    subset_fraction: float = config.SUBSET_FRACTION,
    image_size: int = config.IMAGE_SIZE,
    seed: int = config.SEED,
) -> Tuple[Dataset, Dataset]:
    train_ds = LoveDASemanticDataset(
        root=root,
        split="train",
        image_size=image_size,
        download=download,
        augment=True,
        normalize=True,
    )
    val_ds = LoveDASemanticDataset(
        root=root,
        split="val",
        image_size=image_size,
        download=download,
        augment=False,
        normalize=True,
    )
    train_ds = _build_subset(train_ds, subset_fraction, seed=seed)
    return train_ds, val_ds

