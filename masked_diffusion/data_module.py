"""Lightning data module wrappers for MDT datasets."""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .image_datasets import (
    ImageDataset,
    _list_image_files_recursively,
)
import blobfile as bf


class ImageDataModule(pl.LightningDataModule):
    """Simple data module that reuses the existing image dataset utilities."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        image_size: int,
        class_cond: bool = True,
        num_workers: int = 4,
        random_crop: bool = False,
        random_flip: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_cond = class_cond
        self.num_workers = num_workers
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.dataset: Optional[ImageDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        del stage
        all_files = _list_image_files_recursively(self.data_dir)
        classes = None
        if self.class_cond:
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {name: idx for idx, name in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[name] for name in class_names]

        self.dataset = ImageDataset(
            self.image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
        )

    def train_dataloader(self):  # type: ignore[override]
        if self.dataset is None:
            raise RuntimeError("DataModule.setup must be called before requesting dataloaders.")
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

