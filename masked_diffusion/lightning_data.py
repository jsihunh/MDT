"""PyTorch Lightning data module for MDT training."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .image_datasets import ImageDataset, _list_image_files_recursively
from .script_util import NUM_CLASSES


class MDTDataModule(pl.LightningDataModule):
    """Lightning data module that reproduces the original training loader."""

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
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_cond = class_cond
        self.num_workers = num_workers
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.deterministic = deterministic
        self._dataset: ImageDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._dataset is not None:
            return
        files = _list_image_files_recursively(self.data_dir)
        if not files:
            raise ValueError(f"No image files were found in {self.data_dir}")
        classes = None
        if self.class_cond:
            class_tokens = [file.split("/")[-1].split("_")[0] for file in files]
            unique = {token: idx for idx, token in enumerate(sorted(set(class_tokens)))}
            classes = [unique[token] for token in class_tokens]
        self._dataset = ImageDataset(
            self.image_size,
            files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._dataset is not None
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=not self.deterministic,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch: List[Tuple[np.ndarray, Dict[str, np.ndarray]]]):
        images = torch.from_numpy(np.stack([item[0] for item in batch])).float()
        cond_dict: Dict[str, List[np.ndarray]] = {}
        for _, cond in batch:
            for key, value in cond.items():
                cond_dict.setdefault(key, []).append(value)
        tensor_cond: Dict[str, torch.Tensor] = {}
        for key, values in cond_dict.items():
            stacked = np.stack(values)
            tensor = torch.from_numpy(stacked)
            if tensor.dtype in (torch.int32, torch.int64):
                tensor = tensor.long()
            elif tensor.dtype in (torch.float32, torch.float64):
                tensor = tensor.float()
            else:
                tensor = tensor.float()
            tensor_cond[key] = tensor
        if "y" not in tensor_cond:
            zeros = torch.zeros(images.shape[0], dtype=torch.long)
            tensor_cond["y"] = zeros
        tensor_cond["y"] = torch.clamp(tensor_cond["y"], 0, NUM_CLASSES)
        return images, tensor_cond
