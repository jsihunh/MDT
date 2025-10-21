"""PyTorch Lightning integration for MDT training and sampling."""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl

from diffusers.models import AutoencoderKL

from . import create_diffusion
from .resample import LossAwareSampler, create_named_schedule_sampler
from .train_util import INITIAL_LOG_LOSS_SCALE  # noqa: F401  # For backward compatibility

import masked_diffusion.models as models_mdt


try:
    from adan import Adan
except ImportError:  # pragma: no cover - fallback when Adan is unavailable.
    Adan = None  # type: ignore


class MDTLightningModule(pl.LightningModule):
    """Lightning module that encapsulates the MDT training loop."""

    def __init__(
        self,
        *,
        model_name: str,
        image_size: int,
        mask_ratio: Optional[float],
        decode_layer: Optional[int],
        diffusion_config: Dict[str, Any],
        class_cond: bool = True,
        schedule_sampler: str = "uniform",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        lr_anneal_steps: int = 0,
        opt_type: str = "adan",
        scale_factor: float = 0.18215,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
    ) -> None:
        super().__init__()
        latent_size = image_size // 8
        self.model = models_mdt.__dict__[model_name](
            input_size=latent_size,
            mask_ratio=mask_ratio,
            decode_layer=decode_layer,
        )
        self.diffusion_config = copy.deepcopy(diffusion_config)
        self.diffusion = create_diffusion(**diffusion_config)
        self.schedule_sampler_name = schedule_sampler
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.class_cond = class_cond
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.opt_type = opt_type.lower()
        self.scale_factor = scale_factor
        self.vae_model = vae_model

        self.first_stage_model: Optional[AutoencoderKL] = None

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "image_size": image_size,
                "mask_ratio": mask_ratio,
                "decode_layer": decode_layer,
                "diffusion_config": self.diffusion_config,
                "class_cond": class_cond,
                "schedule_sampler": schedule_sampler,
                "lr": lr,
                "weight_decay": weight_decay,
                "lr_anneal_steps": lr_anneal_steps,
                "opt_type": self.opt_type,
                "scale_factor": scale_factor,
                "vae_model": vae_model,
            }
        )

    def instantiate_first_stage(self) -> None:
        if self.first_stage_model is not None:
            return
        model = AutoencoderKL.from_pretrained(self.vae_model)
        model = model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        self.first_stage_model = model

    @torch.no_grad()
    def encode_first_stage(self, x: Tensor) -> Tensor:
        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        posterior = self.first_stage_model.encode(x, return_dict=True)[0]
        z = posterior.sample()
        return z * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, z: Tensor) -> Tensor:
        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        decoded = self.first_stage_model.decode(z / self.scale_factor).sample
        return decoded

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:  # type: ignore[override]
        images, cond = batch
        if cond is None:
            cond = {}
        images = images.to(self.device)
        latents = self.encode_first_stage(images).detach()

        cond_tensors: Dict[str, Tensor] = {}
        if isinstance(cond, dict):
            cond_tensors = {k: v.to(self.device) for k, v in cond.items()}

        t, weights = self.schedule_sampler.sample(latents.shape[0], self.device)

        kwargs_standard = dict(cond_tensors)
        kwargs_mask = dict(cond_tensors)
        kwargs_mask["enable_mask"] = True

        compute_losses = partial(
            self.diffusion.training_losses,
            self.model,
            latents,
            t,
        )

        losses = compute_losses(model_kwargs=kwargs_standard)
        losses_mask = compute_losses(model_kwargs=kwargs_mask)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            combined = losses["loss"].detach() + losses_mask["loss"].detach()
            self.schedule_sampler.update_with_local_losses(t, combined)

        weighted = (losses["loss"] * weights).mean()
        weighted_mask = (losses_mask["loss"] * weights).mean()
        total_loss = weighted + weighted_mask

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=latents.shape[0])
        self.log("denoising_loss", weighted, on_step=True, on_epoch=True, batch_size=latents.shape[0])
        self.log("mask_loss", weighted_mask, on_step=True, on_epoch=True, batch_size=latents.shape[0])

        return total_loss

    def configure_optimizers(self) -> Any:  # type: ignore[override]
        parameters = self.parameters()
        if self.opt_type == "adan":
            if Adan is None:
                raise ImportError("Adan optimizer is not installed. Please install sail-sg/Adan.")
            optimizer = Adan(parameters, lr=self.lr, weight_decay=self.weight_decay, fused=True, max_grad_norm=1.0)
        elif self.opt_type == "adamw":
            optimizer = AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.opt_type}")

        if self.lr_anneal_steps > 0:
            def lr_lambda(step: int) -> float:
                remaining = max(self.lr_anneal_steps - step, 0)
                return remaining / float(self.lr_anneal_steps)

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        strict: bool = True,
        map_location: Optional[str] = None,
        **kwargs: Any,
    ) -> "MDTLightningModule":
        try:
            return cls.load_from_checkpoint(checkpoint_path, map_location=map_location, strict=strict)
        except (RuntimeError, KeyError):
            if not kwargs:
                raise
            module = cls(**kwargs)
            state_dict = torch.load(checkpoint_path, map_location=map_location or "cpu")
            module.model.load_state_dict(state_dict)
            return module


class MDTDataModule(pl.LightningDataModule):
    """Lightning data module that reuses the existing MDT image datasets."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        image_size: int,
        class_cond: bool,
        random_crop: bool = False,
        random_flip: bool = True,
        num_workers: int = 4,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_cond = class_cond
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.num_workers = num_workers
        self.deterministic = deterministic
        self._dataset = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if self._dataset is not None:
            return
        from .image_datasets import ImageDataset, _list_image_files_recursively

        all_files = _list_image_files_recursively(self.data_dir)
        classes = None
        if self.class_cond:
            import blobfile as bf

            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        self._dataset = ImageDataset(
            self.image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=self.random_crop,
            random_flip=self.random_flip,
        )

    def train_dataloader(self):  # type: ignore[override]
        assert self._dataset is not None, "DataModule.setup must be called before requesting the dataloader."
        return torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=not self.deterministic,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

