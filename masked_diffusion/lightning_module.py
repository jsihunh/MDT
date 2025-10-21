import contextlib
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from diffusers.models import AutoencoderKL

from adan import Adan

from . import create_diffusion
from .image_datasets import ImageDataset, _list_image_files_recursively
from .models import __dict__ as model_registry
from .nn import update_ema
from .resample import LossAwareSampler, create_named_schedule_sampler


class MDTLightningModule(pl.LightningModule):
    """PyTorch Lightning module that encapsulates MDT training and sampling."""

    def __init__(
        self,
        *,
        model_name: str,
        image_size: int,
        mask_ratio: Optional[float],
        decode_layer: int,
        diffusion_kwargs: Dict[str, Any],
        lr: float,
        weight_decay: float,
        ema_rate: str,
        schedule_sampler: str,
        lr_anneal_steps: int,
        scale_factor: float = 0.18215,
        opt_type: str = "adan",
        class_cond: bool = True,
        enable_mask: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        latent_size = image_size // 8
        if model_name not in model_registry:
            raise ValueError(f"Unknown MDT model: {model_name}")
        self.model: nn.Module = model_registry[model_name](
            input_size=latent_size, mask_ratio=mask_ratio, decode_layer=decode_layer
        )
        self.diffusion = create_diffusion(**diffusion_kwargs)
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.scale_factor = scale_factor
        self.opt_type = opt_type
        self.class_cond = class_cond
        self.enable_mask = enable_mask

        if isinstance(ema_rate, float):
            ema_values: Sequence[float] = [ema_rate]
        else:
            ema_values = [float(x) for x in str(ema_rate).split(",") if x]
        self.ema_rate: List[float] = list(ema_values)
        self.ema_params: List[List[torch.Tensor]] = []

        self.first_stage_model = self._instantiate_first_stage_model()

    def to(self, *args: Any, **kwargs: Any) -> "MDTLightningModule":  # type: ignore[override]
        module = super().to(*args, **kwargs)
        if self.ema_params:
            device = next(self.model.parameters()).device
            self._move_ema_params(device)
        return module

    def _instantiate_first_stage_model(self) -> nn.Module:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        try:
            vae = torch.compile(vae)  # type: ignore[attr-defined]
        except Exception:
            # torch.compile is not available on some platforms; fall back to eager execution.
            pass
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:  # type: ignore[override]
        if not self.ema_params:
            self._reset_ema_params()
        else:
            for ema_params in self.ema_params:
                for idx, param in enumerate(self.model.parameters()):
                    ema_params[idx] = ema_params[idx].to(param.device)

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        images, cond = batch
        images = images.to(self.device)
        cond = {k: v.to(self.device) for k, v in cond.items()}
        if self.class_cond and "y" not in cond:
            raise ValueError("Class-conditional training requires labels (key 'y').")

        cond.setdefault("enable_mask", False)
        latents = self.encode_first_stage(images)
        timesteps, weights = self.schedule_sampler.sample(latents.shape[0], self.device)

        losses = self.diffusion.training_losses(
            self.model,
            latents,
            timesteps,
            model_kwargs=cond,
        )

        cond_mask = dict(cond)
        if self.enable_mask:
            cond_mask["enable_mask"] = True
        losses_mask = self.diffusion.training_losses(
            self.model,
            latents,
            timesteps,
            model_kwargs=cond_mask,
        )

        combined_loss = (losses["loss"] * weights).mean()
        combined_loss_mask = (losses_mask["loss"] * weights).mean()
        loss = combined_loss + combined_loss_mask

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                timesteps, losses["loss"].detach() + losses_mask["loss"].detach()
            )

        self.log("train/loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self._log_loss_dict("train/model", losses, weights)
        self._log_loss_dict("train/masked", losses_mask, weights)
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.opt_type.lower() == "adamw":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_type.lower() == "adan":
            optimizer = Adan(params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1, fused=True)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.opt_type}")

        if self.lr_anneal_steps > 0:
            def lr_lambda(step: int) -> float:
                if step >= self.lr_anneal_steps:
                    return 0.0
                return 1 - (step / float(self.lr_anneal_steps))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):  # type: ignore[override]
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
        self._update_ema()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        checkpoint["ema_params"] = [
            [param.detach().cpu() for param in ema_params]
            for ema_params in self.ema_params
        ]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:  # type: ignore[override]
        ema_params = checkpoint.get("ema_params")
        if ema_params:
            self.ema_params = [
                [tensor.to(self.device) for tensor in ema_list]
                for ema_list in ema_params
            ]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        posterior = self.first_stage_model.encode(x, return_dict=True)[0]
        return posterior.sample().to(self.device) * self.scale_factor

    def decode_first_stage(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.scale_factor
        decoded = self.first_stage_model.decode(latents).sample
        return decoded

    def sample(
        self,
        *,
        class_labels: Optional[Iterable[int]] = None,
        batch_size: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        pow_scale: float = 4.0,
        ema_index: Optional[int] = 0,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate images using the trained model."""

        if batch_size is None:
            if class_labels is None:
                raise ValueError("Either batch_size or class_labels must be provided for sampling.")
            batch_size = len(list(class_labels))
        device = self.device

        if latent is None:
            latent_size = self.hparams.image_size // 8
            latent = torch.randn(batch_size, 4, latent_size, latent_size, device=device)

        if class_labels is None:
            fill_value = 1000 if self.class_cond else 0
            y = torch.full((batch_size,), fill_value=fill_value, device=device, dtype=torch.long)
        else:
            if not self.class_cond:
                raise ValueError("Class labels were provided, but the model was trained without class conditioning.")
            labels = list(class_labels)
            if len(labels) != batch_size:
                raise ValueError("The number of class labels must match the batch size.")
            y = torch.tensor(labels, device=device, dtype=torch.long)

        if cfg_scale is not None and self.class_cond:
            latent = torch.cat([latent, latent], dim=0)
            y_null = torch.full((batch_size,), 1000, device=device, dtype=torch.long)
            y = torch.cat([y, y_null], dim=0)

        model_kwargs = {
            "y": y,
            "cfg_scale": cfg_scale,
            "scale_pow": pow_scale,
            "enable_mask": False,
            "diffusion_steps": diffusion.num_timesteps,
        }

        diffusion = self.diffusion
        if num_sampling_steps and num_sampling_steps != self.diffusion.num_timesteps:
            diffusion_config = dict(self.hparams.diffusion_kwargs)
            diffusion_config["timestep_respacing"] = str(num_sampling_steps)
            diffusion = create_diffusion(**diffusion_config)

        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            stack.enter_context(self.ema_scope(ema_index))
            samples = diffusion.p_sample_loop(
                self.model.forward_with_cfg,
                latent.shape,
                latent,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
        if cfg_scale is not None and self.class_cond:
            samples, _ = samples.chunk(2, dim=0)
        decoded = self.decode_first_stage(samples)
        return decoded

    @contextlib.contextmanager
    def ema_scope(self, ema_index: Optional[int] = 0):
        if ema_index is None or not self.ema_params:
            yield
            return

        if ema_index >= len(self.ema_params):
            raise ValueError(f"EMA index {ema_index} is out of bounds for {len(self.ema_params)} EMA rates.")

        backup_params = [param.detach().clone() for param in self.model.parameters()]
        try:
            for param, ema_param in zip(self.model.parameters(), self.ema_params[ema_index]):
                param.data.copy_(ema_param)
            yield
        finally:
            for param, backup in zip(self.model.parameters(), backup_params):
                param.data.copy_(backup)

    def _reset_ema_params(self) -> None:
        self.ema_params = []
        for _ in self.ema_rate:
            ema_params = [param.detach().clone().to(param.device) for param in self.model.parameters()]
            for tensor in ema_params:
                tensor.requires_grad = False
            self.ema_params.append(ema_params)

    def _update_ema(self) -> None:
        if not self.ema_params:
            return
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _move_ema_params(self, device: torch.device) -> None:
        for ema_params in self.ema_params:
            for idx, tensor in enumerate(ema_params):
                ema_params[idx] = tensor.to(device)

    def _log_loss_dict(self, prefix: str, losses: Dict[str, torch.Tensor], weights: torch.Tensor) -> None:
        for key, value in losses.items():
            metric = (value * weights).mean()
            self.log(
                f"{prefix}/{key}",
                metric,
                prog_bar=False,
                on_step=True,
                sync_dist=True,
            )


class MDTDataModule(pl.LightningDataModule):
    """Lightning DataModule for MDT image training."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        image_size: int,
        class_cond: bool,
        num_workers: int = 4,
        random_crop: bool = False,
        random_flip: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: Optional[ImageDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        all_files = _list_image_files_recursively(self.hparams.data_dir)
        classes = None
        if self.hparams.class_cond:
            class_names = [os.path.basename(file).split("_")[0] for file in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        self.dataset = ImageDataset(
            self.hparams.image_size,
            all_files,
            classes=classes,
            shard=0,
            num_shards=1,
            random_crop=self.hparams.random_crop,
            random_flip=self.hparams.random_flip,
        )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("DataModule.setup must be called before accessing the dataloader.")
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )

