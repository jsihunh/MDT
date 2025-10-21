"""PyTorch Lightning module that wraps MDT training and sampling utilities."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW

from diffusers.models import AutoencoderKL

from . import create_diffusion
from .models import __dict__ as model_zoo
from .resample import LossAwareSampler, create_named_schedule_sampler
from .train_util import log_loss_dict
from .script_util import NUM_CLASSES

try:  # Adan is optional during generation, so guard the import.
    from adan import Adan
except Exception:  # pragma: no cover - Adan is only needed for training.
    Adan = None  # type: ignore


class MDTLightningModule(pl.LightningModule):
    """Lightning module that encapsulates MDT training and sampling."""

    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
        schedule_sampler: str = "uniform",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        optimizer: str = "adan",
        scale_factor: float = 0.18215,
        vae_variant: str = "mse",
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = model_kwargs or {}
        diffusion_kwargs = diffusion_kwargs or {}

        if model_name not in model_zoo:
            raise ValueError(f"Unknown MDT architecture: {model_name}")

        self.model: nn.Module = model_zoo[model_name](**model_kwargs)
        if compile_model:
            self.model = torch.compile(self.model)

        self._base_diffusion_kwargs = diffusion_kwargs
        self.diffusion = create_diffusion(**diffusion_kwargs)
        self.schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer.lower()
        self.scale_factor = scale_factor
        self.vae_variant = vae_variant

        self.first_stage_model: Optional[AutoencoderKL] = None

    def instantiate_first_stage(self) -> None:
        if self.first_stage_model is not None:
            return
        vae_id = f"stabilityai/sd-vae-ft-{self.vae_variant}"
        vae = AutoencoderKL.from_pretrained(vae_id)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        self.first_stage_model = vae.to(self.device, dtype=self.model.dtype)

    @torch.no_grad()
    def get_first_stage_encoding(self, inputs: torch.Tensor) -> torch.Tensor:
        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        posterior = self.first_stage_model.encode(inputs, return_dict=True)[0]
        latents = posterior.sample()
        return latents * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, latents: torch.Tensor) -> torch.Tensor:
        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        scaled = latents / self.scale_factor
        decoded = self.first_stage_model.decode(scaled, return_dict=True)["sample"]
        return decoded

    def replace_diffusion(self, **override: Any) -> None:
        new_args = deepcopy(self._base_diffusion_kwargs)
        new_args.update(override)
        self.diffusion = create_diffusion(**new_args)
        self.schedule_sampler = create_named_schedule_sampler(
            self.hparams.schedule_sampler, self.diffusion
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        return self.model(x, t, **model_kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images, cond = batch
        images = images.to(self.device, dtype=self.model.dtype)
        cond = {k: v.to(self.device) for k, v in cond.items()}
        if "y" not in cond:
            cond["y"] = torch.zeros(images.shape[0], dtype=torch.long, device=self.device)

        latents = self.get_first_stage_encoding(images)
        t, weights = self.schedule_sampler.sample(images.shape[0], device=self.device)

        loss_terms = self.diffusion.training_losses(
            self.model,
            latents,
            t,
            model_kwargs=cond,
        )
        cond_mask = dict(cond)
        cond_mask["enable_mask"] = True
        masked_terms = self.diffusion.training_losses(
            self.model,
            latents,
            t,
            model_kwargs=cond_mask,
        )

        weighted = loss_terms["loss"] * weights
        weighted_mask = masked_terms["loss"] * weights
        loss = weighted.mean() + weighted_mask.mean()

        self.log("train/loss", weighted.mean(), prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "train/masked_loss",
            weighted_mask.mean(),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            combined = loss_terms["loss"].detach() + masked_terms["loss"].detach()
            self.schedule_sampler.update_with_local_losses(t, combined)

        log_loss_dict(self.diffusion, t, loss_terms)
        log_loss_dict(self.diffusion, t, {f"m_{k}": v for k, v in masked_terms.items()})

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = self.model.parameters()
        if self.optimizer_name == "adan":
            if Adan is None:
                raise RuntimeError(
                    "The Adan optimizer is not available. Install the adan package or set optimizer='adamw'."
                )
            optimizer = Adan(params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1, fused=True)
        elif self.optimizer_name == "adamw":
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer

    @torch.no_grad()
    def sample(
        self,
        *,
        num_samples: int,
        batch_size: int,
        class_cond: bool = True,
        cfg_cond: bool = False,
        cfg_scale: float = 3.8,
        scale_pow: float = 4.0,
        clip_denoised: bool = False,
        use_ddim: bool = False,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.instantiate_first_stage()
        self.model.eval()

        latent_size = self.hparams.model_kwargs.get("input_size")
        if latent_size is None:
            raise ValueError("'input_size' must be present in model_kwargs to run sampling.")

        results = []
        total = 0
        device = self.device
        while total < num_samples:
            current_bs = min(batch_size, num_samples - total)
            latent_shape = (current_bs, 4, latent_size, latent_size)
            if z is None:
                latents = torch.randn(latent_shape, device=device, dtype=self.model.dtype)
            else:
                latents = z.to(device)

            model_kwargs: Dict[str, Any] = {}
            if class_cond:
                classes = torch.randint(0, NUM_CLASSES, (current_bs,), device=device)
                if cfg_cond:
                    latents = torch.cat([latents, latents], dim=0)
                    null_classes = torch.full((current_bs,), NUM_CLASSES, device=device, dtype=classes.dtype)
                    model_kwargs["y"] = torch.cat([classes, null_classes], dim=0)
                    model_kwargs["cfg_scale"] = cfg_scale
                    model_kwargs["diffusion_steps"] = self.diffusion.num_timesteps
                    model_kwargs["scale_pow"] = scale_pow
                else:
                    model_kwargs["y"] = classes
            else:
                zeros = torch.zeros(current_bs, dtype=torch.long, device=device)
                model_kwargs["y"] = zeros

            sample_fn = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop
            samples = sample_fn(
                self.model.forward_with_cfg if cfg_cond else self.model,
                latents.shape,
                latents,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                device=device,
            )
            if cfg_cond:
                samples, _ = samples.chunk(2, dim=0)
            decoded = self.decode_first_stage(samples)
            decoded = torch.clamp((decoded + 1.0) * 127.5, 0, 255).to(torch.uint8)
            decoded = decoded.permute(0, 2, 3, 1).contiguous()
            results.append(decoded.cpu())
            total += current_bs
        return torch.cat(results, dim=0)[:num_samples]
