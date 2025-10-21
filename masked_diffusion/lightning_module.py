"""PyTorch Lightning module for training and sampling MDT models."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from diffusers.models import AutoencoderKL
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW

from adan import Adan

from . import create_diffusion, diffusion_defaults
from .resample import create_named_schedule_sampler
from .train_util import log_loss_dict
import masked_diffusion.models as models_mdt


class MDTLightningModule(LightningModule):
    """Lightning module that encapsulates MDT training and sampling."""

    def __init__(
        self,
        *,
        model_name: str = "MDTv2_S_2",
        model_config: Optional[Dict] = None,
        diffusion_config: Optional[Dict] = None,
        lr: float = 5e-4,
        weight_decay: float = 0.0,
        opt_type: str = "adan",
        schedule_sampler: str = "uniform",
        lr_anneal_steps: int = 0,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        scale_factor: float = 0.18215,
        gradient_clip_val: Optional[float] = None,
    ) -> None:
        super().__init__()
        model_config = model_config or {}
        diffusion_config = diffusion_config or diffusion_defaults()

        image_size: int = int(model_config.get("image_size", 256))
        mask_ratio = model_config.get("mask_ratio")
        decode_layer = model_config.get("decode_layer", 4)
        self.class_cond = bool(model_config.get("class_cond", True))
        self.use_fp16 = bool(model_config.get("use_fp16", False))

        latent_size = image_size // 8
        if latent_size <= 0:
            raise ValueError("Image size must be positive to build MDT model.")

        if model_name not in models_mdt.__dict__:
            raise ValueError(f"Unknown MDT model '{model_name}'.")
        self.model = models_mdt.__dict__[model_name](
            input_size=latent_size, mask_ratio=mask_ratio, decode_layer=decode_layer
        )

        self.diffusion_config = diffusion_config
        self.diffusion = create_diffusion(**diffusion_config)
        self.schedule_sampler_name = schedule_sampler
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler, self.diffusion
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.opt_type = opt_type
        self.lr_anneal_steps = lr_anneal_steps
        self.vae_model_name = vae_model
        self.scale_factor = scale_factor
        self.gradient_clip_val = gradient_clip_val
        self.latent_size = latent_size

        self.first_stage_model: Optional[AutoencoderKL] = None

        self.save_hyperparameters(
            {
                "model_name": model_name,
                "model_config": model_config,
                "diffusion_config": diffusion_config,
                "lr": lr,
                "weight_decay": weight_decay,
                "opt_type": opt_type,
                "schedule_sampler": schedule_sampler,
                "lr_anneal_steps": lr_anneal_steps,
                "vae_model": vae_model,
                "scale_factor": scale_factor,
                "gradient_clip_val": gradient_clip_val,
            }
        )

    def instantiate_first_stage(self) -> None:
        """Lazily construct the first stage autoencoder used for encoding/decoding."""

        if self.first_stage_model is not None:
            return
        device = self.device if self.device.type != "meta" else torch.device("cpu")
        vae = AutoencoderKL.from_pretrained(self.vae_model_name)
        if self.use_fp16:
            vae = vae.to(device=device, dtype=torch.float16)
        else:
            vae = vae.to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        self.first_stage_model = vae

    @torch.no_grad()
    def encode_first_stage(self, images: Tensor) -> Tensor:
        """Encode input images into the latent space used by MDT."""

        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        posterior = self.first_stage_model.encode(images, return_dict=True)[0]
        latents = posterior.sample()
        return latents.to(self.device) * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, latents: Tensor) -> Tensor:
        """Decode latent representations back to image space."""

        self.instantiate_first_stage()
        assert self.first_stage_model is not None
        latents = latents / self.scale_factor
        decoded = self.first_stage_model.decode(latents).sample
        return decoded

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        images, cond = batch
        if cond is None:
            cond = {}
        if self.class_cond and "y" not in cond:
            raise ValueError(
                "Class-conditioned training requires the dataset to return labels."
            )
        cond_tensors: Dict[str, Tensor] = {
            key: value.to(self.device) if isinstance(value, Tensor) else torch.as_tensor(value, device=self.device)
            for key, value in cond.items()
        }
        batch_size = images.shape[0]
        images = images.to(self.device)
        latents = self.encode_first_stage(images)

        timesteps, weights = self.schedule_sampler.sample(batch_size, self.device)
        weights = weights.to(latents.dtype)

        losses = self.diffusion.training_losses(
            self.model, latents, timesteps, model_kwargs=cond_tensors
        )
        cond_with_mask = dict(cond_tensors)
        cond_with_mask["enable_mask"] = True
        masked_losses = self.diffusion.training_losses(
            self.model, latents, timesteps, model_kwargs=cond_with_mask
        )

        base_loss = (losses["loss"] * weights).mean()
        mask_loss = (masked_losses["loss"] * weights).mean()
        total_loss = base_loss + mask_loss

        self.log("loss/train", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("loss/base", base_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("loss/mask", mask_loss, on_step=True, on_epoch=True, batch_size=batch_size)

        log_loss_dict(self.diffusion, timesteps, losses)
        log_loss_dict(self.diffusion, timesteps, {f"m_{k}": v for k, v in masked_losses.items()})

        return total_loss

    def configure_optimizers(self):  # type: ignore[override]
        if self.opt_type.lower() == "adan":
            optimizer = Adan(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                max_grad_norm=1,
                fused=True,
            )
        elif self.opt_type.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.opt_type}")

        if self.lr_anneal_steps > 0:
            def lr_lambda(step: int) -> float:
                if step >= self.lr_anneal_steps:
                    return 0.0
                return 1.0 - (step / float(self.lr_anneal_steps))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer

    @torch.no_grad()
    def sample_images(
        self,
        class_labels: Iterable[int],
        *,
        num_steps: Optional[int] = None,
        cfg_scale: float = 4.0,
        pow_scale: float = 0.01,
        latent_noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate images given class labels using classifier-free guidance."""

        self.eval()
        device = self.device
        class_labels = list(class_labels)
        if not class_labels:
            raise ValueError("At least one class label must be provided for sampling.")
        n = len(class_labels)

        self.instantiate_first_stage()

        if latent_noise is None:
            latent_noise = torch.randn(
                n, 4, self.latent_size, self.latent_size, device=device
            )
        else:
            latent_noise = latent_noise.to(device)

        y = torch.tensor(class_labels, device=device, dtype=torch.long)
        z = torch.cat([latent_noise, latent_noise], dim=0)
        y_null = torch.tensor([1000] * n, device=device, dtype=torch.long)
        y = torch.cat([y, y_null], dim=0)

        diffusion = self.diffusion
        if num_steps is not None:
            diffusion_config = dict(self.diffusion_config)
            diffusion_config["timestep_respacing"] = str(num_steps)
            diffusion = create_diffusion(**diffusion_config)

        model_kwargs: Dict[str, object] = {
            "y": y,
            "cfg_scale": cfg_scale,
            "scale_pow": pow_scale,
        }

        samples = diffusion.p_sample_loop(
            self.model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)
        images = self.decode_first_stage(samples)
        return images

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):  # type: ignore[override]
        optimizer.zero_grad(set_to_none=True)

