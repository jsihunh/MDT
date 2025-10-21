"""Generate images from MDT Lightning checkpoints."""

import argparse
import os
from typing import Dict

import numpy as np
import torch

from masked_diffusion import diffusion_defaults, model_and_diffusion_defaults
from masked_diffusion.lightning_module import MDTLightningModule
from masked_diffusion.script_util import NUM_CLASSES, add_dict_to_argparser, args_to_dict


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    diffusion_args = args_to_dict(args, diffusion_defaults().keys())
    module = MDTLightningModule.from_pretrained(
        args.checkpoint_path,
        strict=False,
        map_location=device,
        model_name=args.model,
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        decode_layer=args.decode_layer,
        diffusion_config=diffusion_args,
        class_cond=args.class_cond,
    )
    module.to(device)
    module.eval()
    module.instantiate_first_stage()

    diffusion_config = module.hparams.get("diffusion_config", diffusion_args)
    diffusion_config = dict(diffusion_config)
    diffusion_config["timestep_respacing"] = str(args.num_sampling_steps)

    from masked_diffusion import create_diffusion

    diffusion = create_diffusion(**diffusion_config)

    torch.set_grad_enabled(False)

    os.makedirs(args.output_dir, exist_ok=True)

    all_images = []
    remaining = args.num_samples
    latent_size = module.hparams["image_size"] // 8

    while remaining > 0:
        current_batch = min(args.batch_size, remaining)
        samples = sample_batch(module, diffusion, current_batch, latent_size, device, args)
        images = module.decode_first_stage(samples)
        images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        all_images.append(images)
        remaining -= current_batch

    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    shape_str = "x".join(str(x) for x in arr.shape)
    output_path = os.path.join(args.output_dir, f"samples_{shape_str}.npz")
    np.savez(output_path, arr)
    print(f"Saved samples to {output_path}")


def sample_batch(module: MDTLightningModule, diffusion, batch_size: int, latent_size: int, device, args) -> torch.Tensor:
    z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    model_kwargs: Dict[str, torch.Tensor] = {}

    if module.class_cond and args.cfg_cond:
        classes = torch.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
        z = torch.cat([z, z], dim=0)
        classes_null = torch.full_like(classes, fill_value=NUM_CLASSES)
        classes_all = torch.cat([classes, classes_null], dim=0)
        model_kwargs["y"] = classes_all
        model_kwargs["cfg_scale"] = args.cfg_scale
        model_kwargs["scale_pow"] = args.scale_pow
        model_kwargs["diffusion_steps"] = diffusion.num_timesteps
    elif module.class_cond:
        classes = torch.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
        model_kwargs["y"] = classes

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    samples = sample_fn(
        module.model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=args.clip_denoised,
        progress=True,
        model_kwargs=model_kwargs,
        device=device,
    )

    if module.class_cond and args.cfg_cond:
        samples, _ = samples.chunk(2, dim=0)

    return samples


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        num_sampling_steps=250,
        clip_denoised=False,
        num_samples=64,
        batch_size=16,
        use_ddim=False,
        cfg_cond=True,
        cfg_scale=3.8,
        scale_pow=4.0,
        checkpoint_path="",
        output_dir="samples",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--use_cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main()

