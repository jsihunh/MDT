"""Generate samples from an MDT Lightning checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from masked_diffusion.lightning_module import MDTLightningModule


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate images with an MDT Lightning checkpoint")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the Lightning checkpoint")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to store samples")
    parser.add_argument("--output_prefix", type=str, default="mdt_samples", help="Filename prefix for saved arrays")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--class_cond", action="store_true", default=False)
    parser.add_argument("--cfg_cond", action="store_true", default=False)
    parser.add_argument("--cfg_scale", type=float, default=3.8)
    parser.add_argument("--scale_pow", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda)")
    return parser


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    module = MDTLightningModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.replace_diffusion(timestep_respacing=str(args.num_sampling_steps))
    module.to(device)
    module.eval()

    with torch.no_grad():
        samples = module.sample(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            class_cond=args.class_cond or args.cfg_cond,
            cfg_cond=args.cfg_cond,
            cfg_scale=args.cfg_scale,
            scale_pow=args.scale_pow,
            clip_denoised=args.clip_denoised,
            use_ddim=args.use_ddim,
        )

    arr = samples.cpu().numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = "x".join(str(dim) for dim in arr.shape)
    output_path = output_dir / f"{args.output_prefix}_{shape}.npz"
    np.savez(output_path, arr)
    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    main()
