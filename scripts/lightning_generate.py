"""Sample images from a trained MDT Lightning checkpoint."""

from __future__ import annotations

import argparse
import math
from typing import List

import torch
from pytorch_lightning import seed_everything
from torchvision.utils import save_image

from masked_diffusion.lightning_module import MDTLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with a Lightning MDT checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--class_labels", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="samples.png")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--pow_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nrow", type=int, default=None, help="Number of images per row in the saved grid")
    return parser.parse_args()


def determine_nrow(labels: List[int], user_nrow: int | None) -> int:
    if user_nrow is not None:
        return user_nrow
    count = len(labels)
    return int(math.ceil(math.sqrt(count)))


def main() -> None:
    args = parse_args()

    seed_everything(args.seed, workers=True)

    module = MDTLightningModule.load_from_checkpoint(args.checkpoint, map_location=args.device)
    module.to(args.device)
    module.freeze()

    images = module.sample_images(
        args.class_labels,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        pow_scale=args.pow_scale,
    )

    nrow = determine_nrow(args.class_labels, args.nrow)
    save_image(images, args.output, nrow=nrow, normalize=True, value_range=(-1, 1))

    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()

