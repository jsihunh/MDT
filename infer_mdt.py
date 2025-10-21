# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling script that relies on the Lightning MDT module."""

import argparse
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torchvision.utils import save_image

from masked_diffusion.lightning_module import MDTLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with a trained MDT Lightning checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a PyTorch Lightning checkpoint (.ckpt).")
    parser.add_argument("--output", type=Path, default=Path("sample.jpg"), help="File to save the generated grid of images.")
    parser.add_argument(
        "--class_labels",
        type=int,
        nargs="*",
        default=[19, 23, 106, 108, 278, 282],
        help="Optional class labels to condition the model. Leave empty for unconditional generation.",
    )
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--pow_scale",
        type=float,
        default=0.01,
        help="Power scheduling factor for classifier-free guidance interpolation.",
    )
    parser.add_argument("--num_sampling_steps", type=int, default=250, help="Number of diffusion sampling steps.")
    parser.add_argument("--ema_index", type=int, default=0, help="EMA index to use for sampling (-1 to disable EMA).")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for unconditional sampling.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for generation.")
    parser.add_argument("--nrow", type=int, default=3, help="Number of images per row in the output grid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_grad_enabled(False)

    module = MDTLightningModule.load_from_checkpoint(str(args.checkpoint), map_location=args.device)
    module.eval()
    module.to(args.device)

    class_labels: Optional[List[int]] = args.class_labels if args.class_labels else None
    batch_size = args.batch_size if args.batch_size > 0 else None
    ema_index = args.ema_index if args.ema_index >= 0 else None

    samples = module.sample(
        class_labels=class_labels,
        batch_size=batch_size,
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale if class_labels is not None else None,
        pow_scale=args.pow_scale,
        ema_index=ema_index,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, args.output, nrow=args.nrow, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
