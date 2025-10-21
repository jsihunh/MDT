"""PyTorch Lightning training entry-point for MDT."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from masked_diffusion import diffusion_defaults, model_and_diffusion_defaults
from masked_diffusion.data_module import ImageDataModule
from masked_diffusion.lightning_module import MDTLightningModule
from masked_diffusion.script_util import add_dict_to_argparser, args_to_dict


def build_trainer(args: argparse.Namespace) -> pl.Trainer:
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_train_steps=args.checkpoint_every_n_steps,
        dirpath=args.default_root_dir,
        filename="mdt-{step:08d}",
        save_top_k=1,
        monitor="loss/train",
        mode="min",
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    precision = 16 if args.use_fp16 else 32

    devices = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=precision,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_interval,
        default_root_dir=args.default_root_dir,
        callbacks=callbacks,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    return trainer


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MDT with PyTorch Lightning")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--no_random_flip", action="store_true")
    parser.add_argument("--max_steps", type=int, default=250000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000)
    parser.add_argument("--default_root_dir", type=str, default="lightning_logs")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--opt_type", type=str, default="adan", choices=["adan", "adamw"])
    parser.add_argument("--schedule_sampler", type=str, default="uniform")
    parser.add_argument("--lr_anneal_steps", type=int, default=0)
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--scale_factor", type=float, default=0.18215)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    add_dict_to_argparser(parser, model_and_diffusion_defaults())
    return parser


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    model_config = args_to_dict(args, model_and_diffusion_defaults().keys())
    diffusion_config = args_to_dict(args, diffusion_defaults().keys())

    module = MDTLightningModule(
        model_name=args.model,
        model_config=model_config,
        diffusion_config=diffusion_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt_type=args.opt_type,
        schedule_sampler=args.schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
        vae_model=args.vae_model,
        scale_factor=args.scale_factor,
        gradient_clip_val=args.gradient_clip_val,
    )

    datamodule = ImageDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=model_config["image_size"],
        class_cond=model_config["class_cond"],
        num_workers=args.num_workers,
        random_crop=args.random_crop,
        random_flip=not args.no_random_flip,
    )

    trainer = build_trainer(args)
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()

