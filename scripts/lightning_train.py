"""Train MDT with PyTorch Lightning."""

import argparse
from typing import Any, Dict

import pytorch_lightning as pl

from masked_diffusion import diffusion_defaults, model_and_diffusion_defaults
from masked_diffusion.lightning_module import MDTDataModule, MDTLightningModule
from masked_diffusion.script_util import add_dict_to_argparser, args_to_dict


def build_trainer(args: argparse.Namespace) -> pl.Trainer:
    trainer_kwargs: Dict[str, Any] = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.output_dir,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=True,
        precision=args.precision,
    )

    if args.strategy is not None:
        trainer_kwargs["strategy"] = args.strategy
    if args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        trainer_kwargs["max_epochs"] = args.max_epochs

    callbacks = []
    if args.checkpoint_every_n_steps > 0:
        from pytorch_lightning.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            every_n_train_steps=args.checkpoint_every_n_steps,
            filename="mdt-{step:07d}",
        )
        callbacks.append(checkpoint_callback)

    if args.monitor_lr:
        from pytorch_lightning.callbacks import LearningRateMonitor

        callbacks.append(LearningRateMonitor(logging_interval="step"))

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    return pl.Trainer(**trainer_kwargs)


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    diffusion_config = args_to_dict(args, diffusion_defaults().keys())

    module = MDTLightningModule(
        model_name=args.model,
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        decode_layer=args.decode_layer,
        diffusion_config=diffusion_config,
        class_cond=args.class_cond,
        schedule_sampler=args.schedule_sampler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        opt_type=args.opt_type,
        scale_factor=args.scale_factor,
        vae_model=args.vae_model,
    )

    datamodule = MDTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_crop=args.random_crop,
        random_flip=not args.disable_random_flip,
        num_workers=args.num_workers,
        deterministic=args.deterministic_loader,
    )

    trainer = build_trainer(args)
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint or None)


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        scale_factor=0.18215,
        vae_model="stabilityai/sd-vae-ft-mse",
        opt_type="adan",
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_crop", action="store_true")
    parser.add_argument("--disable_random_flip", action="store_true")
    parser.add_argument("--deterministic_loader", action="store_true")
    parser.add_argument("--opt_type", type=str, default="adan", choices=["adan", "adamw"])

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="lightning_logs")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=1000)
    parser.add_argument("--monitor_lr", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser


if __name__ == "__main__":
    main()

