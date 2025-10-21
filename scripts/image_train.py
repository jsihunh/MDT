"""Lightning-based entrypoint for training MDT diffusion models."""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from masked_diffusion import diffusion_defaults, model_and_diffusion_defaults
from masked_diffusion.lightning_module import MDTDataModule, MDTLightningModule
from masked_diffusion.script_util import add_dict_to_argparser, args_to_dict


def main() -> None:
    args = create_argparser().parse_args()

    pl.seed_everything(args.seed, workers=True)

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    module = MDTLightningModule(
        model_name=args.model,
        image_size=args.image_size,
        mask_ratio=args.mask_ratio,
        decode_layer=args.decode_layer,
        diffusion_kwargs=diffusion_kwargs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_rate=args.ema_rate,
        schedule_sampler=args.schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
        scale_factor=args.scale_factor,
        opt_type=args.opt_type,
        class_cond=args.class_cond,
        enable_mask=not args.disable_mask_training,
    )

    datamodule = MDTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        num_workers=args.num_workers,
        random_crop=args.random_crop,
        random_flip=not args.disable_flip,
    )

    callbacks = []
    checkpoint_kwargs = dict(
        dirpath=args.log_dir or None,
        save_last=True,
        filename="mdt-step{step:06d}",
    )
    if args.save_interval > 0:
        checkpoint_kwargs["every_n_train_steps"] = args.save_interval
    callbacks.append(ModelCheckpoint(**checkpoint_kwargs))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if args.precision == "bf16":
        precision = "bf16"
    elif args.precision == "16":
        precision = 16
    else:
        precision = 32

    trainer_kwargs = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=precision,
        default_root_dir=args.log_dir or None,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_interval,
        callbacks=callbacks,
    )
    if args.max_epochs > 0:
        trainer_kwargs["max_epochs"] = args.max_epochs
    if args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps
    elif args.lr_anneal_steps > 0:
        trainer_kwargs["max_steps"] = args.lr_anneal_steps

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume_from)


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=1000,
        model="MDTv2_S_2",
        mask_ratio=None,
        decode_layer=4,
        scale_factor=0.18215,
        opt_type="adan",
        num_workers=4,
        random_crop=False,
        disable_flip=False,
        disable_mask_training=False,
        grad_clip=0.0,
        accumulate_grad_batches=1,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=0,
        max_steps=0,
        resume_from=None,
        log_dir="lightning_logs",
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--save_interval", type=int, default=defaults["save_interval"], help="Steps between checkpoints.")
    parser.add_argument("--precision", choices=["32", "16", "bf16"], default="32", help="Numerical precision for training.")
    return parser


if __name__ == "__main__":
    main()
