"""Train MDT with a PyTorch Lightning workflow."""
from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from masked_diffusion import diffusion_defaults, model_and_diffusion_defaults, logger
from masked_diffusion.lightning_data import MDTDataModule
from masked_diffusion.lightning_module import MDTLightningModule
from masked_diffusion.script_util import add_dict_to_argparser, args_to_dict


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        data_dir="",
        log_dir="lightning_logs",
        checkpoint_dir="checkpoints",
        save_every_n_steps=1000,
        max_epochs=None,
        max_steps=None,
        precision=32,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        accelerator="auto",
        devices="auto",
        num_nodes=1,
        strategy="auto",
        seed=42,
        num_workers=4,
        random_crop=False,
        random_flip=True,
        deterministic=False,
        optimizer="adan",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        scale_factor=0.18215,
        vae_variant="mse",
        compile_model=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(description="Lightning trainer for MDT")
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--model", type=str, default="MDTv2_S_2", help="Model architecture key")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=-1)
    parser.add_argument("--mask_ratio", type=float, default=None)
    parser.add_argument("--decode_layer", type=int, default=4)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    return parser


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()
    if not args.data_dir:
        raise ValueError("--data_dir must be provided")

    pl.seed_everything(args.seed, workers=True)

    logger.configure(dir=args.log_dir)

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    diffusion_kwargs["timestep_respacing"] = args.timestep_respacing

    latent_size = args.image_size // 8
    model_kwargs = {
        "input_size": latent_size,
        "mask_ratio": args.mask_ratio,
        "decode_layer": args.decode_layer,
        "class_dropout_prob": args.class_dropout_prob,
        "learn_sigma": args.learn_sigma,
    }
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    module = MDTLightningModule(
        model_name=args.model,
        model_kwargs=model_kwargs,
        diffusion_kwargs=diffusion_kwargs,
        schedule_sampler=args.schedule_sampler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scale_factor=args.scale_factor,
        vae_variant=args.vae_variant,
        compile_model=args.compile_model,
    )

    data_module = MDTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        num_workers=args.num_workers,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
        deterministic=args.deterministic,
    )

    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="mdt-{step:06d}",
        save_last=True,
        every_n_train_steps=args.save_every_n_steps,
        save_top_k=-1,
    ))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="mdt")

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=tb_logger,
        limit_train_batches=args.limit_train_batches,
    )

    ckpt_path = args.resume_from if args.resume_from else None
    trainer.fit(module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
