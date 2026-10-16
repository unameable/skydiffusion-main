"""Train single-panorama SkyDiffusion; run --help for local checkpoint options."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/single_image.yaml"))
    for name in ("train-data", "val-data", "data-root", "clip-model", "output-dir", "prompt"):
        parser.add_argument("--" + name)
    parser.add_argument("--source-mode", choices=("panorama", "bev"))
    parser.add_argument("--dataset-name", choices=("CVACT", "CVUSA"))
    parser.add_argument("--is-cvusa", action="store_true", default=None,
                        help="Use legacy headerless CVUSA CSV ordering")
    parser.add_argument("--rotation-degrees", type=float)
    for name in ("image-size", "batch-size", "num-workers", "accumulate-grad-batches",
                 "max-steps", "devices", "checkpoint-every", "image-log-every", "ddim-steps"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--precision", type=int, choices=(16, 32))
    parser.add_argument("--accelerator", choices=("cpu", "gpu"))
    parser.add_argument("--train-unet-decoder", action="store_true",
                        help="Also optimize the SD decoder and output layer")
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--sd-checkpoint", help="Original LDM-format SD 1.5 .ckpt/.safetensors")
    weights.add_argument("--init-checkpoint", help="Complete SkyDiffusion weights for a new run")
    weights.add_argument("--resume", help="Lightning checkpoint, including optimizer and global step")
    parser.add_argument("--check-data", action="store_true",
                        help="Save input/GT previews and range report; no model or weights required")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run two optimizer steps, one validation batch, and save last.ckpt")
    return parser


def load_settings(args):
    from omegaconf import OmegaConf
    settings = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    for key in list(settings):
        override = getattr(args, key, None)
        if override is not None:
            settings[key] = override
    if args.train_unet_decoder:
        settings["sd_locked"] = False
    if not settings.get("train_data"):
        raise ValueError("Provide --train-data with a paired image CSV/TSV manifest")
    for name in ("batch_size", "devices", "accumulate_grad_batches", "max_steps",
                 "checkpoint_every", "val_every_epochs"):
        if settings[name] < 1:
            raise ValueError(name + " must be at least 1")
    if settings["num_workers"] < 0 or settings["image_log_every"] < 0:
        raise ValueError("num_workers and image_log_every must be nonnegative")
    if settings["image_size"] < 64 or settings["image_size"] % 64:
        raise ValueError("image_size must be a positive multiple of 64")
    if not 1 <= settings["ddim_steps"] < 1000 or 1000 % settings["ddim_steps"]:
        raise ValueError("ddim_steps must be below 1000 and divide 1000 (for example 20 or 50)")
    if settings["learning_rate"] <= 0 or not 0 <= settings["drop_prompt_ratio"] <= 1:
        raise ValueError("Invalid learning_rate or drop_prompt_ratio")
    if args.smoke_test:
        settings.update(max_steps=2, accumulate_grad_batches=1, image_log_every=0)
    return settings


def make_dataset(settings, validation=False):
    from skydiffusion_dataset import SkyDataset
    return SkyDataset(
        settings["val_data"] if validation else settings["train_data"],
        settings["is_cvusa"], settings["prompt"],
        image_size=(settings["image_size"], settings["image_size"]),
        drop_prompt_ratio=0 if validation else settings["drop_prompt_ratio"],
        dataset_name=settings["dataset_name"], source_mode=settings["source_mode"],
        data_root=settings["data_root"], rotation_degrees=settings["rotation_degrees"],
    )


def check_data(dataset, output_dir, split):
    import numpy as np
    from PIL import Image
    preview_dir = Path(output_dir) / "data_preview" / split
    preview_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for index in range(min(4, len(dataset))):
        sample = dataset[index]
        hint, target = sample["hint"], sample["jpg"]
        if not (np.isfinite(hint).all() and 0 <= hint.min() <= hint.max() <= 1):
            raise ValueError("hint must contain finite RGB values in [0, 1]")
        if not (np.isfinite(target).all() and -1 <= target.min() <= target.max() <= 1):
            raise ValueError("jpg must contain finite RGB values in [-1, 1]")
        pair = np.concatenate([hint, (target + 1) / 2], axis=1)
        Image.fromarray(np.rint(pair * 255).astype(np.uint8)).save(
            str(preview_dir / "{:03d}_bev_gt.png".format(index)))
        report.append({"name": sample["name"], "hint_shape": list(hint.shape),
                       "hint_range": [float(hint.min()), float(hint.max())],
                       "jpg_shape": list(target.shape),
                       "jpg_range": [float(target.min()), float(target.max())]})
    (preview_dir / "ranges.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("{}: {} pairs; checked {} previews in {}".format(
        split, len(dataset), len(report), preview_dir))


def make_preview_callback(settings):
    import torch
    import numpy as np
    from PIL import Image
    from pytorch_lightning.callbacks import Callback

    class TrainingPreview(Callback):
        def __init__(self):
            self.last_step = -1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if (not trainer.is_global_zero or step == 0 or step == self.last_step or
                    step % settings["image_log_every"]):
                return
            self.last_step = step
            was_training = pl_module.training
            try:
                pl_module.eval()
                with torch.no_grad():
                    images = pl_module.log_images(
                        batch, N=1, ddim_steps=settings["ddim_steps"],
                        unconditional_guidance_scale=1.0, negative_prompt="")
                folder = Path(settings["output_dir"]) / "image_log"
                folder.mkdir(parents=True, exist_ok=True)
                for key, value in images.items():
                    value = value[0].detach().float().cpu().clamp(-1, 1)
                    value = ((value + 1) / 2).permute(1, 2, 0).numpy()
                    Image.fromarray(np.rint(value * 255).astype(np.uint8)).save(
                        str(folder / "{:07d}_{}.png".format(step, key)))
            finally:
                pl_module.train(was_training)

    return TrainingPreview()


def main():
    parser = create_argparser()
    args = parser.parse_args()
    if args.resume and args.smoke_test:
        parser.error("--smoke-test starts a new two-step run. For --resume, set --max-steps above the saved step.")
    settings = load_settings(args)
    if not args.check_data and not (args.sd_checkpoint or args.init_checkpoint or args.resume):
        parser.error("Training requires --sd-checkpoint, --init-checkpoint, or --resume")
    if not args.check_data:
        checkpoint_path = args.sd_checkpoint or args.init_checkpoint or args.resume
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError("Checkpoint does not exist: " + checkpoint_path)

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    if not pl.__version__.startswith("1.5."):
        raise RuntimeError("Use the repository's pytorch-lightning==1.5.0 environment; found "
                           + pl.__version__)
    pl.seed_everything(settings["seed"], workers=True)
    train_dataset = make_dataset(settings)
    val_dataset = make_dataset(settings, validation=True) if settings["val_data"] else None
    check_data(train_dataset, settings["output_dir"], "train")
    if val_dataset is not None:
        check_data(val_dataset, settings["output_dir"], "val")
    if args.check_data:
        return

    if settings["accelerator"] == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the CUDA training environment or --accelerator cpu")
    if settings["accelerator"] == "cpu" and settings["precision"] == 16:
        raise ValueError("CPU training requires --precision 32")
    import os
    if os.name == "nt" and settings["devices"] > 1:
        raise ValueError("Use one GPU on Windows; use Linux for multi-GPU DDP")

    from ldm.util import instantiate_from_config
    from skydiffusion_training import (load_weights, initialize_from_sd,
                                      initialize_from_skydiffusion,
                                      configure_trainable_parameters,
                                      make_completed_checkpoint_callback)
    model_config_path = Path(settings["model_config"])
    if not model_config_path.is_absolute():
        model_config_path = PROJECT_ROOT / model_config_path
    model_config = OmegaConf.load(str(model_config_path))
    if model_config.model.target != "cldm.cldm.LAControlLDM":
        raise ValueError("Single-image training expects cldm.cldm.LAControlLDM")
    model_config.model.params.cond_stage_config.params.version = settings["clip_model"]
    model_config.model.params.cond_stage_config.params.device = (
        "cuda" if settings["accelerator"] == "gpu" else "cpu")
    model = instantiate_from_config(model_config.model).cpu()

    checkpoint_path = args.sd_checkpoint or args.init_checkpoint or args.resume
    weights = load_weights(checkpoint_path)
    report = (initialize_from_sd(model, weights) if args.sd_checkpoint else
              initialize_from_skydiffusion(model, weights))
    del weights
    report.update(configure_trainable_parameters(
        model, settings["learning_rate"], settings["sd_locked"]))
    # PL 1.5's save_hyperparameters inspects __init__; this runs after construction.
    model.hparams["single_image_training"] = settings

    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(settings), str(output_dir / "training_config.yaml"))
    OmegaConf.save(model_config, str(output_dir / "model_config.yaml"))
    (output_dir / "initialization.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    loader_options = dict(batch_size=settings["batch_size"],
                          num_workers=settings["num_workers"],
                          pin_memory=settings["accelerator"] == "gpu")
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options) if val_dataset else None
    callbacks = [make_completed_checkpoint_callback(), ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"), filename="step-{step:07d}",
        auto_insert_metric_name=False, every_n_train_steps=settings["checkpoint_every"],
        save_top_k=-1, save_last=True)]
    if settings["image_log_every"]:
        callbacks.append(make_preview_callback(settings))
    trainer_options = dict(
        gpus=settings["devices"] if settings["accelerator"] == "gpu" else 0,
        precision=settings["precision"], max_steps=settings["max_steps"], max_epochs=-1,
        accumulate_grad_batches=settings["accumulate_grad_batches"],
        gradient_clip_val=settings["gradient_clip_val"], callbacks=callbacks,
        logger=CSVLogger(str(output_dir), name="metrics"),
        default_root_dir=str(output_dir), num_sanity_val_steps=0,
        check_val_every_n_epoch=settings["val_every_epochs"],
        limit_val_batches=1 if args.smoke_test else 1.0,
        log_every_n_steps=1 if args.smoke_test else 10)
    if args.smoke_test and val_loader is not None:
        trainer_options["val_check_interval"] = 1
    if settings["accelerator"] == "gpu" and settings["devices"] > 1:
        trainer_options["strategy"] = "ddp"
    trainer = pl.Trainer(**trainer_options)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume)
    trainer.save_checkpoint(str(output_dir / "checkpoints" / "last.ckpt"))


if __name__ == "__main__":
    main()
