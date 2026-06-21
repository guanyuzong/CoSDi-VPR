"""
Train BoQ + PSD on University-1652 cross-view geo-localization.

Mirrors train.py. Differences from VPR training:
  - U1652DataModule (drone+satellite per-building sampling, rotation aug)
  - use_pos_embed=False (rotation invalidates absolute spatial priors)
  - val on drone->satellite + satellite->drone via R@k (mAP via eval_u1652.py)
  - Optionally init from a GSV-Cities pretrained checkpoint
"""

import argparse
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.datamodule_u1652 import U1652DataModule


class HyperParams:
    def __init__(self):
        # Backbone / aggregator (kept compatible with the GSV-Cities ckpt)
        self.backbone_name: str = "dinov2_vitb14"
        self.unfreeze_n_blocks: int = 3
        self.channel_proj: int = 512
        self.num_queries: int = 20
        self.num_layers: int = 1
        self.output_dim: int = 6144
        self.use_pos_embed: bool = False     # rotation augmentation -> no abs pos prior

        # Data paths (EDIT THESE)
        self.u1652_train: str = "data/University-Release/train"
        self.u1652_test:  str = "data/University-Release/test"

        # Optimization
        self.batch_size: int = 32             # P=32 buildings, K=4 imgs -> 128 effective
        self.img_per_place: int = 4
        self.sat_ratio: float = 0.25          # 1 satellite + 3 drone per building
        self.sample_num: int = 4              # MCCG-style: each building visited sample_num times/epoch
        self.max_epochs: int = 80
        self.warmup_epochs: int = 3
        self.lr: float = 2e-4
        self.weight_decay: float = 5e-4
        self.lr_mul: float = 0.1
        self.milestones: list = [40, 60, 70]
        self.num_workers: int = 8

        # Image sizes (DINOv2 patch=14: 224 = 16x16 tokens, 252 = 18x18)
        self.train_img_size = (224, 224)
        self.val_img_size   = (252, 252)

        # Optionally initialize from a VPR-pretrained checkpoint
        self.pretrained_ckpt: str | None = \
            "epoch[43]_R@1[0.9419]_R@5[0.9662].ckpt"

        self.silent: bool = False
        self.compile: bool = False
        self.seed: int = 1997


def _load_pretrained(model: BoQModel, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded from {ckpt_path}")
    print(f"       missing keys ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
    print(f"       unexpected   ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")


def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)

    backbone = DinoV2(backbone_name=hparams.backbone_name,
                      unfreeze_n_blocks=hparams.unfreeze_n_blocks)
    hparams.backbone_name = backbone.backbone_name

    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim // hparams.channel_proj,
        use_pos_embed=hparams.use_pos_embed,
    )

    model = BoQModel(
        backbone, aggregator,
        lr=hparams.lr, lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )

    if hparams.pretrained_ckpt:
        _load_pretrained(model, hparams.pretrained_ckpt)

    if hparams.compile:
        model = torch.compile(model)

    datamodule = U1652DataModule(
        train_path=hparams.u1652_train,
        test_path=hparams.u1652_test,
        img_per_place=hparams.img_per_place,
        sat_ratio=hparams.sat_ratio,
        sample_num=hparams.sample_num,
        train_img_size=hparams.train_img_size,
        val_img_size=hparams.val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        directions=("d2s", "s2d"),
    )

    if not hparams.silent:
        datamodule.setup()
        # display_datasets_stats expects .num_references/.num_queries on val sets
        display_datasets_stats(datamodule)

    tb_logger = TensorBoardLogger(save_dir="./logs",
                                  name=f"{hparams.backbone_name}_u1652",
                                  default_hp_metric=False)
    tb_logger.log_hyperparams(hparams.__dict__)

    ckpt_cb = callbacks.ModelCheckpoint(
        monitor="u1652-d2s/R@1",
        filename="epoch[{epoch:02d}]_d2sR@1[{u1652-d2s/R@1:.4f}]_s2dR@1[{u1652-s2d/R@1:.4f}]",
        auto_insert_metric_name=False,
        save_top_k=5, mode="max",
    )
    cbs = [ckpt_cb]
    if not hparams.silent:
        cbs.append(callbacks.RichProgressBar())

    trainer = Trainer(
        accelerator="gpu", devices=[0],
        logger=tb_logger, precision="16-mixed",
        callbacks=cbs,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=2,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=model, datamodule=datamodule)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    p.add_argument("--silent", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--bs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--ckpt", type=str, help="path to GSV-Cities pretrained ckpt")
    p.add_argument("--train_path", type=str)
    p.add_argument("--test_path", type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    h = HyperParams()
    if args.seed:        h.seed = args.seed
    if args.silent:      h.silent = True
    if args.bs:          h.batch_size = args.bs
    if args.lr:          h.lr = args.lr
    if args.epochs:      h.max_epochs = args.epochs
    if args.ckpt:        h.pretrained_ckpt = args.ckpt
    if args.train_path:  h.u1652_train = args.train_path
    if args.test_path:   h.u1652_test = args.test_path
    train(h, dev_mode=args.dev)
