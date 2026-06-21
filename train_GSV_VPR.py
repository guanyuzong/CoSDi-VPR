import argparse
import torch
from lightning.pytorch import callbacks
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from src.utils import display_datasets_stats
from src.backbones import DinoV2, ResNet
from src.cosdi import CoSDi
from src.model import CoSDiModel
from src.dataloaders.datamodule import VPRDataModule

class HyperParams:
    def __init__(self):
        self.backbone_name: str = "dinov2_vitb14"
        self.unfreeze_n_blocks: int = 3

        self.channel_proj: int = 512
        self.num_queries: int = 20                    # 不变
        self.num_layers: int = 1
        self.output_dim: int = 6144                   # 24*256

        self.gsv_cities_path: str = "/home/ZONG/data/GSVCities"
        self.cities: str | list = "all"
        self.val_sets: dict = {
            "pitts30k-val": "/home/ZONG/data/pitts30k-val",
            "msls-val":     "//home/ZONG/data/msls-val",
        }

        self.batch_size: int = 200
        self.img_per_place: int = 4
        self.max_epochs: int = 70                     
        self.warmup_epochs: int = 5                   
        self.lr: float = 0.0004
        self.weight_decay: float = 0.0005
        self.lr_mul: float = 0.3
        self.milestones: list = [20, 40, 60]          
        self.num_workers: int = 8

        self.silent: bool = False
        self.compile: bool = False
        self.seed: int = 1997


def train(hparams, dev_mode=False):
    seed_everything(hparams.seed, workers=True)

    print(hparams.backbone_name)
    if "dinov2" in hparams.backbone_name:
        backbone = DinoV2(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks)
        train_img_size = (280, 280)
        val_img_size = (322, 322)
        hparams.backbone_name = backbone.backbone_name
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
    elif "resnet" in hparams.backbone_name:
        backbone = ResNet(backbone_name=hparams.backbone_name, unfreeze_n_blocks=hparams.unfreeze_n_blocks, crop_last_block=True)
        train_img_size = (320, 320)
        val_img_size = (384, 384)
        hparams.train_img_size = train_img_size
        hparams.val_img_size = val_img_size
    else:
        raise ValueError(f"backbone {hparams.backbone_name} not recognized!")

    aggregator = CoSDi(
        in_channels=backbone.out_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim // hparams.channel_proj,
    )

    model = CoSDiModel(
        backbone, aggregator,
        lr=hparams.lr, lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        milestones=hparams.milestones,
        silent=hparams.silent,
    )

    if hparams.compile:
        model = torch.compile(model)

    datamodule = VPRDataModule(
        gsv_cities_path=hparams.gsv_cities_path,
        cities=hparams.cities,
        img_per_place=hparams.img_per_place,
        val_sets=hparams.val_sets,
        train_img_size=train_img_size,
        val_img_size=val_img_size,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=False,
    )

    if not hparams.silent:
        datamodule.setup()
        display_datasets_stats(datamodule)

    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"{hparams.backbone_name}_expA",         # ← 区分实验
        default_hp_metric=False
    )
    tensorboard_logger.log_hyperparams(hparams.__dict__)

    checkpointing = callbacks.ModelCheckpoint(
        monitor="msls-val/R@1",
        filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=8,
        mode="max",
    )

    program_bar = callbacks.RichProgressBar()
    callback_list = [checkpointing]
    if not hparams.silent:
        callback_list.append(program_bar)

    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        precision="16-mixed",
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
        gradient_clip_val=1.0,                        # ← 新增
    )

    trainer.fit(model=model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser(description="Train parameters")
    parser.add_argument("--dev",      action="store_true")
    parser.add_argument("--silent",   action="store_true")
    parser.add_argument('--compile',  action='store_true')
    parser.add_argument("--seed",   type=int)
    parser.add_argument("--bs",     type=int)
    parser.add_argument("--lr",     type=float)
    parser.add_argument("--wd",     type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument("--nw",     type=int)
    parser.add_argument('--backbone',   type=str)
    parser.add_argument('--unfreeze_n', type=int)
    parser.add_argument("--dim",        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()
    if args.seed:      hparams.seed = args.seed
    if args.compile:   hparams.compile = True
    if args.silent:    hparams.silent = True
    if args.bs:        hparams.batch_size = args.bs
    if args.lr:        hparams.lr = args.lr
    if args.wd:        hparams.weight_decay = args.wd
    if args.epochs:    hparams.max_epochs = args.epochs
    if args.warmup:    hparams.warmup_epochs = args.warmup
    if args.nw:        hparams.num_workers = args.nw
    if args.backbone:  hparams.backbone_name = args.backbone
    if args.unfreeze_n: hparams.unfreeze_n_blocks = args.unfreeze_n
    if args.dim:       hparams.output_dim = args.dim
    train(hparams, dev_mode=args.dev)
