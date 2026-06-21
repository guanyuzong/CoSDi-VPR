"""DataModule for University-1652 cross-view geo-localization."""

import torch
import lightning as L
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src.dataloaders.University1652Dataset import University1652Dataset
from src.dataloaders.University1652Val import University1652ValDataset


class U1652DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        img_per_place: int = 4,
        sat_ratio: float = 0.25,
        sample_num: int = 4,
        train_img_size=(224, 224),
        val_img_size=(224, 224),
        batch_size: int = 32,
        num_workers: int = 8,
        directions=("d2s", "s2d"),
        mean_std={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.img_per_place = img_per_place
        self.sat_ratio = sat_ratio
        self.sample_num = sample_num
        self.train_img_size = train_img_size
        self.val_img_size = val_img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.directions = directions
        self.mean_std = mean_std

        # Drone images: full augmentation including rotation
        self.train_transform_drone = T.Compose([
            T.Resize(train_img_size, interpolation=3),
            T.RandomRotation(180, interpolation=2),
            T.RandomResizedCrop(train_img_size, scale=(0.7, 1.0), interpolation=3),
            T.RandAugment(num_ops=2, magnitude=10, interpolation=2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])
        # Satellite: rotation is the most important augmentation; keep color jitter mild
        self.train_transform_sat = T.Compose([
            T.Resize(train_img_size, interpolation=3),
            T.RandomRotation(180, interpolation=2),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])
        self.val_transform = T.Compose([
            T.Resize(val_img_size, interpolation=3),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(**mean_std),
        ])

    def setup(self, stage=None):
        if stage in ("fit", "reload", None):
            self.train_dataset = University1652Dataset(
                dataset_path=self.train_path,
                img_per_place=self.img_per_place,
                sat_ratio=self.sat_ratio,
                sample_num=self.sample_num,
                transform_drone=self.train_transform_drone,
                transform_satellite=self.train_transform_sat,
            )
        if stage in ("fit", None):
            self.val_datasets = [
                University1652ValDataset(self.test_path, direction=d, transform=self.val_transform)
                for d in self.directions
            ]

    def train_dataloader(self):
        self.setup(stage="reload")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ds, batch_size=self.batch_size * self.img_per_place,
                num_workers=self.num_workers, pin_memory=True,
            ) for ds in self.val_datasets
        ]
