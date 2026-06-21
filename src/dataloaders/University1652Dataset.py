"""
University-1652 train dataset.

Folder layout (following layumi/University1652-Baseline):
    train/
        drone/<building_id>/image-XX.jpeg     (~54 imgs/building)
        satellite/<building_id>/<id>.jpg      (1 img/building)
        street/<building_id>/...              (unused here)

Sampling protocol: each "place" is one building. For every building we
return img_per_place images, mixing drone and satellite views, so MS loss
pulls cross-view samples of the same building together.
"""

from pathlib import Path
import random
import torch
import torchvision
from torch.utils.data import Dataset


class University1652Dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        img_per_place: int = 4,
        sat_ratio: float = 0.25,   # fraction of the K=img_per_place that should be satellite
        sample_num: int = 4,       # MCCG-style: each building sampled `sample_num` times per epoch
        transform_drone=None,
        transform_satellite=None,
    ):
        super().__init__()
        self.base = Path(dataset_path)
        assert (self.base / "drone").exists() and (self.base / "satellite").exists(), \
            f"Expected drone/ and satellite/ under {self.base}"

        self.img_per_place = img_per_place
        self.sat_ratio = sat_ratio
        self.sample_num = sample_num
        self.transform_drone = transform_drone
        self.transform_satellite = transform_satellite

        self.buildings = sorted([d.name for d in (self.base / "drone").iterdir() if d.is_dir()])
        self.bid_to_int = {bid: i for i, bid in enumerate(self.buildings)}

        # Pre-index image paths per building
        self.drone_imgs = {}
        self.sat_imgs = {}
        for bid in self.buildings:
            self.drone_imgs[bid] = list((self.base / "drone" / bid).glob("*.jp*g")) + \
                                   list((self.base / "drone" / bid).glob("*.png"))
            self.sat_imgs[bid] = list((self.base / "satellite" / bid).glob("*.jp*g")) + \
                                 list((self.base / "satellite" / bid).glob("*.png"))
            if not self.sat_imgs[bid] or not self.drone_imgs[bid]:
                raise RuntimeError(f"Empty drone/satellite folder for building {bid}")

        self.total_nb_images = sum(len(v) for v in self.drone_imgs.values()) + \
                               sum(len(v) for v in self.sat_imgs.values())
        # for compatibility with display_datasets_stats
        self.cities = self.buildings

    def __len__(self):
        # MCCG-style: each building is visited `sample_num` times per epoch,
        # with independent random K=img_per_place draws each visit.
        return len(self.buildings) * self.sample_num

    def _load(self, path, is_sat):
        img = torchvision.io.decode_image(str(path), mode=torchvision.io.ImageReadMode.RGB)
        t = self.transform_satellite if is_sat else self.transform_drone
        return t(img) if t is not None else img

    def __getitem__(self, idx):
        bid = self.buildings[idx % len(self.buildings)]
        K = self.img_per_place
        n_sat = max(1, round(K * self.sat_ratio))     # at least one satellite
        n_drone = K - n_sat

        sat_pool, drone_pool = self.sat_imgs[bid], self.drone_imgs[bid]
        sat_picks = [random.choice(sat_pool) for _ in range(n_sat)]
        drone_picks = random.sample(drone_pool, k=min(n_drone, len(drone_pool)))
        while len(drone_picks) < n_drone:
            drone_picks.append(random.choice(drone_pool))

        imgs = [self._load(p, is_sat=True) for p in sat_picks] + \
               [self._load(p, is_sat=False) for p in drone_picks]

        label = self.bid_to_int[bid]
        return torch.stack(imgs), torch.tensor(label).repeat(K)

    def _refresh_dataframes(self):
        # called by model.on_train_epoch_end; nothing to do here
        pass
