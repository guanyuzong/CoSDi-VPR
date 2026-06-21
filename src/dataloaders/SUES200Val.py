"""
SUES-200 validation dataset.

Layout expected:
    <root>/satellite-view/<loc_id>/0.png
    <root>/drone_view_512/<loc_id>/<height>/{0..49}.jpg
where <loc_id> is a 4-digit string in 0001..0200 and <height> in {150,200,250,300}.

Protocol (MCCG paper, Tab. II):
    Train locations: ids 1..120
    Test  locations: ids 121..200  (80 locs)

For a given height h:
    d2s (Drone -> Satellite):
        query   = drone imgs of test locs at height h          (80 * 50 = 4000)
        gallery = satellite imgs of ALL 200 locs (120 train act as distractors)
    s2d (Satellite -> Drone):
        query   = satellite imgs of test locs                  (80)
        gallery = drone imgs of ALL 200 locs at height h       (200 * 50 = 10000)
"""

from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset


_IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _list_imgs(folder: Path):
    out = []
    for ext in _IMG_EXTS:
        out.extend(folder.glob(ext))
    return sorted(out)


def _default_split(num_total: int = 200, num_train: int = 120):
    train_ids = {f"{i:04d}" for i in range(1, num_train + 1)}
    test_ids  = {f"{i:04d}" for i in range(num_train + 1, num_total + 1)}
    return train_ids, test_ids


class SUES200ValDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        height: int,
        direction: str = "d2s",
        transform=None,
        num_train: int = 120,
        num_total: int = 200,
    ):
        super().__init__()
        assert direction in ("d2s", "s2d")
        assert height in (150, 200, 250, 300)
        self.direction = direction
        self.height = height
        self.dataset_name = f"sues200-h{height}-{direction}"
        self.transform = transform

        base = Path(dataset_path)
        sat_root   = base / "satellite-view"
        drone_root = base / "drone_view_512"
        for r in (sat_root, drone_root):
            assert r.exists(), f"Missing {r}"

        train_ids, test_ids = _default_split(num_total, num_train)

        if direction == "d2s":
            # gallery = all 200 sat; query = test locs drone at this height
            self.gallery_paths, self.gallery_ids = self._collect_sat(sat_root, train_ids | test_ids)
            self.query_paths,   self.query_ids   = self._collect_drone(drone_root, test_ids, height)
        else:  # s2d
            # gallery = all 200 drone at this height; query = test locs sat
            self.gallery_paths, self.gallery_ids = self._collect_drone(drone_root, train_ids | test_ids, height)
            self.query_paths,   self.query_ids   = self._collect_sat(sat_root, test_ids)

        self.num_references = len(self.gallery_paths)
        self.num_queries    = len(self.query_paths)

        gid_to_indices = {}
        for i, bid in enumerate(self.gallery_ids):
            gid_to_indices.setdefault(bid, []).append(i)
        self.ground_truth = [gid_to_indices.get(qid, []) for qid in self.query_ids]

        keep = [i for i, gt in enumerate(self.ground_truth) if len(gt) > 0]
        if len(keep) != len(self.query_paths):
            self.query_paths = [self.query_paths[i] for i in keep]
            self.query_ids   = [self.query_ids[i]   for i in keep]
            self.ground_truth = [self.ground_truth[i] for i in keep]
            self.num_queries = len(self.query_paths)

        self.images = self.gallery_paths + self.query_paths

    @staticmethod
    def _collect_sat(sat_root: Path, keep_ids: set):
        paths, ids = [], []
        for d in sorted(sat_root.iterdir()):
            if not d.is_dir() or d.name not in keep_ids:
                continue
            for p in _list_imgs(d):
                paths.append(p)
                ids.append(d.name)
        return paths, ids

    @staticmethod
    def _collect_drone(drone_root: Path, keep_ids: set, height: int):
        paths, ids = [], []
        h_name = str(height)
        for d in sorted(drone_root.iterdir()):
            if not d.is_dir() or d.name not in keep_ids:
                continue
            h_dir = d / h_name
            if not h_dir.exists():
                continue
            for p in _list_imgs(h_dir):
                paths.append(p)
                ids.append(d.name)
        return paths, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p = self.images[idx]
        img = torchvision.io.decode_image(str(p), mode=torchvision.io.ImageReadMode.RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, idx
