"""
University-1652 validation dataset.

Wraps the cross-view retrieval test (drone -> satellite by default) in the
[references; queries] flat format expected by utils.compute_recall_performance.

Direction:
    "d2s": queries = query_drone,     references = gallery_satellite   (default)
    "s2d": queries = query_satellite, references = gallery_drone
"""

from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset


def _list_imgs(folder: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    out = []
    for ext in exts:
        out.extend(folder.glob(ext))
    return sorted(out)


class University1652ValDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        direction: str = "d2s",
        transform=None,
    ):
        super().__init__()
        base = Path(dataset_path)
        assert direction in ("d2s", "s2d")
        self.direction = direction
        self.dataset_name = f"u1652-{direction}"
        self.transform = transform

        if direction == "d2s":
            q_root = base / "query_drone"
            g_root = base / "gallery_satellite"
        else:
            q_root = base / "query_satellite"
            g_root = base / "gallery_drone"
        for r in (q_root, g_root):
            assert r.exists(), f"Missing {r}"

        self.gallery_paths, self.gallery_ids = self._collect(g_root)
        self.query_paths,   self.query_ids   = self._collect(q_root)

        self.num_references = len(self.gallery_paths)
        self.num_queries = len(self.query_paths)

        # ground_truth: for each query, list of gallery indices sharing the building id
        gid_to_indices = {}
        for i, bid in enumerate(self.gallery_ids):
            gid_to_indices.setdefault(bid, []).append(i)
        self.ground_truth = [gid_to_indices.get(qid, []) for qid in self.query_ids]

        # Drop queries that have no positive in the gallery (e.g. some unseen
        # buildings only appear in gallery_drone). For seen-class evaluation
        # protocol of MCCG/FSRA, query buildings are always covered.
        keep = [i for i, gt in enumerate(self.ground_truth) if len(gt) > 0]
        if len(keep) != len(self.query_paths):
            self.query_paths = [self.query_paths[i] for i in keep]
            self.query_ids = [self.query_ids[i] for i in keep]
            self.ground_truth = [self.ground_truth[i] for i in keep]
            self.num_queries = len(self.query_paths)

        # flat ordering: [gallery..., queries...]
        self.images = self.gallery_paths + self.query_paths

    @staticmethod
    def _collect(root: Path):
        paths, ids = [], []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            for p in _list_imgs(d):
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
