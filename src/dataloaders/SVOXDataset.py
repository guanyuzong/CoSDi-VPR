from typing import Optional, Callable, Tuple, Any
import numpy as np
import torch
import torchvision
import torchvision.io
from pathlib import Path
from torch.utils.data import Dataset

CONDITIONS = ["overcast", "night", "sun", "rain", "snow"]


class SVOXDataset(Dataset):
    """
    SVOX + RobotCar dataset for VPR evaluation (WACV 2021).
    Gallery: 17166 SVOX StreetView images.
    Queries: one of 5 RobotCar weather conditions (night/overcast/rain/snow/sun).

    Run prepare_svox_dataset.py first to generate .npy metadata files.

    Reference:
        Berton et al., "Adaptive-Attentive Geolocalization from Few Queries", WACV 2021.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        condition: str = "night",
        transform: Optional[Callable] = None,
    ):
        assert condition in CONDITIONS, \
            f"condition must be one of {CONDITIONS}, got '{condition}'"

        self.transform = transform
        self.condition = condition
        self.dataset_name = f"svox-{condition}"
        self.dataset_path = self._validate_path(dataset_path, condition)

        self.dbImages = np.load(self.dataset_path / "svox_dbImages.npy", allow_pickle=True)
        self.qImages  = np.load(self.dataset_path / f"svox_{condition}_qImages.npy", allow_pickle=True)
        self.ground_truth = np.load(self.dataset_path / f"svox_{condition}_gt.npy", allow_pickle=True)

        self.image_paths    = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries    = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        full_path = str(self.dataset_path / self.image_paths[index])
        data = torch.from_numpy(
            np.frombuffer(open(full_path, "rb").read(), dtype=np.uint8).copy()
        )
        img = torchvision.io.decode_image(data, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)

    def _validate_path(self, dataset_path, condition):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "svox"
        path = Path(dataset_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Directory {dataset_path} does not exist. "
                "Run prepare_svox_dataset.py first."
            )
        required = [
            "svox_dbImages.npy",
            f"svox_{condition}_qImages.npy",
            f"svox_{condition}_gt.npy",
        ]
        missing = [f for f in required if not (path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing metadata in {dataset_path}: {missing}. "
                "Run prepare_svox_dataset.py first."
            )
        return path
