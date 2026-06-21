from typing import Optional, Callable, Tuple, Any
import numpy as np
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.io
from torch.utils.data import Dataset


class BaiduDataset(Dataset):
    """
    Baidu Mall indoor VPR dataset (CVPR 2017).

    Expected files in `dataset_path` (run prepare_baidu_dataset.py first):
      - baidu_dbImages.npy   : relative paths of 689 database images
      - baidu_qImages.npy    : relative paths of 2292 query images
      - baidu_gt.npy         : ground truth (list of lists, indices into db)

    Reference:
        @InProceedings{Sun_2017_CVPR,
            author = {Sun, Xun and Xie, Yuanfan and Luo, Pei and Wang, Liang},
            title = {A Dataset for Benchmarking Image-Based Localization},
            booktitle = {CVPR},
            year = {2017}
        }
    """

    REQUIRED_FILES = ["baidu_dbImages.npy", "baidu_qImages.npy", "baidu_gt.npy"]

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.dataset_path = self._validate_path(dataset_path)
        self.dataset_name = "baidu"

        self.dbImages   = np.load(self.dataset_path / "baidu_dbImages.npy", allow_pickle=True)
        self.qImages    = np.load(self.dataset_path / "baidu_qImages.npy",  allow_pickle=True)
        self.ground_truth = np.load(self.dataset_path / "baidu_gt.npy",     allow_pickle=True)

        self.image_paths   = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries    = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = str(self.image_paths[index])
        full_path = str(self.dataset_path / img_path)
        data = torch.from_numpy(np.frombuffer(open(full_path, 'rb').read(), dtype=np.uint8).copy())
        img = torchvision.io.decode_image(data, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)

    def _validate_path(self, dataset_path):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "baidu"
        path = Path(dataset_path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Directory {dataset_path} does not exist. "
                "Run prepare_baidu_dataset.py first."
            )
        missing = [f for f in self.REQUIRED_FILES if not (path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing metadata files in {dataset_path}: {missing}. "
                "Run prepare_baidu_dataset.py first."
            )
        return path
