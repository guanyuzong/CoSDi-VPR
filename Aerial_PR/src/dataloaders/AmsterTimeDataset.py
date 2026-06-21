from typing import Optional, Callable, Tuple, Any
from pathlib import Path

import numpy as np
import torchvision
from torch.utils.data import Dataset


class AmsterTimeDataset(Dataset):
    """
    AmsterTime evaluation dataset for visual place recognition.

    Expected files in `dataset_path`:
      - amstertime_dbImages.npy
      - amstertime_qImages.npy
      - amstertime_gt.npy
      - image files referenced by db/query image list npy files
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.dataset_path = self._validate_path(dataset_path)
        self.dataset_name = self.dataset_path.name

        self.dbImages = np.load(self.dataset_path / "amstertime_dbImages.npy", allow_pickle=True)
        self.qImages = np.load(self.dataset_path / "amstertime_qImages.npy", allow_pickle=True)
        self.ground_truth = np.load(self.dataset_path / "amstertime_gt.npy", allow_pickle=True)

        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_rel_path = str(self.image_paths[index])
        img = torchvision.io.decode_image(self.dataset_path / img_rel_path, mode="RGB")

        if self.transform:
            img = self.transform(img)

        return img, index

    def __len__(self) -> int:
        return len(self.image_paths)

    def _validate_path(self, dataset_path):
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "data" / "amstertime"
        path = Path(dataset_path)

        if not path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist.")
        if not (path / "images" / "test" / "database").is_dir() or not (path / "images" / "test" / "queries").is_dir():
            raise FileNotFoundError(
                f"The directory {path} must contain `images/test/database` and `images/test/queries` folders."
            )

        required_files = ["amstertime_dbImages.npy", "amstertime_qImages.npy", "amstertime_gt.npy"]
        missing = [f for f in required_files if not (path / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing required AmsterTime metadata files in {dataset_path}: {missing}"
            )
        return path
