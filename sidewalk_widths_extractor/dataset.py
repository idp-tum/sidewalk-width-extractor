import os
from glob import glob
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from sidewalk_widths_extractor.typing import _PATH


class SatelliteDataset(Dataset):
    """Satelite Dataset."""

    def __init__(
        self,
        images_path: _PATH,
        masks_path: _PATH,
        transform: Optional[Any] = None,
        indices: Optional[List[int]] = None,
    ) -> None:
        """Satelite Dataset.

        Args:
            images_path: path to the folder containing images.
            masks_path: path to the folder containing masks.
            transform (optional): transform function instance.
            indices (optional): specific indices to-be-used for images and masks.
        """
        assert os.path.exists(images_path)
        assert os.path.exists(masks_path)

        self._images_path = images_path
        self._masks_path = masks_path
        self._transform = transform

        self._images = np.array(sorted(glob(os.path.join(self._images_path, "*"))))
        self._masks = np.array(sorted(glob(os.path.join(self._masks_path, "*"))))

        if indices:
            self._images = self._images.take(indices)
            self._masks = self._masks.take(indices)

        self._transform = transform
        self._tensor_transform = T.ToTensor()

        assert len(self._images) == len(self._masks)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: the number of samples.
        """
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the sample in the given index.

        Args:
            idx (int): a sample index.

        Returns:
            Any: the sample.
        """
        image = cv2.imread(self._images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self._masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 255, 1, 0)

        if self._transform:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = self._tensor_transform(image)
        mask = self._tensor_transform(mask)

        return (image, mask)

    @classmethod
    def from_split(
        cls,
        images_path: _PATH,
        masks_path: _PATH,
        split_ratio: float = 0.8,
        train_transform: Optional[Any] = None,
        val_transform: Optional[Any] = None,
        random_seed: int = 42,
    ) -> Tuple[Any, Any]:
        """
        Build a train/val split.

        Args:
            mages_path: path to the folder containing images.
            masks_path: path to the folder containing masks.
            split_ratio: ratio of training dataset over all.
                Defaults to 0.8.
            train_transform (optional): transform function for train dataset.
                Defaults to None.
            val_transform (optional): transform function for validation dataset.
                Defaults to None.
            random_seed: random number generator seed.
                Defaults to None.

        Returns:
            tuple: training dataset and validation dataset
        """
        assert split_ratio > 0.0 and split_ratio < 1.0
        all_indices = range(len(glob(os.path.join(images_path, "*"))))
        assert len(all_indices) != 0

        train_indices, val_indices = train_test_split(
            all_indices, test_size=1 - split_ratio, random_state=random_seed
        )

        train_dataset = cls(images_path, masks_path, train_transform, train_indices)
        val_dataset = cls(images_path, masks_path, val_transform, val_indices)

        return (train_dataset, val_dataset)
