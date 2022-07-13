import os
from glob import glob
from typing import Any, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from sidewalk_widths_extractor.typing import _PATH


class SateliteDataset(Dataset):
    r"""TODO description."""

    def __init__(
        self,
        images_path: _PATH,
        masks_path: _PATH,
        transform: Optional[Any] = None,
    ) -> None:
        """TODO description.

        Args:
            resize (Tuple[int,int]) : to-be-changed input image dimensions.
        """
        assert os.path.exists(images_path)
        assert os.path.exists(masks_path)

        self._images_path = images_path
        self._masks_path = masks_path
        self._transform = transform

        self._images = sorted(glob(os.path.join(self._images_path, "*")))
        self._masks = sorted(glob(os.path.join(self._masks_path, "*")))

        self.transform_img = T.Compose(
            [
                # T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )
        self.transform_mask = T.PILToTensor()
        self.to_tensor = T.PILToTensor()

        assert len(self._images) == len(self._masks)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: the number of samples.
        """
        return len(self._images)

    def __getitem__(self, idx: int) -> Any:
        """Return the sample in the given index.

        Args:
            idx (int): a sample index.

        Returns:
            Any: the sample.
        """
        image = Image.open(self._images[idx])
        mask = Image.open(self._masks[idx])

        image = self.transform_img(image)
        mask = self.to_tensor(mask)

        mask = torch.where(mask == 255, 1, 0)

        if self._transform:
            image = self._transform(image)
            # mask = self._transform(mask)

        return image, mask
