import os
from typing import Any, Dict, Optional, Union

import torch

from sidewalk_widths_extractor.models.unet import UNet
from sidewalk_widths_extractor.modules import BaseModule
from sidewalk_widths_extractor.typing import (
    _BATCH,
    _DEVICE,
    _EPOCH_RESULT,
    _PATH,
    _STEP_RESULT,
)
from sidewalk_widths_extractor.utilities.io import load_checkpoint, save_checkpoint
from sidewalk_widths_extractor.utilities.visuals import create_stacked_segments


class TestModule(BaseModule):
    def __init__(
        self,
        optimizer_params: Dict[str, Any],
        device: _DEVICE,
    ):
        super().__init__(device)

        self._optimizer_params = optimizer_params
        self._network = UNet(
            encChannels=(3, 16, 32, 64, 128), decChannels=(128, 64, 32, 16), num_classes=2
        ).to(self.device)
        self._optimizer = torch.optim.Adam(
            params=self._network.parameters(), **self._optimizer_params
        )
        self._criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.1, 1])).to(self.device)

    def train_step(self, batch: _BATCH, step_idx: int, epoch_idx: int) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device)

        out = self._network(images)
        loss = self._criterion(out, masks.squeeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        preds = torch.argmax(out, dim=1)

        if step_idx == 0:
            self.log_stacked_segments_img(images, preds, masks, "train", epoch_idx)

        return {"loss": loss}

    def validate_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device).squeeze(1)

        out = self._network(images)
        loss = self._criterion(out, masks)
        preds = torch.argmax(out, dim=1)

        if epoch_idx and step_idx == 0:
            self.log_stacked_segments_img(images, preds, masks, "val", epoch_idx)

        return {
            "loss": loss,
        }

    def test_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device).squeeze(1)

        out = self._network(images)
        loss = self._criterion(out, masks)

        return {
            "loss": loss,
        }

    def infer(self, x: Any) -> Any:
        out = self._network(x)
        preds = torch.argmax(out, dim=1)
        return preds

    def save(
        self,
        epoch_idx: Optional[int] = None,
        target_path: Optional[_PATH] = None,
        include_network: bool = True,
        include_optimizer: bool = True,
        network_filename: str = "network",
        optimizer_filename: str = "optimizer",
        file_format: str = ".pth.tar",
    ) -> None:
        """
        Save the network or optimizer weights to checkpoint file(s).

        Args:
            epoch_idx (optional): The current epoch number. Defaults to None.
            target (optional): The target folder path. Defaults to None.
            include_network: Whether to save the network weights. Defaults to True.
            include_optimizer: Whether to save the optimizer weights. Defaults to True.
            network_filename: The network weights file name. Defaults to "network".
            optimizer_filename: The optimizer weights file name. Defaults to  "optimizer".
        """
        assert include_network or include_optimizer
        assert self.trained or self.resumed

        target_folder = (
            target_path
            if target_path
            else os.path.join(self.checkpoint_folder_path, str(epoch_idx))
            if epoch_idx
            else self.checkpoint_folder_path
        )

        if include_network:
            save_checkpoint(target_folder, self._network, epoch_idx, network_filename, file_format)
        if include_optimizer:
            save_checkpoint(
                target_folder, self._optimizer, epoch_idx, optimizer_filename, file_format
            )

    def load(self, checkpoint_path: Union[Dict[str, _PATH], _PATH]) -> Optional[int]:
        """
        Load weights.

        Args:
            checkpoint_path (bool): checkpoint_paths. Keys are 'network' and 'optimizer'.

        Returns:
            int: the last epoch idx if retrieve_epoch_idx is true, otherwise None.
        """

        load_network = "network" in checkpoint_path
        load_optimizer = "optimizer" in checkpoint_path

        assert not load_network or os.path.exists(checkpoint_path["network"])
        assert not load_optimizer or os.path.exists(checkpoint_path["optimizer"])

        epoch = None

        if load_optimizer:
            epoch_optimizer = load_checkpoint(checkpoint_path["optimizer"], self._optimizer)
            if epoch_optimizer:
                epoch = epoch_optimizer
        if load_network:
            epoch_network = load_checkpoint(checkpoint_path["network"], self._network)
            if epoch_network:
                epoch = epoch_network

        return epoch

    def on_train_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        if epoch_idx:  # log during the training
            for name, metric in epoch_results.items():
                value = metric.compute("mean")
                self.writer.add_scalar(metric.identifier, value.item(), epoch_idx)

    def on_val_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        if epoch_idx:  # log during the training
            for name, metric in epoch_results.items():
                self.writer.add_scalar(metric.identifier, metric.compute("mean").item(), epoch_idx)

    def log_stacked_segments_img(
        self, img: torch.Tensor, p: torch.Tensor, t: torch.Tensor, category: str, current_epoch: int
    ) -> None:
        if self.writer is not None:
            self.writer.add_image(
                f"{category}/all",
                create_stacked_segments(
                    img.detach().cpu(),
                    p.detach().cpu(),
                    t.detach().cpu(),
                    min(p.shape[0], 6),
                ),
                current_epoch,
            )
