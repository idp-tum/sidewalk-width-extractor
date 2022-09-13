import os
from typing_extensions import Literal
from typing import Any, Dict, Optional, Union

import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, ReduceLROnPlateau, StepLR
from torchmetrics.functional import stat_scores

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


class SegModule(BaseModule):
    def __init__(
        self,
        network_id: Union[Literal["unet"], Literal["unet++"]],
        network_params: Dict[str, Any],
        optimizer_id: Union[Literal["adam"], Literal["sgd"]],
        optimizer_params: Dict[str, Any],
        criterion_id: Union[Literal["ce"], Literal["wce"]],
        criterion_params: Dict[str, Any],
        scheduler_id: Optional[
            Union[
                Literal["reducelronplateau"],
                Literal["lambdalr"],
                Literal["steplr"],
                Literal["exponentiallr"],
            ]
        ] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        device: Optional[_DEVICE] = None,
        save_network_checkpoint: bool = True,
        save_optimizer_checkpoint: bool = True,
    ):
        super().__init__(device)

        self._network_id = network_id
        self._network_params = network_params
        self._optimizer_id = optimizer_id
        self._optimizer_params = optimizer_params
        self._criterion_id = criterion_id
        self._criterion_params = criterion_params
        self._scheduler_id = scheduler_id
        self._scheduler_params = scheduler_params

        self._save_network_checkpoint = save_network_checkpoint
        self._save_optimizer_checkpoint = save_optimizer_checkpoint

        if self._network_id == "unet":
            self._network = smp.Unet(**network_params)
        elif self._network_id == "unet++":
            self._network = smp.UnetPlusPlus(**network_params)
        else:
            raise Exception("invaild network id")

        if self._optimizer_id == "adam":
            self._optimizer = torch.optim.Adam(
                params=self._network.parameters(), **self._optimizer_params
            )
        elif self._optimizer == "sgd":
            self._optimizer = torch.optim.SGD(
                params=self._network.parameters(), **self._optimizer_params
            )
        else:
            raise Exception("invaild optimizer id")

        if self._criterion_id == "wce":
            weight = torch.Tensor(criterion_params["weight"])
            self._criterion = torch.nn.CrossEntropyLoss(weight=weight)
        elif self._criterion_id == "ce":
            self._criterion = torch.nn.CrossEntropyLoss()
        else:
            raise Exception("invaild criterion id")

        if self._scheduler_id and isinstance(self._scheduler_params, dict):
            if self._scheduler_id == "reducelronplateau":
                self._scheduler = ReduceLROnPlateau(self._optimizer, **self._scheduler_params)
            elif self._scheduler_id == "steplr":
                self._scheduler = StepLR(self._optimizer, **self._scheduler_params)
            elif self._scheduler_id == "exponentiallr":
                self._scheduler = ExponentialLR(self._optimizer, **self._scheduler_params)
            elif self._scheduler_id == "lambdalr":
                self._scheduler = LambdaLR(self._optimizer, **self._scheduler_params)
            else:
                raise Exception("invaild scheduler id")

        self._network.to(self.device)
        self._criterion.to(self.device)

    def get_settings(self) -> Dict[str, Any]:
        return {
            "network": {"id": self._network_id, "params": self._network_params},
            "optimizer": {"id": self._optimizer_id, "params": self._optimizer_params},
            "criterion": {"id": self._criterion_id, "params": self._criterion_params},
            "scheduler": {"id": self._scheduler_id, "params": self._scheduler_params},
        }

    def train_step(self, batch: _BATCH, step_idx: int, epoch_idx: int) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device).squeeze(1).long()

        out = self._network(images)
        loss = self._criterion(out, masks)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        preds = torch.argmax(out, dim=1)

        if epoch_idx and epoch_idx % 5 == 0 and step_idx == 0:
            self.log_stacked_segments_img(images, preds, masks, "train", epoch_idx)

        return {"loss": loss}

    def validate_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device).squeeze(1).long()

        out = self._network(images)
        loss = self._criterion(out, masks)
        preds = torch.argmax(out, dim=1)

        stats = stat_scores(preds, masks, reduce="macro", mdmc_reduce="global", num_classes=2)[1]

        if epoch_idx and epoch_idx % 5 == 0 and step_idx == 0:
            self.log_stacked_segments_img(images, preds, masks, "val", epoch_idx)

        return {
            "loss": loss,
            "tp": stats[0],
            "fp": stats[1],
            "tn": stats[2],
            "fn": stats[3],
        }

    def test_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        images = batch[0].to(self.device)
        masks = batch[1].to(self.device).squeeze(1).long()

        out = self._network(images)
        loss = self._criterion(out, masks)
        preds = torch.argmax(out, dim=1)

        stats = stat_scores(preds, masks, reduce="macro", mdmc_reduce="global", num_classes=2)[1]

        return {
            "loss": loss,
            "tp": stats[0],
            "fp": stats[1],
            "tn": stats[2],
            "fn": stats[3],
        }

    def infer(self, x: Any) -> Any:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        out = self._network(x.to(self.device))
        preds = torch.argmax(out, dim=1)
        return preds

    def save(
        self,
        epoch_idx: Optional[int] = None,
        target_path: Optional[_PATH] = None,
        include_network: Optional[bool] = None,
        include_optimizer: Optional[bool] = None,
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

        assert self.trained or self.resumed

        if include_network is None:
            include_network = self._save_network_checkpoint
        if include_optimizer is None:
            include_optimizer = self._save_optimizer_checkpoint

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

    def load(self, checkpoint_path: Dict[str, _PATH]) -> Optional[int]:
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
                self.curr_epoch_idx = epoch_optimizer
        if load_network:
            epoch_network = load_checkpoint(checkpoint_path["network"], self._network)
            if epoch_network:
                epoch = epoch_network
                self.curr_epoch_idx = epoch_network
        return epoch

    def on_train_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        if epoch_idx and self.writer:  # only log during the training
            self.writer.add_scalar("train/lr", self._optimizer.param_groups[0]["lr"], epoch_idx)
            for _, metric in epoch_results.items():
                value = metric.compute("mean")
                self.writer.add_scalar(metric.identifier, value.item(), epoch_idx)

    def on_val_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        if epoch_idx and self.writer:  # only log during the training
            if "loss" in epoch_results:
                metric = epoch_results["loss"]
                self.writer.add_scalar(metric.identifier, metric.compute("mean").item(), epoch_idx)
            if "tp" in epoch_results:
                tp = epoch_results["tp"].compute("sum")
                fp = epoch_results["fp"].compute("sum")
                fn = epoch_results["fn"].compute("sum")
                tn = epoch_results["tn"].compute("sum")

                accuracy = (tp + tn) / (tp + tn + fp + fn)
                self.writer.add_scalar("val/accuracy", accuracy.item(), epoch_idx)

                if not tp + fp == 0:
                    precision = tp / (tp + fp)
                else:
                    precision = torch.tensor(0.0)

                self.writer.add_scalar("val/precision", precision.item(), epoch_idx)

                recall = tp / (tp + fn)
                self.writer.add_scalar("val/recall", recall.item(), epoch_idx)

                iou = tp / (tp + fn + fp)
                self.writer.add_scalar("val/iou", iou.item(), epoch_idx)

                dice = 2 * tp / (2 * tp + fn + fp)
                self.writer.add_scalar("val/dice", dice.item(), epoch_idx)

                if self._scheduler:
                    self._scheduler.step(dice)

    def log_stacked_segments_img(
        self, img: torch.Tensor, p: torch.Tensor, t: torch.Tensor, category: str, current_epoch: int
    ) -> None:
        if self.writer:
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
