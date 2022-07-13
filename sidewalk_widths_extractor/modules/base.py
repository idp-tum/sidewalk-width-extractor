from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

from sidewalk_widths_extractor.typing import (
    _BATCH,
    _DEVICE,
    _EPOCH_RESULT,
    _PATH,
    _STEP_RESULT,
)
from sidewalk_widths_extractor.utilities import get_device


class BaseModule(ABC):
    """
    Base Deep Learning Module.

    Attributes:
        device: associated device.
        writer: Tensorboard writer.
        checkpoint_folder_path: path to the checkpoints.
        log_path: path to the log folder.
        trained: whether the module is trained or not.
        resumed: whether the module is resumed or not.
        curr_epoch_idx: current epoch index.
    """

    def __init__(self, device: Optional[_DEVICE] = None):
        """
        Base Deep Learning Module.

        Args:
            device (optional): the associated device. If None, it will be automatically assigned.
                Defaults to None.
        """
        super().__init__()
        self.device = device if device else get_device()
        self.writer: Optional[SummaryWriter] = None
        self.checkpoint_folder_path = None
        self.log_path = None
        self.trained: bool = False
        self.resumed: bool = False
        self.curr_epoch_idx: Optional[int] = None

    @abstractmethod
    def train_step(self, batch: _BATCH, step_idx: int, epoch_idx: int) -> _STEP_RESULT:
        pass

    @abstractmethod
    def validate_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        pass

    @abstractmethod
    def test_step(
        self, batch: _BATCH, step_idx: int, epoch_idx: Optional[int] = None
    ) -> _STEP_RESULT:
        pass

    @abstractmethod
    def infer(self, x: Any) -> Any:
        pass

    @abstractmethod
    def save(
        self, epoch_idx: Optional[int] = None, *args: Tuple[Any], **kwargs: Dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    def load(
        self,
        checkpoint_path: Union[Dict[str, _PATH], _PATH],
        *args: Tuple[Any],
        **kwargs: Dict[str, Any]
    ) -> Optional[int]:
        pass

    def get_settings(self) -> Dict[str, Any]:
        return {}

    def on_start(self) -> None:
        pass

    def on_end(self) -> None:
        pass

    def on_train_epoch_start(self, epoch_idx: Optional[int] = None) -> None:
        pass

    def on_train_step_start(
        self, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None
    ) -> None:
        pass

    def on_train_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        pass

    def on_train_step_end(
        self,
        batch_results: _STEP_RESULT,
        epoch_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> None:
        pass

    def on_val_epoch_start(self, epoch_idx: Optional[int] = None) -> None:
        pass

    def on_val_step_start(
        self, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None
    ) -> None:
        pass

    def on_val_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        pass

    def on_val_step_end(
        self,
        batch_results: _STEP_RESULT,
        epoch_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> None:
        pass

    def on_test_epoch_start(self, epoch_idx: Optional[int] = None) -> None:
        pass

    def on_test_step_start(
        self, epoch_idx: Optional[int] = None, step_idx: Optional[int] = None
    ) -> None:
        pass

    def on_test_epoch_end(
        self, epoch_results: _EPOCH_RESULT, epoch_idx: Optional[int] = None
    ) -> None:
        pass

    def on_test_step_end(
        self,
        batch_results: _STEP_RESULT,
        epoch_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
    ) -> None:
        pass
