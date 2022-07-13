from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, Literal["cpu"]]
_STEP_RESULT = Dict[str, Any]
_EPOCH_RESULT = Dict[str, Any]
_BATCH = Tuple[torch.Tensor, torch.Tensor]
