from typing import Any, Callable, Optional, Union

import torch
from typing_extensions import Literal

from sidewalk_widths_extractor.utilities.enums import Category


class Metric:
    def __init__(
        self,
        name: str,
        category: Category,
        value: Optional[torch.Tensor] = None,
        transfer_to_cpu: bool = False,
    ) -> None:
        self.name = name
        self.category = category
        self.identifier = category + "/" + name
        self.transfer_to_cpu = transfer_to_cpu
        if value is not None:
            if self.transfer_to_cpu:
                self.values = [value.cpu()]
            else:
                self.values = [value]

        else:
            self.values = []

    def __eq__(self, other) -> bool:
        return isinstance(other, Metric) and self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __len__(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        return self.identifier

    def update(self, value: torch.Tensor) -> None:
        if self.transfer_to_cpu:
            self.values.append(value.cpu())
        else:
            self.values.append(value)

    def compute(
        self,
        reduction: Union[Literal["mean"], Literal["sum"], Literal["min"], Literal["max"], Callable],
    ) -> Any:
        if reduction == "mean":
            return torch.mean(torch.stack(self.values), dim=0)
        elif reduction == "sum":
            return torch.sum(torch.stack(self.values), dim=0)
        elif reduction == "min":
            return torch.min(torch.stack(self.values), dim=0).values
        elif reduction == "max":
            return torch.max(torch.stack(self.values), dim=0).values
        elif callable(reduction):
            return reduction(torch.stack(self.values))
        else:
            raise ValueError("Invalid reduction type given")
